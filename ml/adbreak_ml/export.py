"""Export the model for Rust (`rust/crates/adbreak-model`): `model.json` and a test vector.

The exported model takes the inputs adbreak-model's `Row` has (picture embedding, caption timing, black and silence;
no caption text, which needs a tokenizer in Rust and is worth under a point of F1) and computes the table exactly
as `adbreak_model::Model::table` does. Training is as in `fusion.py`: the trees learn from out-of-fold frame
probabilities, and from each captioned recording a second time with its captions removed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .data import BREAK, IGNORE, Recording
from .embed import NAME as EMBED, MODEL as IMAGE_MODEL
from .fusion import WINDOWS, Data, window_mean

FORMAT = 1
EDGE_CAP = 120.0
POST = {"threshold": 0.35, "min_len": 30.0, "max_gap": 25.0, "snap": 20.0}


def feature_names() -> list[str]:
    v = ["p", "has_captions", "visible", "roll_up", "new_cue", "black", "silent", "edge_dist", "edge_here"]
    for w in WINDOWS:
        v += [f"p_w{w}", f"visible_w{w}", f"roll_up_w{w}", f"edges_w{w}"]
    return v + ["p_std15"]


def inputs(d: Data, drop_captions: bool = False):
    """Per-second (has, visible, roll_up, new, black, silent) and the edges, as adbreak-model's Row and Inputs."""
    cap = np.zeros_like(d.cap) if drop_captions else d.cap
    edges = [float(e) for e in d.edges if e > -1e8]
    return cap[:, 0], cap[:, 1], cap[:, 2], cap[:, 3], d.av[:, 0], d.av[:, 1], edges


def table(d: Data, p: np.ndarray, drop_captions: bool = False) -> np.ndarray:
    has, vis, roll, new, black, silent, edges = inputs(d, drop_captions)
    t = np.arange(len(p), dtype=np.float64)
    if edges:
        e = np.sort(np.array(edges))
        j = np.searchsorted(e, t)
        dd = np.minimum(np.abs(t - e[np.clip(j - 1, 0, len(e) - 1)]), np.abs(t - e[np.clip(j, 0, len(e) - 1)]))
    else:
        dd = np.full(len(t), np.inf)
    dist, here = np.minimum(dd, EDGE_CAP) / EDGE_CAP, (dd < 1.0).astype(np.float64)
    cols = [p, has, vis, roll, new, black, silent, dist, here]
    for w in WINDOWS:
        cols += [window_mean(p, w), window_mean(vis, w), window_mean(roll, w), window_mean(here, w) * (2 * w + 1)]
    m1, m2 = window_mean(p, 15), window_mean(p * p, 15)
    cols.append(np.sqrt(np.maximum(m2 - m1 * m1, 0)))
    # Rounded to float32: Rust and NumPy can differ in the last bit of float64 arithmetic, which would send a value
    # sitting on a tree threshold the other way.
    return np.stack([np.asarray(c, np.float64) for c in cols], 1).astype(np.float32).astype(np.float64)


def frame_model(train: list[Data]) -> LogisticRegression:
    X = np.concatenate([d.x[d.y != IGNORE] for d in train])
    Y = np.concatenate([d.y[d.y != IGNORE] == BREAK for d in train])
    return LogisticRegression(C=1.0, max_iter=3000).fit(X, Y)


def fit_trees(train: list[Data], oof: dict) -> HistGradientBoostingClassifier:
    views = [(o, False) for o in train] + [(o, True) for o in train if o.cap[:, 0].any()]
    X = np.concatenate([table(o, oof[o.rec.id], drop)[o.y != IGNORE] for o, drop in views])
    Y = np.concatenate([o.y[o.y != IGNORE] == BREAK for o, _ in views])
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=0).fit(X, Y)


def trees_json(clf: HistGradientBoostingClassifier) -> tuple[float, list]:
    out = []
    for it in clf._predictors:
        nodes = it[0].nodes
        out.append([
            {"leaf": True, "v": float(n["value"])} if n["is_leaf"] else
            {"f": int(n["feature_idx"]), "t": float(n["num_threshold"]), "l": int(n["left"]), "r": int(n["right"]),
             "missing_left": bool(n["missing_go_to_left"])}
            for n in nodes])
    return float(np.ravel(clf._baseline_prediction)[0]), out


def to_breaks(p: np.ndarray, edges: list[float]) -> list[tuple[float, float]]:
    """As adbreak_model::to_breaks."""
    spans, start = [], None
    for i, v in enumerate(p):
        if v >= POST["threshold"] and start is None:
            start = float(i)
        elif v < POST["threshold"] and start is not None:
            spans.append([start, float(i)])
            start = None
    if start is not None:
        spans.append([start, float(len(p) - 1) + 1.0])
    merged: list[list[float]] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= POST["max_gap"]:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    def snapped(x: float) -> float:
        if not edges:
            return x
        e = min(edges, key=lambda e: abs(e - x))
        return e if abs(e - x) <= POST["snap"] else x

    out = []
    for a, b in merged:
        a, b = snapped(a), snapped(b)
        if b - a >= POST["min_len"]:
            out.append((round(a * 100) / 100, round(b * 100) / 100))
    return out


def export(recs: list[Recording], out: Path, vector_secs: int = 600) -> None:
    data = [Data(r, EMBED) for r in recs]
    labelled = [d for d in data if (d.y != IGNORE).any()]
    oof = {d.rec.id: frame_model([o for o in labelled if o is not d]).predict_proba(d.x)[:, 1] for d in labelled}
    trees = fit_trees(labelled, oof)
    frame = frame_model(labelled)
    baseline, tj = trees_json(trees)
    model = {"format": FORMAT, "image_model": EMBED, "embedding_dim": int(labelled[0].x.shape[1]),
             "frame_coef": [float(c) for c in frame.coef_[0]], "frame_intercept": float(frame.intercept_[0]),
             "windows": WINDOWS, "edge_cap": EDGE_CAP, "features": feature_names(), "baseline": baseline, "trees": tj,
             "post": POST, "trained_on": [d.rec.id for d in labelled], "image_model_source": f"open_clip {IMAGE_MODEL}"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model))
    # A test vector: the first `vector_secs` seconds of a captioned recording, the exported model's outputs on it.
    d = next(d for d in labelled if d.cap[:, 0].any())
    n = min(vector_secs, len(d.t))
    x = d.x[:n].astype(np.float32)
    has, vis, roll, new, black, silent, edges = inputs(d)
    edges = [e for e in edges if e < n + EDGE_CAP]
    z = x.astype(np.float64) @ np.asarray(model["frame_coef"], np.float32).astype(np.float64) + model["frame_intercept"]
    p = 1 / (1 + np.exp(-z))
    sub = type("Sub", (), {})()
    sub.cap, sub.av, sub.edges = d.cap[:n], d.av[:n], np.array(edges) if edges else np.array([-1e9])
    prob = trees.predict_proba(table(sub, p))[:, 1]
    vec = {"recording": d.rec.id, "rows": [[x[i].tolist(), bool(has[i]), bool(vis[i]), bool(roll[i]), bool(new[i]),
                                             float(black[i]), float(silent[i])] for i in range(n)],
           "edges": edges, "probabilities": prob.tolist(), "breaks": to_breaks(prob, edges)}
    (out / "test_vector.json").write_text(json.dumps(vec))
    print(f"exported to {out}: {len(tj)} trees, trained on {len(labelled)} recordings; test vector {d.rec.id} "
          f"({n} s, {len(vec['breaks'])} breaks)")
