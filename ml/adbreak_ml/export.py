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

from .data import BREAK, IGNORE, MINOR_KINDS, Recording, kind_labels
from .embed import NAME as EMBED, MODEL as IMAGE_MODEL
from .data import second_labels
from .fusion import WINDOWS, window_mean

RUST = "rust-v1"


class Data:
    """A recording's per-second inputs as `adbreak features` computes them (the code a recorder runs), and its labels."""

    def __init__(self, rec: Recording, _name: str = RUST):
        self.rec = rec
        d = rec.features(RUST).parent
        meta = json.loads((d / f"{RUST}.json").read_text())
        raw = np.fromfile(d / f"{RUST}.f32", np.float32).reshape(meta["rows"], meta["dim"] + 6)
        x = raw[:, : meta["dim"]].copy()
        # Seconds without a picture: the previous second's (adbreak-model carries the previous probability).
        for i in range(len(x)):
            if np.isnan(x[i, 0]):
                x[i] = x[i - 1] if i else 0.0
        self.x = x
        self.t = np.arange(meta["rows"], dtype=np.float64)
        self.y = second_labels(rec, self.t)
        self.ys = {k: kind_labels(rec, self.t, k) for k in MINOR_KINDS}
        self.cap = raw[:, meta["dim"] : meta["dim"] + 4].astype(np.float64)
        self.av = raw[:, meta["dim"] + 4 :].astype(np.float64)
        self.edges = np.array(meta["edges"]) if meta["edges"] else np.array([-1e9])

FORMAT = 1
EDGE_CAP = 120.0
POST = {"threshold": 0.5, "min_len": 30.0, "max_gap": 15.0, "snap": 20.0}
# The other kinds come from the picture alone, averaged over ±KIND_SMOOTH s: with minutes of examples, trees on the
# other inputs learn the examples rather than the kind (leave-one-out AUC 0.64 against 0.82 for ad reads, 0.76
# against 0.94 for double boxes). Their runs are short (an ad read is often under 10 s): shorter runs kept, smaller
# gaps closed, nearer edges; thresholds from leave-one-out.
KIND_SMOOTH = 5
KIND_POST = {"double_box": {"threshold": 0.5, "min_len": 4.0, "max_gap": 3.0, "snap": 3.0},
             "ad_read": {"threshold": 0.4, "min_len": 4.0, "max_gap": 3.0, "snap": 3.0}}


def target(d: "Data", kind: str) -> np.ndarray:
    """Per-second BREAK / PROGRAMME / IGNORE for an output: "ad" (commercial breaks) or one of MINOR_KINDS."""
    return d.y if kind == "ad" else d.ys[kind]


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


class Never:
    """A frame model for a kind its training recordings have no examples of: probability 0."""

    coef_ = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.stack([np.ones(len(x)), np.zeros(len(x))], 1)


def frame_model(train: list[Data], kind: str = "ad"):
    X = np.concatenate([d.x[target(d, kind) != IGNORE] for d in train])
    Y = np.concatenate([target(d, kind)[target(d, kind) != IGNORE] == BREAK for d in train])
    if not Y.any():
        return Never()
    # The other kinds are rare (minutes in hours): weighted so they count as much as everything else.
    return LogisticRegression(C=1.0, max_iter=3000, class_weight=None if kind == "ad" else "balanced").fit(X, Y)


def fit_trees(train: list[Data], oof: dict, kind: str = "ad") -> HistGradientBoostingClassifier:
    views = [(o, False) for o in train] + [(o, True) for o in train if o.cap[:, 0].any()]
    X = np.concatenate([table(o, oof[o.rec.id], drop)[target(o, kind) != IGNORE] for o, drop in views])
    Y = np.concatenate([target(o, kind)[target(o, kind) != IGNORE] == BREAK for o, _ in views])
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          class_weight=None if kind == "ad" else "balanced", random_state=0).fit(X, Y)


def train_head(train: list[Data], kind: str):
    """One output. Commercial breaks: a frame model and trees on its out-of-fold frame probabilities. The other
    kinds: a frame model alone (see KIND_SMOOTH)."""
    if kind != "ad":
        return frame_model(train, kind), None
    oof = {o.rec.id: frame_model([q for q in train if q is not o], kind).predict_proba(o.x)[:, 1] for o in train}
    return frame_model(train, kind), fit_trees(train, oof, kind)


def predict_head(head, d: Data) -> tuple[np.ndarray, np.ndarray]:
    frame, trees = head
    p = frame.predict_proba(d.x)[:, 1]
    if trees is None:
        return p, window_mean(p, KIND_SMOOTH)
    return p, trees.predict_proba(table(d, p))[:, 1]


def has_kind(train: list[Data], kind: str) -> bool:
    return any((target(d, kind) == BREAK).any() for d in train)


def subtract(spans: list, cuts: list) -> list:
    """`spans` with every part inside `cuts` taken out."""
    for x, y in cuts:
        spans = [q for (s, e) in spans for q in ((s, min(e, x)), (max(s, y), e)) if q[1] > q[0]]
    return spans


def combine(ad: list, minor: dict, probs: dict) -> list[tuple[float, float, str]]:
    """Commercial breaks and the other kinds' runs as one list, as adbreak_model::Model::kind_breaks. A double box
    takes its time from a commercial break it overlaps (it is often the break's tail, which the break output also
    covers); an ad read loses any part inside a commercial break or a double box. Pieces shorter than their kind's
    min_len are dropped, and commercial breaks shorter than POST's min_len."""
    db = [(s, e) for s, e in minor.get("double_box", []) if e - s >= KIND_POST["double_box"]["min_len"]]
    ads = [(s, e) for s, e in subtract(list(ad), db) if e - s >= POST["min_len"]]
    reads = [(s, e) for s, e in subtract(minor.get("ad_read", []), ads + db) if e - s >= KIND_POST["ad_read"]["min_len"]]
    return sorted([(s, e, "ad") for s, e in ads] + [(s, e, "double_box") for s, e in db] + [(s, e, "ad_read") for s, e in reads])


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


def to_breaks(p: np.ndarray, edges: list[float], post: dict | None = None) -> list[tuple[float, float]]:
    """As adbreak_model::to_breaks."""
    POST = post or globals()["POST"]
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


def model_version(text: str) -> str:
    """model.json's version: FNV-1a 64 of its bytes, as `adbreak_model::version`."""
    h = 0xCBF29CE484222325
    for b in text.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def export(recs: list[Recording], out: Path, vector_secs: int = 600) -> None:
    data = [Data(r) for r in recs if r.features(RUST).with_suffix(".json").exists()]
    labelled = [d for d in data if (d.y != IGNORE).any()]
    oof = {d.rec.id: frame_model([o for o in labelled if o is not d]).predict_proba(d.x)[:, 1] for d in labelled}
    trees = fit_trees(labelled, oof)
    frame = frame_model(labelled)
    baseline, tj = trees_json(trees)
    model = {"format": FORMAT, "image_model": EMBED, "embedding_dim": int(labelled[0].x.shape[1]),
             "frame_coef": [float(c) for c in frame.coef_[0]], "frame_intercept": float(frame.intercept_[0]),
             "windows": WINDOWS, "edge_cap": EDGE_CAP, "features": feature_names(), "baseline": baseline, "trees": tj,
             "post": POST, "trained_on": [d.rec.id for d in labelled], "image_model_source": f"open_clip {IMAGE_MODEL}"}
    # The other kinds' outputs (see KIND_SMOOTH), for those with examples to learn from.
    model["kinds"] = {}
    for k in MINOR_KINDS:
        f = frame_model(labelled, k)
        if f.coef_ is not None:
            model["kinds"][k] = {"frame_coef": [float(c) for c in f.coef_[0]], "frame_intercept": float(f.intercept_[0]),
                                 "smooth": KIND_SMOOTH, "post": KIND_POST[k]}
    text = json.dumps(model)
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

    def head_p(h: dict) -> np.ndarray:
        z = x.astype(np.float64) @ np.asarray(h["frame_coef"], np.float32).astype(np.float64) + h["frame_intercept"]
        return window_mean(1 / (1 + np.exp(-z)), h["smooth"])

    kp = {k: head_p(h) for k, h in model["kinds"].items()}
    kinds = combine(to_breaks(prob, edges), {k: to_breaks(v, edges, KIND_POST[k]) for k, v in kp.items()}, kp)
    vec = {"model": model_version(text), "recording": d.rec.id, "rows": [[x[i].tolist(), bool(has[i]), bool(vis[i]), bool(roll[i]), bool(new[i]),
                                             float(black[i]), float(silent[i])] for i in range(n)],
           "edges": edges, "probabilities": prob.tolist(), "breaks": to_breaks(prob, edges),
           "kind_probabilities": {k: v.tolist() for k, v in kp.items()}, "kind_breaks": [list(b) for b in kinds]}
    # Installed atomically, the test vector first: a recorder watching model.json reloads when it changes, and
    # checks the model against the test vector made for it (its "model" is model.json's version).
    out.mkdir(parents=True, exist_ok=True)
    for name, body in (("test_vector.json", json.dumps(vec)), ("model.json", text)):
        tmp = out / f".{name}.tmp"
        tmp.write_text(body)
        tmp.replace(out / name)
    print(f"exported to {out}: {len(tj)} trees, trained on {len(labelled)} recordings; test vector {d.rec.id} "
          f"({n} s, {len(vec['breaks'])} breaks)")


def evaluate(recs: list[Recording]) -> None:
    """The shipped model's results for every recording, written as the "model" suggestion (score with `adbreak
    eval`) and, per second, as `features/model-out.json` (the picture score and the final probability, for the
    labelling page). Labelled recordings: leave-one-recording-out, from frame and tree models trained on the others
    only. Unlabelled ones: from models trained on every labelled recording (a first draft to label against)."""
    from .baseline import save_source
    from .data import labels_fingerprint

    data = [Data(r) for r in recs if r.features(RUST).with_suffix(".json").exists()]
    labelled = [d for d in data if (d.y != IGNORE).any()]
    fp = labels_fingerprint()

    def write(d: Data, train: list[Data], how: str) -> None:
        p_frame, prob = predict_head(train_head(train, "ad"), d)
        edges = [float(e) for e in d.edges if e > -1e8]
        probs = {k: predict_head(train_head(train, k), d)[1] for k in MINOR_KINDS if has_kind(train, k)}
        minor = {k: to_breaks(p, edges, KIND_POST[k]) for k, p in probs.items()}
        found = combine(to_breaks(prob, edges), minor, probs)
        breaks = [{"start": a, "end": b, "kind": k} for a, b, k in found]
        save_source(d.rec, "model", f"adbreak-model inputs (rust-v1), {how}", breaks)
        out = d.rec.features(RUST).parent / "model-out.json"
        out.write_text(json.dumps({"trained_on": fp, "how": how, "picture": [round(float(v), 3) for v in p_frame],
                                   "probability": [round(float(v), 3) for v in prob],
                                   "kinds": {k: [round(float(v), 3) for v in p] for k, p in probs.items()}}))
        n = {k: sum(1 for *_, kk in found if kk == k) for k in ("ad", *MINOR_KINDS)}
        print(f"  {d.rec.id}: " + ", ".join(f"{v} {k}" for k, v in n.items()))

    for d in labelled:
        write(d, [o for o in labelled if o is not d], f"leave-one-recording-out over {len(labelled)}")
    for d in data:
        if d not in labelled:
            write(d, labelled, f"trained on all {len(labelled)} labelled recordings")
