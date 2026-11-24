"""Fusion: per-second frame decisions, captions and black-and-silent edges, over a window of context, into breaks.

Everything is leave-one-recording-out. For a held-out recording, the frame classifier and the caption-text
classifier that score it were trained on the other recordings only; the fusion model is trained on the other
recordings' own out-of-fold scores (each from classifiers that never saw *that* recording), so at no point does a
model score something it was trained on.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import av, captions
from .baseline import save_source
from .data import BREAK, IGNORE, Recording, load_features, second_labels

WINDOWS = [2, 7, 15, 30, 60, 150]


def window_mean(x: np.ndarray, half: int) -> np.ndarray:
    k = 2 * half + 1
    c = np.cumsum(np.concatenate([[0.0], x]))
    i = np.arange(len(x))
    lo, hi = np.clip(i - half, 0, len(x)), np.clip(i + half + 1, 0, len(x))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1) if k else x


class Data:
    def __init__(self, rec: Recording, frame_feature: str):
        self.rec = rec
        self.t, x = load_features(rec, frame_feature)
        self.x = x.astype(np.float32)
        self.y = second_labels(rec, self.t)
        z = np.load(rec.features(captions.NAME))
        self.cap, self.text = z["dense"], z["text"].astype(np.float32)
        z = np.load(rec.features(av.NAME))
        self.av, self.edges = z["dense"], z["edges"]
        n = min(len(self.t), len(self.cap), len(self.av))
        for k in ("t", "x", "y", "cap", "text", "av"):
            setattr(self, k, getattr(self, k)[:n])


def frame_scores(train: list[Data], test: Data, C: float = 1.0) -> np.ndarray:
    X = np.concatenate([d.x[d.y != IGNORE] for d in train])
    Y = np.concatenate([d.y[d.y != IGNORE] == BREAK for d in train])
    return LogisticRegression(C=C, max_iter=3000).fit(X, Y).predict_proba(test.x)[:, 1]


def text_scores(train: list[Data], test: Data) -> np.ndarray:
    """Break probability from the caption on screen; 0.5 where there is none (or no captions to learn from)."""
    out = np.full(len(test.t), 0.5)
    vis = lambda d: (d.y != IGNORE) & (d.cap[:, 1] > 0)  # noqa: E731
    X = np.concatenate([d.text[vis(d)] for d in train])
    Y = np.concatenate([d.y[vis(d)] == BREAK for d in train])
    if len(set(Y)) < 2:
        return out
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(X, Y)
    m = test.cap[:, 1] > 0
    if m.any():
        out[m] = clf.predict_proba(test.text[m])[:, 1]
    return out


def table(d: Data, p: np.ndarray, q: np.ndarray, drop_captions: bool = False) -> np.ndarray:
    """Model inputs per second. ``drop_captions`` makes the recording look as if it had none (training augmentation,
    so the model also learns to decide from frames alone: some recordings have no captions)."""
    cap = np.zeros_like(d.cap) if drop_captions else d.cap
    if drop_captions:
        q = np.full_like(q, 0.5)
    has = cap[:, 0]
    vis, roll = cap[:, 1], cap[:, 2]
    black, silent, edist, ehere = d.av.T
    cols = [p, q, has, vis, roll, cap[:, 3], black, silent, edist, ehere]
    for w in WINDOWS:
        cols += [window_mean(p, w), window_mean(q, w), window_mean(vis, w), window_mean(roll, w), window_mean(ehere, w) * (2 * w + 1)]
    cols.append(np.sqrt(np.maximum(window_mean(p**2, 15) - window_mean(p, 15) ** 2, 0)))
    return np.stack(cols, 1).astype(np.float32)


def to_breaks(t: np.ndarray, p: np.ndarray, edges: np.ndarray, threshold: float = 0.35, min_len: float = 30.0,
              max_gap: float = 25.0, snap: float = 20.0) -> list[dict]:
    on = p >= threshold
    spans, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = float(t[i])
        if not v and start is not None:
            spans.append([start, float(t[i])])
            start = None
    if start is not None:
        spans.append([start, float(t[-1]) + 1])
    merged: list[list[float]] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    def snapped(x: float) -> float:
        if len(edges) and edges[0] > -1e8:
            j = int(np.argmin(np.abs(edges - x)))
            if abs(edges[j] - x) <= snap:
                return float(edges[j])
        return x

    out = []
    for a, b in merged:
        a, b = snapped(a), snapped(b)
        if b - a >= min_len:
            out.append({"start": round(a, 2), "end": round(b, 2)})
    return out


def fit(train: list[Data], oof: dict) -> HistGradientBoostingClassifier:
    # Each training recording as recorded, and again with its captions removed (if it had any).
    views = [(o, False) for o in train] + [(o, True) for o in train if o.cap[:, 0].any()]
    X = np.concatenate([table(o, *oof[o.rec.id], drop_captions=drop)[o.y != IGNORE] for o, drop in views])
    Y = np.concatenate([o.y[o.y != IGNORE] == BREAK for o, _ in views])
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=0).fit(X, Y)


DETAIL = ("frames ({}, logistic regression) + captions (text and timing) + black/silence, gradient-boosted over "
          "windows to 150 s; edges snapped to black-and-silent moments; {}")


def run(recs: list[Recording], frame_feature: str, write: bool = True) -> dict:
    """Labelled recordings are scored leave-one-recording-out; unlabelled ones get breaks from models trained on
    every labelled recording (a first draft to label against)."""
    data = [Data(r, frame_feature) for r in recs]
    labelled = [d for d in data if (d.y != IGNORE).any()]
    # Out-of-fold frame and text scores for every labelled recording, each from models trained on the others.
    oof = {}
    for d in labelled:
        others = [o for o in labelled if o is not d]
        oof[d.rec.id] = (frame_scores(others, d), text_scores(others, d))
    results = {}
    for d in labelled:
        others = [o for o in labelled if o is not d]
        # The fusion model must not see d; its inputs for the others must not come from models that saw d either,
        # strictly; with few recordings the others' scores come from models trained without them but with d.
        p = fit(others, oof).predict_proba(table(d, *oof[d.rec.id]))[:, 1]
        m = d.y != IGNORE
        f = oof[d.rec.id][0]
        results[d.rec.id] = dict(frame_auc=roc_auc_score(d.y[m] == BREAK, f[m]), fusion_auc=roc_auc_score(d.y[m] == BREAK, p[m]),
                                 frame_acc=((f[m] >= 0.5) == (d.y[m] == BREAK)).mean(), fusion_acc=((p[m] >= 0.5) == (d.y[m] == BREAK)).mean())
        if write:
            save_source(d.rec, "fusion", DETAIL.format(frame_feature, "leave-one-recording-out"), to_breaks(d.t, p, d.edges))
    unlabelled = [d for d in data if d not in labelled]
    if unlabelled and write:
        model = fit(labelled, oof)
        for d in unlabelled:
            p = model.predict_proba(table(d, frame_scores(labelled, d), text_scores(labelled, d)))[:, 1]
            save_source(d.rec, "fusion", DETAIL.format(frame_feature, f"trained on {len(labelled)} labelled recordings"),
                        to_breaks(d.t, p, d.edges))
            results[d.rec.id] = {}
    return results
