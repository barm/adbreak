"""Experiment: segment, then decide. Not part of the model; kept to rerun when there is more labelled data.

Cuts each recording at candidate boundaries (black-and-silent edges, black pictures, picture jumps: the top 3% of
second-to-second embedding changes), describes each segment by what is inside it (length, the picture model's
score, captions, the kinds of cut at either end, its neighbours) and classifies segments with gradient-boosted trees
trained on the other recordings' segments (leave-one-recording-out). Writes the result as the "segments" source, to
score with `adbreak eval` beside the model.

Run on a copy of the dataset (it writes a source):

    sqlite3 "$HOME/Library/Application Support/adbreak/adbreak.db" ".backup /tmp/segments.db"
    cd ml && ADBREAK_DB=/tmp/segments.db uv run python experiments/segment_classifier.py
    ADBREAK_DB=/tmp/segments.db adbreak eval

Switches (environment): X_GAP=15 closes gaps between breaks up to that many seconds; X_FUSED=1 adds the per-second
model's leave-one-out probability inside each segment (from model-out.json, so run `adbreak-ml evaluate` first; it
leaks slightly for the training recordings, whose probabilities came from models that saw the test recording);
X_SIDE=1 adds one-sided context (the scores before a segment's start and after its end); X_JUMP_Q sets the picture
jump quantile.

Results on 14 recordings (25.4 h, 160 breaks), against the per-second model with snapping (adbreak-0.2-300):

    | variant                                   | F1  | start / end error | show cut / h | ads left / h |
    | per-second model + snapping (installed)   | 96% | 4.8 / 6.1 s       | 34 s         | 45 s         |
    | segments, picture model only              | 94% | 6.7 / 8.6 s       | 57 s         | 58 s         |
    | + X_GAP=15                                | 95% | 6.0 / 7.6 s       | 61 s         | 46 s         |
    | + X_FUSED=1                               | 95% | 6.3 / 7.8 s       | 59 s         | 50 s         |
    | X_SIDE=1 (one-sided context)              | 90% | 16.8 / 28.5 s     | 165 s        | 54 s         |

The cut points are good (87% of labelled edges have one within 1 s: 172 black-and-silent edges, 85 black pictures,
29 picture jumps); deciding per segment is the weak part, probably short programme pieces beside breaks (bumpers,
"we'll be right back") classed as break. The one-sided context result is worse than expected and may be a bug.
"""
import json, os, sqlite3, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from adbreak_ml.export import Data, frame_model, black_points
from adbreak_ml.data import recordings, BREAK, IGNORE

JUMP_Q = float(os.environ.get("X_JUMP_Q", "0.97"))  # picture jumps: the top 3% of second-to-second changes

def candidates(d):
    """(time, kind) cut points: 0 edge, 1 black, 2 picture jump. Close ones merged, the stronger kind kept."""
    pts = [(float(e), 0) for e in d.edges if e > -1e8] + [(b, 1) for b in black_points(d.av[:, 0])]
    x = d.x / np.maximum(np.linalg.norm(d.x, axis=1, keepdims=True), 1e-9)
    jump = np.r_[0, 1 - np.sum(x[1:] * x[:-1], axis=1)]
    th = np.quantile(jump, JUMP_Q)
    for t in range(1, len(jump) - 1):
        if jump[t] >= th and jump[t] >= jump[t - 1] and jump[t] >= jump[t + 1]:
            pts.append((float(t), 2))
    pts.sort()
    out = []
    for t, k in pts:
        if out and t - out[-1][0] < 2.0:
            if k < out[-1][1]:
                out[-1] = (t, k)
        else:
            out.append((t, k))
    return out, jump

def fused(d):
    """The per-second model's leave-one-out probability for this recording (model-out.json)."""
    f = d.rec.features("rust-v1").parent / "model-out.json"
    q = np.array(json.load(open(f))["probability"])
    return np.pad(q, (0, max(len(d.t) - len(q), 0)), constant_values=q[-1] if len(q) else 0)[:len(d.t)]

def segments(d, p):
    """Per segment: (start, end, features, break fraction, known seconds)."""
    cuts, jump = candidates(d)
    P, CAP = p, d.cap
    Q = fused(d) if os.environ.get("X_FUSED") else None
    n = len(d.t)
    bounds = [(0.0, 3)] + [c for c in cuts if 0 < c[0] < n] + [(float(n), 3)]
    rows = []
    for (a, ka), (b, kb) in zip(bounds, bounds[1:]):
        i, j = int(a), max(int(np.ceil(b)), int(a) + 1)
        seg = p[i:j]; cap = d.cap[i:j]; y = d.y[i:j]
        known = y != IGNORE
        rows.append([a, b, ka, kb, seg, cap, (y[known] == BREAK).mean() if known.any() else np.nan, known.sum(), jump[i:j]])
    feats = []
    for k, (a, b, ka, kb, seg, cap, frac, nk, jmp) in enumerate(rows):
        prev = rows[k - 1] if k else None
        nxt = rows[k + 1] if k + 1 < len(rows) else None
        ctx = lambda r: [r[4].mean(), np.log1p(r[1] - r[0])] if r else [np.nan, np.nan]  # noqa: E731
        # One-sided context: the picture score and captions before the start and after the end, never across.
        i0, j0 = int(a), max(int(np.ceil(b)), int(a) + 1)
        side = []
        for w in (15, 60, 150):
            before, after = P[max(i0 - w, 0):i0], P[j0:j0 + w]
            cb, ca = CAP[max(i0 - w, 0):i0, 1], CAP[j0:j0 + w, 1]
            side += [before.mean() if len(before) else np.nan, after.mean() if len(after) else np.nan,
                     cb.mean() if len(cb) else np.nan, ca.mean() if len(ca) else np.nan]
        f = [np.log1p(b - a), seg.mean(), seg.min(), seg.max(), seg.std(), (seg > 0.5).mean(),
             cap[:, 1].mean(), cap[:, 2].mean(), cap[:, 3].mean(), cap[:, 0].max(), jmp.mean(),
             *([Q[i0:j0].mean(), Q[i0:j0].min(), Q[i0:j0].max(), (Q[i0:j0] > 0.5).mean()] if Q is not None else []),
             *np.eye(4)[ka], *np.eye(4)[kb], *ctx(prev), *ctx(nxt), *(side if os.environ.get('X_SIDE') else [])]
        feats.append(f)
    return [(r[0], r[1], f, r[6], r[7]) for r, f in zip(rows, feats)]

recs = [r for r in recordings() if r.labels and r.features("rust-v1").with_suffix(".json").exists()]
data = [Data(r) for r in recs]
data = [d for d in data if (d.y != IGNORE).any()]
db = sqlite3.connect(os.environ["ADBREAK_DB"])
for test in data:
    train = [d for d in data if d is not test]
    X, Y, W = [], [], []
    for o in train:
        p = frame_model([q for q in train if q is not o]).predict_proba(o.x)[:, 1]
        for a, b, f, frac, nk in segments(o, p):
            if nk:
                X.append(f); Y.append(frac >= 0.5); W.append(nk)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
    clf.fit(np.array(X), np.array(Y), sample_weight=np.array(W, float))
    segs = segments(test, frame_model(train).predict_proba(test.x)[:, 1])
    prob = clf.predict_proba(np.array([s[2] for s in segs]))[:, 1]
    breaks, cur = [], None
    for (a, b, *_), q in zip(segs, prob):
        if q >= 0.5:
            cur = [cur[0], b] if cur else [a, b]
        elif cur:
            breaks.append(cur); cur = None
    if cur: breaks.append(cur)
    merged = []
    for a, b in breaks:  # gaps up to 15 s closed, as the per-second model's post-processing does
        if merged and a - merged[-1][1] <= float(os.environ.get('X_GAP', '0')):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    breaks = merged
    breaks = [(round(a, 2), round(b, 2)) for a, b in breaks if b - a >= 30]
    src = {"name": "segments", "detail": "experiment", "breaks": [{"start": a, "end": b, "kind": "ad"} for a, b in breaks]}
    db.execute("INSERT INTO sources (id, name, source) VALUES (?, ?, ?) ON CONFLICT(id, name) DO UPDATE SET source = excluded.source", (test.rec.id, "segments", json.dumps(src)))
    db.commit()
    print(f"{test.rec.id:22} {len(segs):5} segments  {len(breaks):3} breaks", flush=True)
