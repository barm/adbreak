"""Baselines on frame embeddings alone: zero-shot CLIP prompts, and a linear classifier scored leave-one-recording-out
(trained on every other recording, so each recording's predictions come from a model that never saw it)."""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .data import BREAK, DATASET, IGNORE, Recording, load_features, second_labels
from .embed import NAME

AD_PROMPTS = ["a TV commercial", "an advertisement for a product", "a car commercial", "a lawyer advertisement on TV",
              "a promo for an upcoming TV show"]
SHOW_PROMPTS = ["a TV news broadcast", "a live football game on TV", "a talk show", "a game show on TV",
                "a weather forecast on TV", "a sports highlights show"]


def zero_shot(x: np.ndarray) -> np.ndarray:
    import open_clip
    import torch

    from .embed import MODEL, device, load

    model = load()
    tok = open_clip.get_tokenizer(MODEL)
    with torch.no_grad():
        t = model.encode_text(tok(AD_PROMPTS + SHOW_PROMPTS).to(device()))
        t = torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
    s = x.astype(np.float32) @ t.T * 100
    s = np.exp(s - s.max(1, keepdims=True))
    s /= s.sum(1, keepdims=True)
    return s[:, : len(AD_PROMPTS)].sum(1)


def to_breaks(t: np.ndarray, p: np.ndarray, smooth: int = 15, threshold: float = 0.5, min_len: float = 30.0,
              max_gap: float = 10.0) -> list[dict]:
    """Per-second break probabilities to breaks: a moving average, a threshold, short gaps closed, short runs dropped."""
    k = np.ones(smooth) / smooth
    ps = np.convolve(p, k, mode="same")
    on = ps >= threshold
    spans, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = t[i]
        if not v and start is not None:
            spans.append([start, t[i]])
            start = None
    if start is not None:
        spans.append([start, t[-1] + 1])
    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [{"start": round(float(a), 2), "end": round(float(b), 2)} for a, b in merged if b - a >= min_len]


def save_source(rec: Recording, name: str, detail: str, breaks: list[dict]) -> None:
    path = DATASET / rec.id / "suggestions.json"
    s = json.loads(path.read_text()) if path.exists() else {"format": 1, "recording": rec.meta, "sources": []}
    s["sources"] = sorted([x for x in s["sources"] if x["name"] != name] + [{"name": name, "detail": detail, "breaks": breaks}],
                          key=lambda x: x["name"])
    path.write_text(json.dumps(s, indent=2) + "\n")


def run(recs: list[Recording]) -> None:
    data = {}
    for r in recs:
        t, x = load_features(r, NAME)
        data[r.id] = (t, x.astype(np.float32), second_labels(r, t))
    labelled = [r for r in recs if (data[r.id][2] != IGNORE).any()]
    print(f"{'recording':22} {'network':9} {'seconds':>8} {'break%':>7} {'zero-shot AUC':>14} {'linear AUC':>11} {'linear acc':>11}")
    for r in labelled:
        t, x, y = data[r.id]
        m = y != IGNORE
        zs = zero_shot(x)
        train = [o for o in labelled if o.id != r.id]
        X = np.concatenate([data[o.id][1][data[o.id][2] != IGNORE] for o in train])
        Y = np.concatenate([data[o.id][2][data[o.id][2] != IGNORE] for o in train])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, Y)
        p = clf.predict_proba(x)[:, 1]
        acc = ((p[m] >= 0.5) == (y[m] == BREAK)).mean()
        print(f"{r.id:22} {r.network:9} {m.sum():8d} {100 * (y[m] == BREAK).mean():6.0f}% "
              f"{roc_auc_score(y[m], zs[m]):14.3f} {roc_auc_score(y[m], p[m]):11.3f} {100 * acc:10.1f}%")
        save_source(r, "frames-lr", f"{NAME} 1 fps, logistic regression trained on the other recordings "
                    f"({', '.join(o.id for o in train)}), smoothed 15 s", to_breaks(t, p))
        save_source(r, "frames-zeroshot", f"{NAME} 1 fps, zero-shot ad vs programme prompts, smoothed 15 s", to_breaks(t, zs))
