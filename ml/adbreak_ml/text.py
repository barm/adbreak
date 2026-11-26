"""Caption text as model input: the words on screen, hashed (so a recorder can do the same with no vocabulary), and
two small learned scores per second (see `TextHeads`):

- now: whether the caption on screen belongs to a commercial (ads are captioned too);
- trail: whether the captions of the last TRAIL seconds say a break is coming within AHEAD seconds ("we'll be right
  back", "stay with us", ...), learned from the labels, not listed.

Tokens: lowercased runs of letters, digits and apostrophes, single words and adjacent pairs, FNV-1a 32 of their
UTF-8 bytes modulo BUCKETS; present or not.
"""

from __future__ import annotations

import re

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

BUCKETS = 1 << 14
TRAIL = 20
AHEAD = 30
WORD = re.compile(r"[a-z0-9']+")


def fnv32(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def tokens(text: str) -> set[int]:
    w = WORD.findall(text.translate(_LOWER))  # ASCII only, as the Rust side lowercases
    return {fnv32(t) % BUCKETS for t in w + [a + " " + b for a, b in zip(w, w[1:])]}


def cues(vtt: str) -> list[tuple[float, float, str]]:
    """(start, end, text) of each WebVTT cue, exactly as adbreak_captions::parse_vtt reads them (the text: its lines,
    trimmed, joined with a space)."""

    def stamp(x: str) -> float | None:
        secs = 0.0
        for part in x.split(":"):
            try:
                secs = secs * 60 + float(part)
            except ValueError:
                return None
        return secs

    out = []
    lines = vtt.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if "-->" not in line:
            continue
        a, b = line.split("-->", 1)
        start, end = stamp(a.strip()), stamp((b.split() or [""])[0])
        if start is None or end is None:
            continue
        text = []
        while i < len(lines) and lines[i].strip():
            text.append(lines[i].strip())
            i += 1
        out.append((start, end, " ".join(text)))
    return out


def per_second(cue_list: list[tuple[float, float, str]], n: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """For each second: the tokens on screen during it, and the tokens of cues that started in the last TRAIL seconds."""
    now = [set() for _ in range(n)]
    started = [set() for _ in range(n)]
    for a, b, text in cue_list:
        tk = tokens(text)
        for t in range(max(int(a), 0), min(int(np.ceil(b)), n)):
            now[t] |= tk
        if 0 <= int(a) < n:
            started[int(a)] |= tk
    trail = []
    for t in range(n):
        s = set()
        for u in range(max(t - TRAIL + 1, 0), t + 1):
            s |= started[u]
        trail.append(s)

    def matrix(rows):
        ind = [sorted(r) for r in rows]
        indptr = np.cumsum([0] + [len(r) for r in ind])
        return sparse.csr_matrix((np.ones(indptr[-1], np.float32), np.concatenate([np.array(r, np.int64) for r in ind]) if indptr[-1] else np.array([], np.int64), indptr), shape=(n, BUCKETS))

    return matrix(now), matrix(trail)


def ahead_target(y: np.ndarray, break_cls: int, programme_cls: int) -> np.ndarray:
    """1 for a programme second with a break starting within AHEAD seconds, 0 for other programme seconds, -1 else."""
    b = y == break_cls
    out = np.where(y == programme_cls, 0, -1).astype(np.int8)
    starts = [i for i in range(1, len(y)) if b[i] and not b[i - 1]]
    for s in starts:
        lo = max(s - AHEAD, 0)
        seg = out[lo:s]
        seg[seg == 0] = 1
    return out


class TextHeads:
    """The two text scores, trained on some recordings' per-second tokens and labels."""

    def __init__(self, train: list, break_cls: int, programme_cls: int):
        def fit(X, Y):
            if len(set(Y.tolist())) < 2:
                return None
            return LogisticRegression(C=0.5, max_iter=2000, solver="liblinear").fit(X, Y)

        Xn = sparse.vstack([d.tok_now[d.y != -1] for d in train])
        Yn = np.concatenate([d.y[d.y != -1] == break_cls for d in train])
        self.now = fit(Xn, Yn)
        tg = [ahead_target(d.y, break_cls, programme_cls) for d in train]
        Xt = sparse.vstack([d.tok_trail[t >= 0] for d, t in zip(train, tg)])
        Yt = np.concatenate([t[t >= 0] == 1 for t in tg])
        self.trail = fit(Xt, Yt)

    def scores(self, d) -> tuple[np.ndarray, np.ndarray]:
        n = d.tok_now.shape[0]
        now = self.now.predict_proba(d.tok_now)[:, 1] if self.now is not None else np.zeros(n)
        trail = self.trail.predict_proba(d.tok_trail)[:, 1] if self.trail is not None else np.zeros(n)
        # No caption on screen / none lately: no evidence either way (0, not the model's prior on an empty row).
        now[np.asarray(d.tok_now.sum(1)).ravel() == 0] = 0.0
        trail[np.asarray(d.tok_trail.sum(1)).ravel() == 0] = 0.0
        return now, trail


def score(X: sparse.csr_matrix, head: dict) -> np.ndarray:
    """A text score from exported weights (`model.json` "text"), as adbreak-model computes it: the weights of the
    tokens present, in bucket order, plus the intercept, through the logistic; 0 for a second with no tokens."""
    coef = np.asarray(head["coef"], np.float32).astype(np.float64)
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        idx = X.indices[X.indptr[i]:X.indptr[i + 1]]
        if len(idx):
            z = 0.0
            for j in np.sort(idx):
                z += coef[j]
            out[i] = 1 / (1 + np.exp(-(z + head["intercept"])))
    return out


def export(heads: TextHeads) -> dict:
    """The weights for model.json's "text"."""
    def one(m):
        return {"coef": [float(c) for c in m.coef_[0]], "intercept": float(m.intercept_[0])} if m is not None else None
    return {"buckets": BUCKETS, "trail": TRAIL, "now": one(heads.now), "trail_head": one(heads.trail)}
