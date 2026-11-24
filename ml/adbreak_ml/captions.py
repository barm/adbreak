"""Per-second caption features: what caption is on screen each second, and how it behaves.

For each second: whether captions exist for the recording at all, whether a cue is visible, whether it rolled up
(kept a line of the previous cue, as live shows caption) or popped on (whole new text, as ads and pre-produced
shows do), whether a new cue started, how long the current cue has been up, how much text it has, and an embedding
of its text (MobileCLIP2's text tower, so "call now 1-800..." and "touchdown" land in different places).
"""

from __future__ import annotations

import re

import numpy as np

from .data import MEDIA, Recording

NAME = "captions"
DENSE = ["has_captions", "visible", "roll_up", "new_cue", "cue_age", "chars", "upper_ratio", "digits"]


def parse_vtt(text: str) -> list[tuple[float, float, list[str]]]:
    def ts(s: str) -> float:
        v = 0.0
        for part in s.split(":"):
            v = v * 60 + float(part)
        return v

    cues = []
    for m in re.finditer(r"([\d:.]+) --> ([\d:.]+)[^\n]*\n((?:.+\n?)*)", text):
        cues.append((ts(m[1]), ts(m[2]), [l.strip() for l in m[3].strip().split("\n") if l.strip()]))
    return cues


def features(rec: Recording, times: np.ndarray, text_model=None, tokenizer=None) -> tuple[np.ndarray, np.ndarray]:
    """(dense [n, len(DENSE)], text embedding [n, 512] float16, zeros where nothing is on screen)."""
    n = len(times)
    dense = np.zeros((n, len(DENSE)), np.float32)
    emb = np.zeros((n, 512), np.float16)
    path = MEDIA / rec.id / "captions.vtt"
    if not path.is_file():
        return dense, emb
    cues = parse_vtt(path.read_text(errors="replace"))
    dense[:, 0] = 1.0
    # Roll-up: the cue's first line was in the previous cue.
    roll = [bool(i and c[2] and c[2][0] in cues[i - 1][2]) for i, c in enumerate(cues)]
    # The cue on screen at each time (the latest one started before it and not yet ended).
    starts = np.array([c[0] for c in cues])
    idx = np.searchsorted(starts, times, side="right") - 1
    texts: dict[int, int] = {}
    for k, (t, i) in enumerate(zip(times, idx)):
        if i < 0 or t >= cues[i][1] + 0.5:
            continue
        s, _, lines = cues[i]
        text = " ".join(lines)
        letters = [ch for ch in text if ch.isalpha()]
        dense[k, 1] = 1.0
        dense[k, 2] = float(roll[i])
        dense[k, 3] = float(s > t - 1.0)
        dense[k, 4] = min(t - s, 10.0) / 10.0
        dense[k, 5] = min(len(text), 100) / 100.0
        dense[k, 6] = sum(ch.isupper() for ch in letters) / max(len(letters), 1)
        dense[k, 7] = min(sum(ch.isdigit() for ch in text), 10) / 10.0
        texts.setdefault(i, len(texts))
    if text_model is not None and texts:
        import torch

        order = sorted(texts, key=texts.get)
        vecs = []
        with torch.no_grad():
            for j in range(0, len(order), 256):
                batch = [" ".join(cues[i][2]).lower() for i in order[j : j + 256]]
                device = next(text_model.parameters()).device
                v = text_model.encode_text(tokenizer(batch).to(device))
                vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float16))
        table = np.concatenate(vecs)
        for k, i in enumerate(idx):
            if dense[k, 1] and i in texts:
                emb[k] = table[texts[i]]
    return dense, emb


def build(rec: Recording, times: np.ndarray, text_model, tokenizer) -> None:
    out = rec.features(NAME)
    dense, emb = features(rec, times, text_model, tokenizer)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t=times.astype(np.float32), dense=dense, text=emb)
