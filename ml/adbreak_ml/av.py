"""Per-second black and silence features: how much of each second is black, how much silent, and how far it is
to the nearest moment that is both (the edge networks put around breaks).

Measured with ffmpeg's blackdetect and silencedetect on the copy (the same thresholds as the adbreak detector).
"""

from __future__ import annotations

import re
import subprocess

import numpy as np

from .data import Recording

NAME = "av"
DENSE = ["black", "silent", "edge_dist", "edge_here"]
EDGE_CAP = 120.0


def spans(rec: Recording) -> tuple[list, list, list]:
    r = subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-nostdin", "-v", "info", "-i", str(rec.media),
         "-vf", "blackdetect=d=0.03:pix_th=0.10:pic_th=0.98", "-af", "silencedetect=n=-50dB:d=0.1", "-f", "null", "-"],
        capture_output=True, text=True)
    black = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", r.stderr)]
    ss = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    se = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    silence = list(zip(ss, se))
    edges = []
    for a, b in black:
        for c, d in silence:
            if c < b + 0.3 and a < d + 0.3:
                edges.append(((a + b) / 2))
                break
    return black, silence, edges


def cover(times: np.ndarray, sp: list) -> np.ndarray:
    out = np.zeros(len(times), np.float32)
    for a, b in sp:
        lo, hi = int(np.floor(a)), int(np.ceil(b))
        for s in range(max(lo, 0), min(hi, len(times))):
            out[s] += max(0.0, min(b, s + 1) - max(a, s))
    return np.clip(out, 0, 1)


def build(rec: Recording, times: np.ndarray) -> None:
    black, silence, edges = spans(rec)
    e = np.array(sorted(edges)) if edges else np.array([-1e9])
    j = np.searchsorted(e, times)
    d = np.minimum(np.abs(times - e[np.clip(j - 1, 0, len(e) - 1)]), np.abs(times - e[np.clip(j, 0, len(e) - 1)]))
    dense = np.stack([cover(times, black), cover(times, silence), np.minimum(d, EDGE_CAP) / EDGE_CAP, (d < 1.0).astype(np.float32)], 1)
    out = rec.features(NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t=times.astype(np.float32), dense=dense, edges=e)
