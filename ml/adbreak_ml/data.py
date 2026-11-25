"""The dataset as the training code sees it: recordings, per-second labels, frames and their embeddings.

Labels and suggestions come from ``../dataset`` (git); media and cached features from the media store
(``ADBREAK_MEDIA``, default ``~/Movies/adbreak``), never from git.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATASET = Path(os.environ.get("ADBREAK_DATASET", REPO / "dataset"))
MEDIA = Path(os.environ.get("ADBREAK_MEDIA", Path.home() / "Movies" / "adbreak"))

# Per-second classes. IGNORE is left out of training and scoring.
PROGRAMME, BREAK, IGNORE = 0, 1, -1
# Seconds either side of a labelled edge that are ignored: fades and cuts are ambiguous.
EDGE_MARGIN = 1.0


@dataclass
class Recording:
    id: str
    meta: dict
    labels: dict | None
    suggestions: dict | None

    @property
    def duration(self) -> float:
        return float(self.meta["duration"])

    @property
    def network(self) -> str:
        return self.meta.get("network") or self.meta.get("channel", "")

    @property
    def media(self) -> Path:
        return MEDIA / self.id / "media.mp4"

    def features(self, name: str) -> Path:
        return MEDIA / self.id / "features" / f"{name}.npz"


def recordings() -> list[Recording]:
    out = []
    for d in sorted(DATASET.iterdir()):
        meta = d / "recording.json"
        if not meta.is_file():
            continue
        read = lambda n: json.loads((d / n).read_text()) if (d / n).is_file() else None  # noqa: E731
        out.append(Recording(d.name, json.loads(meta.read_text()), read("labels.json"), read("suggestions.json")))
    return out


def labels_fingerprint() -> str:
    """Which labels a model was trained against: FNV-1a 64 over every recording's labels.json, in id order. The
    labelling page computes the same, and marks sources trained on other labels as stale."""
    h = 0xCBF29CE484222325
    for d in sorted(DATASET.iterdir()):
        f = d / "labels.json"
        if f.is_file():
            for b in f.read_bytes():
                h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def second_labels(rec: Recording, times: np.ndarray) -> np.ndarray:
    """A class for each time: BREAK inside a commercial break, PROGRAMME elsewhere in reviewed time (ad reads
    included: the programme is on screen), IGNORE for unreviewed time, double box, paid programming, and within
    EDGE_MARGIN of an edge. Kinds are in docs/labels.md."""
    y = np.full(len(times), IGNORE, dtype=np.int8)
    if not rec.labels:
        return y
    for r in rec.labels["reviewed"]:
        y[(times >= r["start"]) & (times < r["end"])] = PROGRAMME
    for b in rec.labels["breaks"]:
        inside = (times >= b["start"]) & (times < b["end"])
        kind = b.get("kind", "ad")
        if kind == "ad_read":
            continue
        # Double box and paid programming are neither break nor programme: left out.
        y[inside & (y != IGNORE)] = IGNORE if kind in ("double_box", "paid_programming") else BREAK
    for b in rec.labels["breaks"]:
        for edge in (b["start"], b["end"]):
            y[np.abs(times - edge) < EDGE_MARGIN] = IGNORE
    return y


# The kinds besides commercial breaks that the model also finds, each with an output of its own.
MINOR_KINDS = ("double_box", "ad_read")


def kind_labels(rec: Recording, times: np.ndarray, kind: str) -> np.ndarray:
    """For one of MINOR_KINDS: BREAK (1) inside a label of that kind, PROGRAMME (0) elsewhere in reviewed time
    (commercial breaks and the other kinds included), IGNORE for unreviewed time, paid programming, and within
    EDGE_MARGIN of that kind's edges."""
    y = np.full(len(times), IGNORE, dtype=np.int8)
    if not rec.labels:
        return y
    for r in rec.labels["reviewed"]:
        y[(times >= r["start"]) & (times < r["end"])] = PROGRAMME
    for b in rec.labels["breaks"]:
        inside = (times >= b["start"]) & (times < b["end"]) & (y != IGNORE)
        k = b.get("kind", "ad")
        if k == kind:
            y[inside] = BREAK
        elif k == "paid_programming":
            y[inside] = IGNORE
    for b in rec.labels["breaks"]:
        if b.get("kind", "ad") == kind:
            for edge in (b["start"], b["end"]):
                y[np.abs(times - edge) < EDGE_MARGIN] = IGNORE
    return y


def frames(rec: Recording, fps: float = 1.0, size: int = 256):
    """Yields (time, RGB uint8 array size×size) from the recording's copy, the whole picture squashed to a square
    (no crop: the network logo is in a corner). Decoded by ffmpeg for now."""
    cmd = ["/opt/homebrew/bin/ffmpeg", "-nostdin", "-v", "error", "-i", str(rec.media),
           "-vf", f"fps={fps}:round=down,scale={size}:{size}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n, frame = 0, size * size * 3
    assert p.stdout
    while buf := p.stdout.read(frame):
        if len(buf) < frame:
            break
        # fps filter with round=down puts frame n at n/fps.
        yield n / fps, np.frombuffer(buf, np.uint8).reshape(size, size, 3)
        n += 1
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {rec.media}")


def load_features(rec: Recording, name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(rec.features(name))
    return z["t"], z["x"]
