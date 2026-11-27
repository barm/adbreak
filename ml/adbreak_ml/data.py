"""The dataset as the training code sees it: recordings, per-second labels, frames and their embeddings.

Recordings, labels and suggestions come from the adbreak database (``ADBREAK_DB``, default
``~/Library/Application Support/adbreak/adbreak.db``; ``adbreak-db``), shared with the app; media and cached features
from the media store (``ADBREAK_MEDIA``, default ``~/Movies/adbreak``). Neither is in git.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sqlite3

REPO = Path(__file__).resolve().parents[2]
DB = Path(os.environ.get("ADBREAK_DB", Path.home() / "Library" / "Application Support" / "adbreak" / "adbreak.db"))
MEDIA = Path(os.environ.get("ADBREAK_MEDIA", Path.home() / "Movies" / "adbreak"))
# The installed model (what a recorder loads), beside the database by default.
MODEL = Path(os.environ.get("ADBREAK_MODEL_DIR", Path.home() / "Library" / "Application Support" / "adbreak" / "model"))

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


def connect() -> sqlite3.Connection:
    if not DB.is_file():
        raise SystemExit(f"no adbreak database at {DB} (run the app, or adbreak dataset import, first)")
    c = sqlite3.connect(DB, timeout=10)
    c.execute("PRAGMA busy_timeout = 10000")
    return c


def recordings() -> list[Recording]:
    with connect() as c:
        recs = c.execute("SELECT id, recording FROM recordings ORDER BY id").fetchall()
        labels = dict(c.execute("SELECT id, labels FROM labels").fetchall())
        sources: dict[str, list] = {}
        for i, s in c.execute("SELECT id, source FROM sources ORDER BY id, name"):
            sources.setdefault(i, []).append(json.loads(s))
    out = []
    for i, meta in recs:
        meta = json.loads(meta)
        sugg = {"format": 1, "recording": meta, "sources": sources[i]} if i in sources else None
        out.append(Recording(i, meta, json.loads(labels[i]) if i in labels else None, sugg))
    return out


def save_source(rec_id: str, source: dict) -> None:
    """Adds or replaces one of a recording's sources (by name)."""
    with connect() as c:
        c.execute(
            "INSERT INTO sources (id, name, source) VALUES (?, ?, ?) ON CONFLICT(id, name) DO UPDATE SET "
            "source = excluded.source, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')",
            (rec_id, source["name"], json.dumps(source)),
        )


def holdout_ids() -> set[str]:
    """The held-out recordings (adbreak holdout): never trained on, and what every model is scored on."""
    try:
        with connect() as c:
            return {i for (i,) in c.execute("SELECT id FROM holdout")}
    except sqlite3.OperationalError:  # a database from before the held-out set
        return set()


def labels_fingerprint() -> str:
    """Which labels a model was trained against: FNV-1a 64 over every recording's labels JSON in id order, as
    adbreak_db::Db::fingerprint. The app marks sources trained on other labels as stale."""
    h = 0xCBF29CE484222325
    with connect() as c:
        for (t,) in c.execute("SELECT labels FROM labels ORDER BY id"):
            for b in t.encode():
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
    for x in rec.labels.get("excluded", []):  # bad reception, say: never trained on
        y[(times >= x["start"]) & (times < x["end"])] = IGNORE
    for b in rec.labels["breaks"]:
        inside = (times >= b["start"]) & (times < b["end"])
        kind = b.get("kind", "ad")
        if kind == "ad_read":
            continue
        # Double box, bumper, credits and paid programming are neither break nor programme: left out.
        y[inside & (y != IGNORE)] = IGNORE if kind in ("double_box", "bumper", "credits", "paid_programming") else BREAK
    for b in rec.labels["breaks"]:
        for edge in (b["start"], b["end"]):
            y[np.abs(times - edge) < EDGE_MARGIN] = IGNORE
    return y


# The kinds besides commercial breaks that the model also finds, each with an output of its own.
MINOR_KINDS = ("double_box", "ad_read", "bumper", "credits")
# Kinds labelled only in newer recordings: learned only from recordings that list them in "labelled_kinds".
OPTIONAL_KINDS = ("bumper", "credits")


def kind_labels(rec: Recording, times: np.ndarray, kind: str) -> np.ndarray:
    """For one of MINOR_KINDS: BREAK (1) inside a label of that kind, PROGRAMME (0) elsewhere in reviewed time
    (commercial breaks and the other kinds included), IGNORE for unreviewed time, paid programming, and within
    EDGE_MARGIN of that kind's edges."""
    y = np.full(len(times), IGNORE, dtype=np.int8)
    if not rec.labels:
        return y
    # Bumpers and credits are learned only from recordings that label them: elsewhere an unlabelled one isn't programme.
    if kind in OPTIONAL_KINDS and kind not in rec.labels.get("labelled_kinds", []):
        return y
    for r in rec.labels["reviewed"]:
        y[(times >= r["start"]) & (times < r["end"])] = PROGRAMME
    for x in rec.labels.get("excluded", []):  # bad reception, say: never trained on
        y[(times >= x["start"]) & (times < x["end"])] = IGNORE
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
