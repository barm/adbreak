"""Training runs, recorded in the dataset's database (`runs`): what ran, from where, how far it got, how it ended, and
for a retrain the model it made and its scores. A command started from the CLI records its own run; one started by
the app (`ADBREAK_RUN_ID`) reports into the app's run, so a retrain's steps are one run."""

from __future__ import annotations

import json
import os

from . import data

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL,                 -- evaluate, retrain, export, learning-curve, install, refresh
    origin        TEXT NOT NULL,                 -- app or cli
    note          TEXT NOT NULL DEFAULT '',
    pid           INTEGER,                       -- the process that owns it: a dead one means interrupted
    started_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    finished_at   TEXT,
    status        TEXT NOT NULL DEFAULT 'running', -- running, ok, failed
    progress      TEXT NOT NULL DEFAULT '',
    error         TEXT,
    labels        TEXT,                          -- the labels fingerprint it trained against
    recordings    INTEGER,
    model_name    TEXT,
    model_version TEXT,
    scores        TEXT                           -- JSON: the commercial-break totals after a retrain
)"""

_id: int | None = None
_own = False


def _db():
    c = data.connect()
    c.execute(SCHEMA)
    return c


def start(kind: str, note: str = "") -> None:
    """Records a run for this command, or joins the app's (ADBREAK_RUN_ID)."""
    global _id, _own
    if os.environ.get("ADBREAK_RUN_ID"):
        _id, _own = int(os.environ["ADBREAK_RUN_ID"]), False
        return
    with _db() as c:
        origin = os.environ.get("ADBREAK_RUN_ORIGIN", "cli")  # "ingest" for a new recording's draft
        _id = c.execute("INSERT INTO runs (kind, origin, note, pid, labels) VALUES (?, ?, ?, ?, ?)",
                        (kind, origin, note, os.getpid(), data.labels_fingerprint())).lastrowid
    _own = True


def update(**fields) -> None:
    """Sets fields of the current run (progress, recordings, model_name, model_version, scores)."""
    if _id is None or not fields:
        return
    if "scores" in fields and not isinstance(fields["scores"], str):
        fields["scores"] = json.dumps(fields["scores"])
    with _db() as c:
        c.execute(f"UPDATE runs SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?", (*fields.values(), _id))


def progress(text: str) -> None:
    update(progress=text)
    print(text, flush=True)


def finish(error: str | None = None) -> None:
    """Ends a run this command started (the app ends its own)."""
    if _id is None or not _own:
        return
    with _db() as c:
        c.execute("UPDATE runs SET status = ?, error = ?, finished_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                  ("failed" if error else "ok", error, _id))
