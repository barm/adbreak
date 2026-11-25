"""Earlier installed models, so a retrain that turns out worse can be undone: `export` keeps the model it replaces in
`<model dir>/history/<version>/` (the newest KEEP), and `install` puts one back. Only the JSON files: the image model
(`.mlpackage`) doesn't change between versions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# In the order a recorder needs them replaced: the test vector first, model.json (what it watches) last.
FILES = ("test_vector.json", "model-info.json", "model.json")
KEEP = 10


def entries(out: Path) -> list[dict]:
    """Kept models' info (model-info.json), newest first."""
    found = []
    for d in (out / "history").glob("*/model-info.json"):
        try:
            found.append(json.loads(d.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(found, key=lambda i: i.get("exported_at", ""), reverse=True)


def keep(out: Path) -> None:
    """Keeps a copy of the installed model, if there is one and it isn't kept already; prunes to the newest KEEP."""
    if not all((out / f).is_file() for f in FILES):
        return
    version = json.loads((out / "model-info.json").read_text()).get("version")
    if not version:
        return
    dest = out / "history" / version
    if not dest.is_dir():
        tmp = out / "history" / f".{version}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for f in FILES:
            shutil.copy2(out / f, tmp / f)
        tmp.rename(dest)
    for old in entries(out)[KEEP:]:
        shutil.rmtree(out / "history" / old["version"], ignore_errors=True)


def install(out: Path, version: str) -> None:
    """Puts a kept model back (keeping the one it replaces), atomically file by file in FILES order."""
    src = out / "history" / version
    if not all((src / f).is_file() for f in FILES):
        raise SystemExit(f"no kept model {version} in {out / 'history'} (adbreak-ml history lists them)")
    keep(out)
    for f in FILES:
        tmp = out / f".{f}.tmp"
        shutil.copy2(src / f, tmp)
        tmp.replace(out / f)
    print(f"installed {version} in {out}")
