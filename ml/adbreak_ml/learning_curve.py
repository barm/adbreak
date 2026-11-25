"""The learning curve: the commercial-break model trained on k labelled recordings, scored on one it never saw.

Every labelled recording is held out in turn; for each k, up to `repeats` random training sets of k others. Written
as `learning-curve.json` beside the model, for the app's Model page.
"""

from __future__ import annotations

import itertools
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from . import export as E
from .data import BREAK, IGNORE, Recording


def run(recs: list[Recording], out: Path, repeats: int = 3) -> None:
    data = [E.Data(r) for r in recs if r.features(E.RUST).with_suffix(".json").exists()]
    lab = [d for d in data if (d.y != IGNORE).any()]
    rng = random.Random(0)
    runs = []
    for test in lab:
        others = [d for d in lab if d is not test]
        m = test.y != IGNORE
        y = test.y[m] == BREAK
        if not y.any() or y.all():
            continue
        for k in range(1, len(others) + 1):
            combos = list(itertools.combinations(range(len(others)), k))
            for c in rng.sample(combos, min(repeats, len(combos))):
                train = [others[i] for i in c]
                # Out-of-fold frame probabilities for the trees, as the shipped model; with one recording there is no fold.
                if k >= 2:
                    oof = {o.rec.id: E.frame_model([q for q in train if q is not o]).predict_proba(o.x)[:, 1] for o in train}
                else:
                    oof = {o.rec.id: E.frame_model(train).predict_proba(o.x)[:, 1] for o in train}
                trees = E.fit_trees(train, oof)
                p = trees.predict_proba(E.table(test, E.frame_model(train).predict_proba(test.x)[:, 1]))[:, 1][m]
                runs.append({"test": test.rec.id, "k": k, "seconds": int(sum((o.y != IGNORE).sum() for o in train)),
                             "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 5),
                             "auc": round(float(roc_auc_score(y, p)), 5), "f1": round(float(f1_score(y, p >= 0.5)), 5)})
        print(f"  {test.rec.id}", flush=True)

    def q(v: list[float], p: float) -> float:
        return float(np.quantile(v, p))

    points = []
    for k in sorted({r["k"] for r in runs}):
        r = [x for x in runs if x["k"] == k]
        points.append({"k": k, "runs": len(r), "hours": float(np.median([x["seconds"] for x in r])) / 3600,
                       **{m: [q([x[m] for x in r], p) for p in (0.25, 0.5, 0.75)] for m in ("log_loss", "f1", "auc")}})
    out.mkdir(parents=True, exist_ok=True)
    doc = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "recordings": len(lab),
           "repeats": repeats, "points": points, "runs": runs}
    tmp = out / ".learning-curve.json.tmp"
    tmp.write_text(json.dumps(doc))
    tmp.replace(out / "learning-curve.json")
    for p in points:
        print(f"{p['k']} recordings: log loss {p['log_loss'][1]:.3f}, F1 {p['f1'][1]:.3f} ({p['runs']} runs)")
