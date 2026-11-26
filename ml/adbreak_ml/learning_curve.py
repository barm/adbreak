"""The learning curve: the commercial-break model trained on k labelled recordings, scored on the ones it never saw.

For each k, `sets` random training sets of k recordings (seeded); each set's model is scored on every labelled
recording outside it. One fit per set, rather than one per set per held-out recording, so the curve costs about
`sets` fits per k. Written as `learning-curve.json` beside the model, for the app's Model page.
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


def run(recs: list[Recording], out: Path, sets: int = 6) -> None:
    data = [E.Data(r) for r in recs if r.features(E.RUST).with_suffix(".json").exists()]
    lab = [d for d in data if (d.y != IGNORE).any()]
    rng = random.Random(0)
    runs = []
    fits = 0
    from . import runs

    for k in range(1, len(lab)):
        runs.progress(f"learning curve {k}/{len(lab) - 1}")
        combos = list(itertools.combinations(range(len(lab)), k))
        for c in rng.sample(combos, min(sets, len(combos))):
            train = [lab[i] for i in c]
            # Out-of-fold frame probabilities for the trees, as the shipped model; with one recording there is no fold.
            if k >= 2:
                oof = {o.rec.id: E.frame_model([q for q in train if q is not o]).predict_proba(o.x)[:, 1] for o in train}
            else:
                oof = {o.rec.id: E.frame_model(train).predict_proba(o.x)[:, 1] for o in train}
            trees = E.fit_trees(train, oof)
            frames = E.frame_model(train)
            fits += 1
            seconds = int(sum((o.y != IGNORE).sum() for o in train))
            for i, test in enumerate(lab):
                if i in c:
                    continue
                m = test.y != IGNORE
                y = test.y[m] == BREAK
                if not y.any() or y.all():
                    continue
                p = trees.predict_proba(E.table(test, frames.predict_proba(test.x)[:, 1]))[:, 1][m]
                runs.append({"test": test.rec.id, "k": k, "seconds": seconds,
                             "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 5),
                             "auc": round(float(roc_auc_score(y, p)), 5), "f1": round(float(f1_score(y, p >= 0.5)), 5)})

    def q(v: list[float], p: float) -> float:
        return float(np.quantile(v, p))

    points = []
    for k in sorted({r["k"] for r in runs}):
        r = [x for x in runs if x["k"] == k]
        points.append({"k": k, "runs": len(r), "hours": float(np.median([x["seconds"] for x in r])) / 3600,
                       **{m: [q([x[m] for x in r], p) for p in (0.25, 0.5, 0.75)] for m in ("log_loss", "f1", "auc")}})
    out.mkdir(parents=True, exist_ok=True)
    doc = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "recordings": len(lab),
           "sets": sets, "fits": fits, "points": points, "runs": runs}
    tmp = out / ".learning-curve.json.tmp"
    tmp.write_text(json.dumps(doc))
    tmp.replace(out / "learning-curve.json")
    for p in points:
        print(f"{p['k']} recordings: log loss {p['log_loss'][1]:.3f}, F1 {p['f1'][1]:.3f} ({p['runs']} runs)")
