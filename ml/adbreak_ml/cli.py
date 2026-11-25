"""adbreak-ml: features for the dataset's recordings, baselines, and the fusion model.

    uv run adbreak-ml features     # frame embeddings, caption and black/silence features (cached in the media store)
    uv run adbreak-ml baseline     # frame-only baselines, leave-one-recording-out; written as suggestions
    uv run adbreak-ml refresh      # retrain every learned source against the current labels (after labelling)
    uv run adbreak-ml export DIR   # the model for Rust (adbreak-model): model.json and a test vector
    uv run adbreak-ml fusion       # the fusion model: scored leave-one-recording-out on labelled recordings,
                                   # and a first draft for unlabelled ones; written as the "fusion" suggestion
"""

import argparse
import json
import warnings

from . import av, baseline, captions, embed, fusion
from .data import load_features, recordings


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="adbreak-ml")
    ap.add_argument("command", choices=["features", "baseline", "fusion", "export", "evaluate", "refresh", "learning-curve", "history", "install"])
    ap.add_argument("out", nargs="?", default=None, help="the model directory (default: ADBREAK_MODEL_DIR, or model/ beside the database)")
    ap.add_argument("--version", help="install: which kept model (see history)")
    ap.add_argument("--only", action="append", default=[], help="evaluate: just these recordings (repeatable)")
    a = ap.parse_args()
    from .data import MODEL

    a.out = a.out or str(MODEL)
    if a.command in ("history", "install"):
        from pathlib import Path

        from . import history

        out = Path(a.out)
        if a.command == "install":
            if not a.version:
                raise SystemExit("install needs --version (see adbreak-ml history)")
            history.install(out, a.version)
        else:
            installed = json.loads((out / "model-info.json").read_text()).get("version") if (out / "model-info.json").is_file() else None
            for i in history.entries(out):
                print(f"{i['version']}  {i.get('exported_at', '')}  {len(i.get('trained_on', []))} recordings")
            print(f"installed: {installed}")
        return
    recs = [r for r in recordings() if r.media.is_file()]
    if a.command == "features":
        import open_clip

        model = embed.load()
        tok = open_clip.get_tokenizer(embed.MODEL)
        for r in recs:
            embed.embed(r, model)
            t, _ = load_features(r, embed.NAME)
            if not r.features(captions.NAME).exists():
                captions.build(r, t, model, tok)
            if not r.features(av.NAME).exists():
                av.build(r, t)
                print(f"  {r.id}: captions and black/silence")
    elif a.command == "export":
        from pathlib import Path

        from . import export

        export.export([r for r in recs if r.labels], Path(a.out))
    elif a.command == "refresh":
        # Every learned source, retrained leave-one-out against the current labels: after labelling or adding a recording.
        from . import export

        labelled = [r for r in recs if r.labels]
        print("baseline (frames-lr, frames-zeroshot)")
        baseline.run([r for r in labelled if r.features(embed.NAME).exists()])
        print("fusion")
        ready = [r for r in recs if all(r.features(n).exists() for n in (embed.NAME, captions.NAME, av.NAME))]
        fusion.run(ready, embed.NAME)
        print("model")
        export.evaluate(recs)
    elif a.command == "evaluate":
        from . import export

        export.evaluate(recs, only=set(a.only) or None)
    elif a.command == "learning-curve":
        from pathlib import Path

        from . import learning_curve

        learning_curve.run([r for r in recs if r.labels], Path(a.out))
    elif a.command == "baseline":
        baseline.run([r for r in recs if r.features(embed.NAME).exists() and r.labels])
    else:
        ready = [r for r in recs if all(r.features(n).exists() for n in (embed.NAME, captions.NAME, av.NAME))]
        for k, v in fusion.run(ready, embed.NAME).items():
            print(k, " ".join(f"{a} {b:.3f}" for a, b in v.items()) or "draft written (no labels yet)")
