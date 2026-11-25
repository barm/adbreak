"""adbreak-ml: features for the dataset's recordings, baselines, and the fusion model.

    uv run adbreak-ml features     # frame embeddings, caption and black/silence features (cached in the media store)
    uv run adbreak-ml baseline     # frame-only baselines, leave-one-recording-out; written as suggestions
    uv run adbreak-ml refresh      # retrain every learned source against the current labels (after labelling)
    uv run adbreak-ml export DIR   # the model for Rust (adbreak-model): model.json and a test vector
    uv run adbreak-ml fusion       # the fusion model: scored leave-one-recording-out on labelled recordings,
                                   # and a first draft for unlabelled ones; written as the "fusion" suggestion
"""

import argparse
import warnings

from . import av, baseline, captions, embed, fusion
from .data import load_features, recordings


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="adbreak-ml")
    ap.add_argument("command", choices=["features", "baseline", "fusion", "export", "evaluate", "refresh"])
    ap.add_argument("out", nargs="?", default="model")
    a = ap.parse_args()
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
        export.evaluate(labelled)
    elif a.command == "evaluate":
        from . import export

        export.evaluate([r for r in recs if r.labels])
    elif a.command == "baseline":
        baseline.run([r for r in recs if r.features(embed.NAME).exists() and r.labels])
    else:
        ready = [r for r in recs if all(r.features(n).exists() for n in (embed.NAME, captions.NAME, av.NAME))]
        for k, v in fusion.run(ready, embed.NAME).items():
            print(k, " ".join(f"{a} {b:.3f}" for a, b in v.items()) or "draft written (no labels yet)")
