"""adbreak-ml: embed the dataset's recordings, then run the baselines.

    uv run adbreak-ml embed        # frame embeddings for every recording with a copy in the media store
    uv run adbreak-ml baseline     # zero-shot and leave-one-recording-out linear classifier; writes suggestions
"""

import argparse

from . import baseline, embed
from .data import recordings


def main() -> None:
    ap = argparse.ArgumentParser(prog="adbreak-ml")
    ap.add_argument("command", choices=["embed", "baseline"])
    a = ap.parse_args()
    recs = [r for r in recordings() if r.media.is_file()]
    if a.command == "embed":
        model = embed.load()
        for r in recs:
            embed.embed(r, model)
    else:
        baseline.run([r for r in recs if r.features(embed.NAME).exists()])
