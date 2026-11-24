# ml

Python for training. Reads `../dataset/` (labels, suggestions) and the media store (`ADBREAK_MEDIA`, default
`~/Movies/adbreak`); caches features there too (`<id>/features/*.npz`), never in git.

```sh
cd ml
uv sync
uv run adbreak-ml features   # frame embeddings (MobileCLIP2-S0, 1 fps, GPU), caption and black/silence features
uv run adbreak-ml baseline   # frame-only baselines, leave-one-recording-out; written as suggestions
uv run adbreak-ml fusion     # the fusion model; written as the "fusion" suggestion
cd .. && adbreak eval --each # scores them next to Comskip
```

- Frames are the whole picture squashed to 256×256, not centre-cropped: the network logo is in a corner.
- Per-second labels: break (ad or promo), programme, or ignored (unreviewed, paid programming, within 1 s of an
  edge). See `adbreak_ml/data.py`.
- Every score is leave-one-recording-out: a recording's predictions come from a model that never saw it. With one
  recording per network so far, that is also a test on an unseen network.

Decoding is ffmpeg (Homebrew) for now; it should move to avkit so training sees the same pictures the DVR will.

## Fusion

`adbreak_ml/fusion.py` turns per-second evidence into breaks:

- the frame classifier's break probability (each recording scored by a classifier trained on the others);
- the caption on screen: whether captions exist, whether one is visible, roll-up or pop-on, a new cue, and a
  break probability from a classifier on the caption's text (same leave-one-out rule);
- black, silence, and the black-and-silent moments networks put between ads (distance to the nearest, how many
  nearby);
- all of it averaged over windows of ±2 s to ±150 s, into gradient-boosted trees; then a threshold, short gaps
  closed, and each break's edges snapped to the nearest black-and-silent moment within 8 s.

Training also sees each captioned recording with its captions removed, so the model learns to decide without them
(some recordings have none). Unlabelled recordings get a first draft from models trained on every labelled one.

## Results so far (3 recordings, 65 breaks, every score leave-one-recording-out)

| source | NBC game | FOX news | CBS late night | all (F1) |
|---|---|---|---|---|
| Comskip | 95% | 91% | 74% | 87% |
| frames + logistic regression | 79% | 79% | 79% | 79% |
| **fusion** | 89% | 94% | 96% | **93%** |

Fusion: precision 95%, recall 91%; mean edge error 11 s at the start, 19 s at the end.
