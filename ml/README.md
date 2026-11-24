# ml

Python for training. Reads `../dataset/` (labels, suggestions) and the media store (`ADBREAK_MEDIA`, default
`~/Movies/adbreak`); caches features there too (`<id>/features/*.npz`), never in git.

```sh
cd ml
uv sync
uv run adbreak-ml embed      # MobileCLIP2-S0 embeddings, 1 frame a second, on the GPU (about 1 min per hour)
uv run adbreak-ml baseline   # zero-shot and leave-one-recording-out linear baselines; writes them as suggestions
cd .. && adbreak eval --each # scores them next to Comskip
```

- Frames are the whole picture squashed to 256×256, not centre-cropped: the network logo is in a corner.
- Per-second labels: break (ad or promo), programme, or ignored (unreviewed, paid programming, within 1 s of an
  edge). See `adbreak_ml/data.py`.
- Every score is leave-one-recording-out: a recording's predictions come from a model that never saw it. With one
  recording per network so far, that is also a test on an unseen network.

Decoding is ffmpeg (Homebrew) for now; it should move to avkit so training sees the same pictures the DVR will.

## Results so far (3 recordings, 65 breaks)

| source | precision | recall | F1 |
|---|---|---|---|
| Comskip | 98% | 78% | 87% |
| frame embeddings + logistic regression | 67% | 97% | 79% |
| frame embeddings, zero-shot prompts | 69% | 82% | 75% |

Per-second AUC of the linear model: 0.88 (NBC game), 0.94 (FOX), 0.92 (CBS). It finds nearly every break but marks
too much; the fusion model (captions, black-and-silent edges, logo) is the next step.
