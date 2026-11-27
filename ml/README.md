# ml

Python for training. Reads the adbreak database (`ADBREAK_DB`, default
`~/Library/Application Support/adbreak/adbreak.db`: recordings, labels, suggestions; it writes its sources there) and
the media store (`ADBREAK_MEDIA`, default `~/Movies/adbreak`); features live there too (`<id>/features/`), never in git.

Whatever a recorder also computes (the trees' input table, breaks from probabilities, combining the kinds, caption
parsing and text tokens) is adbreak-model's Rust, called through `adbreak_rs`; only the learning is Python.

The shipped model trains on the inputs `adbreak features` computes in Rust (`rust-v1`), the same code a recorder
runs, and exports for `rust/crates/adbreak-model`. A recording ingested from a recorder also has `full-v1`, the same
inputs from the full-resolution recording (the pictures a recorder scans; the 360p copy's embeddings differ about five
times as much as the recorder's compression changes them). The picture model learns from both versions, each weighted
half; everything else uses the full-resolution one where there is one.

```sh
adbreak features <ids...>            # from the repository root: per-second inputs, Rust (avkit, Core ML)
adbreak features <id> --full <mp4>  # the same from a full-resolution recording (`full-v1`); ingest does this itself
cd ml && uv sync                     # also builds adbreak_rs (rust/crates/adbreak-py; needs cargo)
uv run adbreak-ml evaluate           # leave-one-recording-out ("model"), and leave-one-network-out ("model-new-network")
uv run adbreak-ml export [DIR]       # model.json and test_vector.json (the image encoder's .mlpackage goes beside it);
                                     # DIR: ADBREAK_MODEL_DIR, default ~/Library/Application Support/adbreak/model.
                                     # The model it replaces is kept in DIR/history/ (the last 10).
uv run adbreak-ml history            # the kept models, named adbreak-0.<n>-<trees> (export --note "what changed", --name for a bigger change)
uv run adbreak-ml install --version V  # put a kept model back (recorders pick it up on their own)
cd .. && adbreak eval --each         # scores every source
```

The commands below are the experiments that led there, on features Python computes itself (ffmpeg decoding).

```sh
cd ml
uv sync
uv run adbreak-ml features   # frame embeddings (MobileCLIP2-S0, 1 fps, GPU), caption and black/silence features
uv run adbreak-ml baseline   # frame-only baselines, leave-one-recording-out; written as suggestions
uv run adbreak-ml fusion     # the fusion model; written as the "fusion" suggestion
cd .. && adbreak eval --each # scores them next to Comskip
```

- Frames are the whole picture squashed to 256×256, not centre-cropped: the network logo is in a corner.
- Per-second labels: break (commercial break), programme (ad reads included), or ignored (unreviewed, double box,
  paid programming, within 1 s of an edge). See `adbreak_ml/data.py` and `docs/labels.md`.
- The shipped model also has an output for each of the other kinds (double box, ad read): a logistic model of the
  picture embedding, class-weighted and averaged over ±5 s. With minutes of examples, trees on the other inputs
  learned the examples rather than the kind (leave-one-out AUC 0.64 against 0.82 for ad reads, 0.76 against 0.94
  for double boxes). `combine` in `export.py` (and `Model::kind_breaks` in Rust) merges them with the breaks.
- Every score is leave-one-recording-out: a recording's predictions come from a model that never saw it. With one
  recording per network so far, that is also a test on an unseen network.

## Fusion

`adbreak_ml/fusion.py` turns per-second evidence into breaks:

- the frame classifier's break probability (each recording scored by a classifier trained on the others);
- the caption on screen: whether captions exist, whether one is visible, roll-up or pop-on, a new cue; and two
  text scores (`adbreak_ml/text.py`, hashed words and word pairs, logistic regression, out-of-fold for the trees):
  whether the caption on screen is a commercial's, and whether the last 20 s of captions say a break is coming
  within 30 s. `adbreak-model`'s `text` module computes them the same way, from the weights in model.json;
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
