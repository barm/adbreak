# adbreak

Find the commercial breaks in recordings of over-the-air US television.

Broadcasts carry no markers for their ad breaks, so they have to be found in the picture, the sound and the
captions. adbreak is a small model that does that, the Rust code to run it inside a recorder, and the tools and
labelled data to train and measure it.

## How it works

For each second of a recording, adbreak computes a few inputs, all from code in this repository:

- an **embedding of one picture** from that second (MobileCLIP2-S0's image encoder, through Core ML), the whole
  picture squashed to 256×256 so the network's corner logo stays in view;
- what the **captions** are doing: whether the recording has any, whether a cue is on screen, whether it rolled up
  from the one before (live shows) or popped on (ads and pre-produced shows), whether it just started; and what
  they **say**: the words, hashed, read by two small learned models, one asking whether the caption on screen is a
  commercial's and one whether what was said in the last 20 s means a break is coming ("we'll be right back");
- how much of the second was **black** and how much **silent**, and how far it is to the nearest moment that was
  both, which networks put around breaks.

Then:

1. a logistic regression on the embedding gives each second a break probability from the picture alone;
2. that, the captions, the two text scores and the black/silence inputs, averaged over windows from ±2 s to ±150 s,
   go into gradient-boosted trees for the final probability;
3. a threshold, short gaps closed, short runs dropped, and each break's ends snapped to a black-and-silent moment
   within 20 s, or else to a black picture.

Training is in Python (`ml/`); the trained model is a `model.json` that `rust/crates/adbreak-model` runs, with
results identical to the Python code's (checked by a test vector).

## How well

Scored on 8 labelled recordings (134 breaks, about 22 hours: news, talk and court shows, game shows, an NFL game;
NBC, CBS, FOX, ABC and an independent station), each recording by a model trained only on the others, so every
score is also on a network or show the model didn't see in training:

| | precision | recall | F1 | median edge error |
|---|---|---|---|---|
| this model | 94% | 96% | 95–96% | 1.2 s |
| [Comskip](https://github.com/erikkaashoek/Comskip) | 97% | 68% | 80% | |
| `logo-detector`: black-and-silent edges plus the corner logo, no learning (`crates/adbreak`, the first approach) | 77% | 29% | 43% | |

Precision and recall are by the second; F1 combines them. Comskip rarely marks programme as a break but cuts breaks
short and misses some; the logo approach fails on networks without a corner logo (CBS) and when the logo isn't
learned. The labels were made by an AI assistant from the recordings' frames and haven't all been checked by a
person yet (each file says how it was made).

Running it costs about 4 ms of Core ML per second of video (one picture a second) on an Apple Silicon Mac, plus the
decoding, which a recorder does anyway.

## Layout

- `rust/`: a Cargo workspace.
  - `crates/adbreak-model`: the model: per-second inputs to break probabilities and breaks, from `model.json`.
    Pure Rust.
  - `crates/adbreak-features`: the per-second inputs, from decoded pictures, sound and captions (Core ML for the
    embeddings with the `coreml` feature, macOS). A recorder feeds it what it decodes, as it records.
  - `crates/adbreak-captions`: captions (WebVTT) and what is on screen each second.
  - `crates/adbreak`: black, silence and their edges (and the first, logo-based detector). Pure Rust, no
    dependencies.
  - `crates/adbreak-labels`: the dataset format (recordings, labels, suggestions) and scoring.
  - `crates/adbreak-cli`: the `adbreak` command, below.
- `ml/`: Python for training and export (see `ml/README.md`).
- `web/`: the Adbreak app (labelling, and the Model page), a React app on the Astryx design system. Optional and separate from the Rust code:
  build it with `cd web && npm ci && npm run build`, and `adbreak label` serves `web/dist` (or `--web <dir>`).
- The dataset lives in a SQLite database (`ADBREAK_DB`, default `~/Library/Application Support/adbreak/adbreak.db`;
  the `adbreak-db` crate): each recording's metadata, its labels (every save kept) and what each automatic source
  said. The app saves labels there as you go and training reads and writes the same database. `dataset/` in this
  repository is a **snapshot**: `adbreak dataset export` writes one (one directory per recording: `recording.json`,
  `labels.json`, `suggestions.json`) to commit or publish; `adbreak dataset import` reads one into the database. No
  media: recordings live in the media store (`ADBREAK_MEDIA`, default `~/Movies/adbreak`), outside the repository,
  because broadcasts can't be redistributed.
- Backups: `adbreak dataset backup` writes a consistent copy of the database, label history included, to `backups/`
  beside it (the newest 30 kept). `scripts/backup.sh` does that and also commits a snapshot to the
  `dataset-snapshots` branch and pushes it when the dataset changed; run it nightly (a launchd agent, cron).

Decoding and the small copies use [barm/avkit](https://github.com/barm/avkit) (VideoToolbox, AudioToolbox, Core
ML), so the tools run on macOS.

## Using the model in a recorder

```toml
adbreak-model = { git = "https://github.com/barm/adbreak", rev = "..." }
adbreak-features = { git = "https://github.com/barm/adbreak", rev = "...", features = ["coreml"] }
```

Load `model.json` with `adbreak_model::Model::from_json` and the image encoder with
`adbreak_features::CoreMlEmbedder::load(path, 256)`; feed an `Extractor` small grey pictures (480×270, 10 a
second) and mono sound for black and silence, and one 256×256 colour picture a second (avkit's
`fmp4_scan::Scan::with_color` produces both); then `model.breaks(&extractor.inputs(captions_vtt))`. It can run
again on what has arrived so far, so breaks can show up during a live recording.

## Building the dataset

```sh
cargo install --path rust/crates/adbreak-cli

# All at once: import, the automatic sources, the model's inputs and its draft (the steps below).
adbreak ingest /path/to/export

# Or step by step. Add a recording from an import directory (below): keeps a 360p copy and the captions in the media store.
adbreak import /path/to/export

# What the automatic sources think (Comskip if installed, the captions, the logo detector), and the model's inputs.
adbreak suggest 2026-11-24_2.1_32 --comskip ~/src/comskip/comskip
adbreak features 2026-11-24_2.1_32

# Review in a browser: the recording, every source on a timeline, the labels.
adbreak label            # http://127.0.0.1:8585

# Score every source against the labels.
adbreak eval --each
```

A recorder can hand recordings over without anyone running a command: `adbreak label` watches an inbox (`--inbox`,
`ADBREAK_INBOX`, default `inbox/` beside the database) and ingests each import directory that appears there, one at a
time, deleting it once in (a failed one is renamed `<name>.failed`; the Model page shows the progress). Write each
directory under a name starting with `.` and rename it when complete.

An import directory holds `recording.json` (`adbreak_labels::Recording`), `media.mp4` (a fragmented MP4 with HEVC
video and AAC sound), and optionally `captions.vtt`, `suggestions.json` (sources the recorder already has, such as
its own break scan) and `labels.json`. Recordings are named by their dataset id or any unique part of it. Every command uses
the database (`--db`, `ADBREAK_DB`); start one from the snapshot with `adbreak dataset import dataset`.

On the labelling page, `I` and `O` mark a label's start and end, `A` accepts a selected suggestion and `N`/`P` jump
between the sources' edges. `[` and `]` (or dragging across the bottom row) choose a stretch: `X` excludes it (bad
quality: never scored or trained on), `⌫` includes it again; as many as needed. When its labels are done, the
**Ready for training** switch at the top makes the whole recording count, less what is excluded. Kinds: `ad` (commercial break), `double_box`, `ad_read` and
`paid_programming`; [docs/labels.md](docs/labels.md) defines them and how to place them. `adbreak eval` scores
commercial breaks, and shows how much double box and ad read time each source marked.

The copy is 640×360 HEVC with every picture (break edges are a few pictures long), mono AAC, and a byte-range
playlist to seek by: about 0.6 GB for 3.5 hours. Comskip finds the same breaks in it as in the original, to 0.02 s.

## Next

- More recordings, especially sports, and a person's pass over the assistant's labels.
- Corrections from use: breaks fixed in a recorder's player, as labels; skips undone or ads scrubbed past, as hints.
- Caption text (a small classifier on what the caption says) in the Rust model: worth about a point of F1 in
  Python, but it needs CLIP's tokenizer in Rust.

## License

This project is dual-licensed. You may use it under either of these licenses, whichever you prefer:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

Both grants apply to everyone; the choice between them is yours as the user, and there is nothing to ask for.

Unless you say otherwise, any contribution you intentionally submit for inclusion in this project, as defined in
the Apache-2.0 license, is dual-licensed as above, with no additional terms or conditions.
