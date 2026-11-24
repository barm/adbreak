# adbreak

Find the commercial breaks in recordings of over-the-air US television.

Broadcasts carry no markers for their ad breaks, so they have to be found in the picture, the sound and the
captions. This repository holds the detector and, as they're built, the tools to measure and improve it: labelled
recordings, an evaluation harness and a trained model.

It started inside [barm/dvr](https://github.com/barm/dvr), which still uses it to mark breaks for "Skip break" and
"Remove commercials".

## Layout

- `rust/`: a Cargo workspace.
  - `crates/adbreak`: the detector library. Pure Rust, no dependencies, `#![forbid(unsafe_code)]`. You decode;
    it takes 8-bit luma frames and mono audio samples with their timestamps and returns breaks.
  - `crates/adbreak-captions`: break hints from closed captions (WebVTT).
  - `crates/adbreak-labels`: the dataset format (recordings, labels, suggestions) and scoring.
  - `crates/adbreak-cli`: the `adbreak` command, below.
- `web/`: the labelling page (`label.html`, plain JS, built into the `adbreak` binary).
- `ml/`: Python for training (to come).
- `dataset/`: one directory per recording: `recording.json`, `labels.json` (a person's) and `suggestions.json`
  (the automatic sources'). No media: that lives in the media store, outside the repository, because broadcasts
  can't be redistributed.

adbreak depends on no recorder. Recordings come in as files; [barm/dvr](https://github.com/barm/dvr) writes them
with `dvr export`, and depends on the `adbreak` library, not the other way round. Media work (the small copy, the
detector's decoding) is done in hardware by [barm/avkit](https://github.com/barm/avkit), on macOS; no ffmpeg.

## Building the dataset

```sh
cargo install --path rust/crates/adbreak-cli

# Add a recording: the DVR writes an import directory (below), adbreak keeps a 360p copy and the captions in the
# media store (ADBREAK_MEDIA, default ~/Movies/adbreak) and its metadata in dataset/.
dvr export 32 --to /tmp/export-32
adbreak import /tmp/export-32

# What the automatic sources think: Comskip (if installed), the captions, this detector.
adbreak suggest 32 --comskip ~/src/comskip/comskip

# Review in a browser: the recording, every source on a timeline, your labels.
adbreak label            # http://127.0.0.1:8585

# Score every source against the labels.
adbreak eval --each
```

Recordings are named by their dataset id (`2026-11-24_2.1_32`) or any unique part of it, such as the job number.
Run the commands from the repository root so `dataset/` is this repository's (or pass `--dataset`).

An import directory holds `recording.json` (`adbreak_labels::Recording`), `media.mp4` (a fragmented MP4 with HEVC
video and AAC sound, as barm/dvr records), and optionally `captions.vtt`, `suggestions.json` (sources the recorder already has, such as its
own break scan) and `labels.json` (labels it collected, such as corrections made while watching).

On the page, `[` and `]` mark a break's start and end, `A` accepts a selected suggestion, `N`/`P` jump between
the sources' edges and `R` marks everything up to the playhead as reviewed: only reviewed time is scored, so a
recording can be labelled a part at a time. Kinds: `ad`, `promo` (the station's own), and `paid_programming`
(infomercials; never scored).

The copy is 640×360 HEVC with every picture (break edges are a few pictures long), mono AAC, and a byte-range
playlist to seek by: about 0.6 GB for a 3.5-hour recording, made at about 14× real time. Comskip finds the same
breaks in it as in the original, to 0.02 s. Comskip is optional and needs building from source; see `AGENTS.md`.

## Where it stands

The current detector finds a break at a moment that is black and silent at once, then uses the network's corner
logo to tell programme from ad. Checked by hand against frames from two recordings (November 2026):

| Recording | Real breaks | Found |
|---|---|---|
| FOX, 45 min of news | 5 | 4 |
| FOX, 3.5 h | 19 | 0: no logo learned over the whole recording |
| CBS, 45 min | 5 | 0: CBS shows no corner logo |

[Comskip](https://github.com/erikkaashoek/Comskip) did much better on the same recordings but cut two CBS breaks
short. Captions catch what it misses: programmes are 81–93% captioned and breaks 42–49%. Roll-up captions
(live shows) against pop-on (ads) only helps on live programming, since pre-produced shows caption pop-on too.

## Plan

1. **Labels and evaluation** (built). Label recordings against the sources' suggestions; `eval` scores any
   detector against the labels.
2. **Corrections from use.** Breaks fixed in the DVR's player come in as labels around the fixed break; skips
   undone or ads scrubbed past as weak hints.
3. **A frame model.** A pretrained image backbone with a small classifier on frames sampled once or twice a second.
4. **Fusion.** A small sequence model over the frame model's probabilities, captions and black-and-silent edges,
   with break edges snapped to the detector's frame-accurate edges. Trained in Python, run through Core ML.

## License

MIT or Apache-2.0, at your option.
