# adbreak

Find the commercial breaks in recordings of over-the-air US television.

Broadcasts carry no markers for their ad breaks, so they have to be found in the picture, the sound and the
captions. This repository holds the detector and, as they're built, the tools to measure and improve it: labelled
recordings, an evaluation harness and a trained model.

It started inside [barm/dvr](https://github.com/barm/dvr), which still uses it to mark breaks for "Skip break" and
"Remove commercials".

## Layout

- `crates/adbreak`: the detector library. Pure Rust, no dependencies, `#![forbid(unsafe_code)]`. You decode;
  it takes 8-bit luma frames and mono audio samples with their timestamps and returns breaks.

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

1. **Labels and evaluation.** A labelling tool merges what Comskip, the captions and this detector say about a
   recording and shows frames wherever they disagree, to settle by eye. Labels are small JSON files kept in this
   repository; `eval` scores any detector against them.
2. **Features.** The DVR writes per-second features (black, silence, logo, captions, later image embeddings) next to
   each recording as it records, so labelling, evaluation and training don't need to decode multi-gigabyte video.
3. **Corrections from use.** Breaks adjusted in the DVR's player become labels; skips undone or ads scrubbed past
   become hints for review.
4. **A model.** A small sequence model over the features, trained in Python and run through Core ML.

## License

MIT or Apache-2.0, at your option.
