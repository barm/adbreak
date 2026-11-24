# adbreak agent guide

Commercial break detection for TV recordings. Consumed by barm/dvr as a Cargo git dependency (only the
`adbreak` library), so a change here reaches the DVR only when its `Cargo.toml` pin moves. **adbreak never
depends on the DVR**, at build time or at run time: recordings arrive as import directories.

- `rust/` Cargo workspace (run cargo there):
  - `crates/adbreak` the detector: `Detector::push_frame` / `push_audio` / `finish` / `breaks`. Pure Rust, no
    dependencies, `#![forbid(unsafe_code)]`; keep it that way (decoding belongs to the caller).
  - `crates/adbreak-captions` caption hints; `crates/adbreak-labels` dataset format and `score`;
    `crates/adbreak-cli` the `adbreak` command (`import`, `suggest`, `label`, `eval`); macOS, decoding and the
    small copy through barm/avkit (`fmp4-scan`, `fmp4-shrink`, pinned by rev; bump it to take changes). No ffmpeg.
  - New self-contained functionality gets its own crate.
- `web/label.html` the labelling page: plain JS, no build step, embedded with `include_str!`.
- `ml/` Python training code.
- `dataset/<date>_<channel>_<job>/` recording.json, labels.json, suggestions.json; commit them. The copy
  (`media.mp4`, `index.m3u8`) and captions live in the media store (`ADBREAK_MEDIA`, default `~/Movies/adbreak`) and are never committed: the repository is
  public and broadcasts can't be redistributed.
- Comskip 0.83 doesn't build against ffmpeg 8+: `ticks_per_frame` is gone. In `mpeg2dec.c`, replace the
  `TICKS_PER_FRAME` macro's use of it with a `static int ticks_override` (set where it assigned
  `ticks_per_frame = 1`), then `./autogen.sh && ./configure && make` with Homebrew's include and lib paths.
- Before committing (in `rust/`): `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`.
- Land changes through squash-merged PRs. CI runs on GitHub Actions (public repo).
