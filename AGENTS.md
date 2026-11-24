# adbreak agent guide

Commercial break detection for TV recordings. Consumed by barm/dvr as a Cargo git dependency, so a change here
reaches the DVR only when its `Cargo.toml` pin moves.

- `crates/adbreak` the detector: `Detector::push_frame` / `push_audio` / `finish` / `breaks`. Pure Rust, no
  dependencies, `#![forbid(unsafe_code)]`; keep it that way (decoding belongs to the caller).
- `crates/adbreak-captions` caption hints, `crates/adbreak-labels` dataset format and `score`, `crates/adbreak-cli`
  the `adbreak` command (`suggest`, `label` with its page in `src/label.html`, plain JS, no build step; `eval`).
- `dataset/<date>_<channel>_<job>/` labels.json and suggestions.json; commit both. Never commit media.
- Comskip 0.83 doesn't build against ffmpeg 8+: `ticks_per_frame` is gone. In `mpeg2dec.c`, replace the
  `TICKS_PER_FRAME` macro's use of it with a `static int ticks_override` (set where it assigned
  `ticks_per_frame = 1`), then `./autogen.sh && ./configure && make` with Homebrew's include and lib paths.
- New self-contained functionality gets its own crate under `crates/` (labelling, evaluation, features).
- Before committing: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
- Land changes through squash-merged PRs. CI runs on GitHub Actions (public repo).
- Recordings are never committed; labels and features reference them by path or job id.
