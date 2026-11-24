# adbreak agent guide

Commercial break detection for TV recordings. Consumed by barm/dvr as a Cargo git dependency, so a change here
reaches the DVR only when its `Cargo.toml` pin moves.

- `crates/adbreak` the detector: `Detector::push_frame` / `push_audio` / `finish` / `breaks`. Pure Rust, no
  dependencies, `#![forbid(unsafe_code)]`; keep it that way (decoding belongs to the caller).
- New self-contained functionality gets its own crate under `crates/` (labelling, evaluation, features).
- Before committing: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
- Land changes through squash-merged PRs. CI runs on GitHub Actions (public repo).
- Recordings are never committed; labels and features reference them by path or job id.
