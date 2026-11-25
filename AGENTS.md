# adbreak agent guide

Commercial break detection for TV recordings: a model, the Rust code to run it in a recorder, and the tools and
labelled data to train it. Recorders use the crates as Cargo git dependencies pinned by rev, so a change reaches
them only when their pin moves. adbreak depends on no recorder, at build time or run time: recordings arrive as
import directories.

**This repository is public.** Don't mention private projects, hosts, people's machines or paths in code, docs,
data or commit messages. A recording's source is `"recorder"`.

- `rust/` Cargo workspace (run cargo there):
  - `crates/adbreak-model` the model (`model.json` from `ml/`): per-second `Row`s to probabilities and breaks.
    Pure Rust. Its test checks it against the Python export (`ADBREAK_MODEL_DIR` with model.json and
    test_vector.json from `uv run adbreak-ml export DIR`): keep them identical; change both sides together.
  - `crates/adbreak-features` the per-second inputs from decoded pictures, sound and captions; Core ML embeddings
    with the `coreml` feature (avkit's `apple-coreml`). Training and recorders both use it.
  - `crates/adbreak-captions` WebVTT, `at_seconds` (the model's caption inputs) and caption hints.
  - `crates/adbreak` black, silence, edges and the first logo-based detector. Pure Rust, no dependencies,
    `#![forbid(unsafe_code)]`; keep it that way.
  - `crates/adbreak-labels` dataset format and `score`; `crates/adbreak-cli` the `adbreak` command (`import`,
    `suggest`, `features`, `label`, `eval`); macOS, decoding through barm/avkit (pinned by rev). No ffmpeg.
  - New self-contained functionality gets its own crate.
- `ml/` Python (uv): `features`/`baseline`/`fusion` (experiments on Python-computed features), `evaluate` and
  `export` (the shipped model, on the Rust-computed `rust-v1` features). See `ml/README.md`.
- `web/` the Adbreak app: Recordings (labelling) and Model (the installed model, its training and leave-one-out
  scores, and actions that run `adbreak-ml` in `ml/` through uv: `adbreak label --ml`, `--model`; `src/model.rs` in the
  CLI). React + Vite on the Astryx design system (follow `web/AGENTS.md`). Optional and
  decoupled from Rust: `adbreak label` serves its build from `web/dist` at run time (`--web`), and says how to build
  it when there is none. `src/editor/engine.ts` is the editing engine (canvas timeline, keys, undo, saving), ported
  from the plain-JS page; the React components around it are Astryx. Before committing web changes:
  `cd web && npm run typecheck && npm run build`.
- `dataset/<date>_<channel>_<job>/` recording.json, labels.json, suggestions.json; commit them. Label kinds
  and how to place them: `docs/labels.md`; keep it, `adbreak-labels`' `Kind`, `ml/adbreak_ml/data.py` and
  `web/src/editor/engine.ts` in step. The copy
  (`media.mp4`, `index.m3u8`), captions and features live in the media store (`ADBREAK_MEDIA`, default
  `~/Movies/adbreak`) and are never committed: broadcasts can't be redistributed.
- Comskip 0.83 doesn't build against ffmpeg 8+: `ticks_per_frame` is gone. In `mpeg2dec.c`, replace the
  `TICKS_PER_FRAME` macro's use of it with a `static int ticks_override` (set where it assigned
  `ticks_per_frame = 1`), then `./autogen.sh && ./configure && make` with Homebrew's include and lib paths.
- Before committing (in `rust/`): `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`.
- Land changes through squash-merged PRs. CI (GitHub Actions) runs fmt, tests and clippy as separate checks.
