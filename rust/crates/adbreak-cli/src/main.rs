//! `adbreak`: build a dataset of labelled commercial breaks and score detectors against it.
//!
//! Works on files only; any recorder can feed it by writing an import directory (see `import`).
//!
//! - `adbreak import <dir>` adds a recording: a low-resolution copy and its captions to the media store,
//!   its metadata (and any suggestions or labels it came with) to the dataset.
//! - `adbreak suggest <id>` asks the automatic sources (Comskip, the captions, the logo detector).
//! - `adbreak label` serves a page to review recordings against the suggestions and save labels.
//! - `adbreak eval` scores every source against the labels.

<<<<<<< HEAD
mod cuts;
=======
mod bench;
>>>>>>> f332fa1 (adbreak bench: time the break scan against Comskip on the same file)
mod dataset;
mod features;
mod label;
mod model;
mod tools;

use adbreak_labels::{Kind, Labels, Recording, Score, Source, Span, Suggestions, score};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use dataset::Dataset;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(version, about = "Build a dataset of labelled commercial breaks and score detectors against it")]
struct Cli {
    /// The dataset: recordings, labels and suggestions (SQLite; `adbreak dataset export` writes a snapshot for git).
    #[arg(long, env = "ADBREAK_DB", default_value_os_t = adbreak_db::default_path(), global = true)]
    db: PathBuf,
    /// The media store: each recording's low-resolution copy and captions (never in git).
    #[arg(long, env = "ADBREAK_MEDIA", default_value_os_t = home().join("Movies/adbreak"), global = true)]
    media: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Snapshots of the dataset in the directory layout kept in git (one directory per recording:
    /// recording.json, labels.json, suggestions.json).
    Dataset {
        #[command(subcommand)]
        cmd: DatasetCmd,
    },
    /// Add a recording from an import directory: `recording.json` (adbreak_labels::Recording), `media.mp4` (a
    /// fragmented MP4, HEVC and AAC), and optionally `captions.vtt`, `suggestions.json` and
    /// `labels.json`.
    Import {
        dirs: Vec<PathBuf>,
        /// Height of the copy kept, in lines.
        #[arg(long, default_value_t = 360)]
        height: u32,
    },
    /// Everything a new recording needs before labelling, in one step: import it, ask the automatic sources (as
    /// `suggest`), compute the model's inputs (as `features`) and have the model draft its breaks (`adbreak-ml
    /// evaluate --only`, if the ML project is there).
    Ingest {
        /// Import directories (as for `import`).
        #[arg(required = true)]
        dirs: Vec<PathBuf>,
        /// Comskip binary (default: `comskip` on PATH; skipped if there is none).
        #[arg(long, env = "ADBREAK_COMSKIP")]
        comskip: Option<PathBuf>,
        /// Directory with the image model.
        #[arg(long, env = "ADBREAK_MODEL_DIR", default_value_os_t = adbreak_db::default_model_dir())]
        model_dir: PathBuf,
        /// The ML project (`ml/`), for the model's draft.
        #[arg(long, env = "ADBREAK_ML", default_value = "ml")]
        ml: PathBuf,
        /// Delete each import directory once it is in.
        #[arg(long)]
        remove: bool,
    },
    /// Ask Comskip, the captions and the logo detector where a recording's breaks are.
    Suggest {
        /// Recordings: their dataset id, or a unique part of it such as the job number.
        #[arg(required = true)]
        ids: Vec<String>,
        /// Comskip binary (default: `comskip` on PATH; skipped if there is none).
        #[arg(long, env = "ADBREAK_COMSKIP")]
        comskip: Option<PathBuf>,
        /// Only the captions: no decoding.
        #[arg(long)]
        quick: bool,
    },
    /// Compute the per-second model inputs from a recording's copy (picture embeddings, captions, black and
    /// silence), as a recorder computes them, for training: `features/rust-v1.f32` and `.json` in the media store.
    Features {
        #[arg(required = true)]
        ids: Vec<String>,
        /// Directory with the image model (`mobileclip2-s0-image.mlpackage`).
        #[arg(long, env = "ADBREAK_MODEL_DIR", default_value_os_t = adbreak_db::default_model_dir())]
        model_dir: PathBuf,
        /// Compute again even if the files exist.
        #[arg(long)]
        force: bool,
        /// The full-resolution recording (one id only): its inputs are written as `full-v1`.
        #[arg(long)]
        full: Option<PathBuf>,
    },
    /// Find the shot cuts in recordings' copies (every picture), for the labelling page: `features/cuts.json`.
    Cuts {
        #[arg(required = true)]
        ids: Vec<String>,
        /// Find them again even if the file exists.
        #[arg(long)]
        force: bool,
    },
    /// Serve the labelling page.
    Label {
        /// Address to listen on. Anyone who can reach it can watch the recordings: keep it on loopback (publish it
        /// with something like `tailscale serve` if needed).
        #[arg(long, default_value = "127.0.0.1:8585")]
        bind: String,
        /// The labelling page: the web app's build (`cd web && npm run build`). Optional: without it the CLI works
        /// and the page says how to build it.
        #[arg(long, env = "ADBREAK_WEB", default_value = "web/dist")]
        web: PathBuf,
        /// The installed model, for the Model page (what `adbreak-ml export` writes and a recorder loads).
        #[arg(long, env = "ADBREAK_MODEL_DIR", default_value_os_t = adbreak_db::default_model_dir())]
        model: PathBuf,
        /// The ML project (`ml/`), run through uv by the Model page's actions. Optional: without it they're off.
        #[arg(long, env = "ADBREAK_ML", default_value = "ml")]
        ml: PathBuf,
        /// Import directories dropped here (by a recorder) are ingested on their own, one at a time, and deleted
        /// once in; one that fails is renamed `<name>.failed`. Default: `inbox/` beside the database. A writer
        /// should fill a directory whose name starts with `.` and rename it when complete.
        #[arg(long, env = "ADBREAK_INBOX")]
        inbox: Option<PathBuf>,
    },
<<<<<<< HEAD
    /// The held-out set: recordings never trained on, which every model is scored on. With none, models are scored
    /// leave-one-recording-out.
    Holdout {
        #[command(subcommand)]
        cmd: Option<HoldoutCmd>,
=======
    /// Time the break scan on a media file against Comskip on the same file (see `bench.rs`).
    Bench {
        /// A recording's media: a fragmented MP4 (HEVC, AAC), a recorder's or a copy in the media store.
        media: PathBuf,
        #[arg(long, env = "ADBREAK_COMSKIP")]
        comskip: Option<PathBuf>,
        #[arg(long, env = "ADBREAK_MODEL_DIR", default_value_os_t = adbreak_db::default_model_dir())]
        model_dir: PathBuf,
>>>>>>> f332fa1 (adbreak bench: time the break scan against Comskip on the same file)
    },
    /// Score each source against the labels.
    Eval {
        /// Show each recording, not just the totals.
        #[arg(long)]
        each: bool,
    },
}

#[derive(Subcommand)]
enum HoldoutCmd {
    /// Hold recordings out (their dataset id, or a unique part of it).
    Add { ids: Vec<String> },
    /// Put recordings back in training.
    Remove { ids: Vec<String> },
}

#[derive(Subcommand)]
enum DatasetCmd {
    /// Write the dataset to a directory (default: dataset/, to commit).
    Export {
        #[arg(default_value = "dataset")]
        dir: PathBuf,
    },
    /// Read a directory into the dataset. Labels already in it are kept unless --replace.
    Import {
        #[arg(default_value = "dataset")]
        dir: PathBuf,
        #[arg(long)]
        replace: bool,
    },
    /// Write a consistent copy of the database (label history included) to a dated file, keeping the newest --keep.
    Backup {
        /// Default: `backups/` beside the database.
        #[arg(long)]
        dir: Option<PathBuf>,
        #[arg(long, default_value_t = 30)]
        keep: usize,
    },
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let data = Dataset::open(&cli.db, cli.media.clone())?;
    match cli.cmd {
        Cmd::Import { dirs, height } => {
            for d in dirs {
                import(&data, &d, height)?;
            }
        }
        Cmd::Ingest { dirs, comskip, model_dir, ml, remove } => {
            let comskip = tools::find(comskip.as_deref(), "comskip");
            if comskip.is_none() {
                eprintln!("no Comskip found (--comskip or ADBREAK_COMSKIP); skipping it");
            }
            for d in dirs {
                let id = import(&data, &d, 360)?;
                suggest(&data, &id, comskip.clone(), true)?;
                features::compute(&data, &id, &model_dir, None, false)?;
                features::compute(&data, &id, &model_dir, Some(&d.join("media.mp4")), false)?;
                cuts::compute(&data, &id, false)?;
                draft(&data, &ml, &id)?;
                if remove {
                    std::fs::remove_dir_all(&d).with_context(|| format!("removing {}", d.display()))?;
                }
                eprintln!("ingested {id}");
            }
        }
        Cmd::Suggest { ids, comskip, quick } => {
            let comskip = if quick { None } else { tools::find(comskip.as_deref(), "comskip") };
            if !quick && comskip.is_none() {
                eprintln!("no Comskip found (--comskip or ADBREAK_COMSKIP); skipping it");
            }
            for name in ids {
                suggest(&data, &data.resolve(&name)?, comskip.clone(), !quick)?;
            }
        }
        Cmd::Cuts { ids, force } => {
            for name in ids {
                cuts::compute(&data, &data.resolve(&name)?, force)?;
            }
        }
        Cmd::Features { ids, model_dir, force, full } => {
            if full.is_some() && ids.len() != 1 {
                bail!("--full is one recording's media: give one id");
            }
            for name in ids {
                features::compute(&data, &data.resolve(&name)?, &model_dir, full.as_deref(), force)?;
            }
        }
        Cmd::Label { bind, web, model, ml, inbox } => {
            let inbox = inbox.unwrap_or_else(|| data.db.path.with_file_name("inbox"));
            label::serve(data, &bind, &web, &model, &ml, inbox).await?
        }
        Cmd::Eval { each } => eval(&data, each)?,
<<<<<<< HEAD
        Cmd::Holdout { cmd } => {
            match cmd {
                Some(HoldoutCmd::Add { ids }) => ids.iter().try_for_each(|i| data.db.set_holdout(&data.resolve(i)?, true))?,
                Some(HoldoutCmd::Remove { ids }) => ids.iter().try_for_each(|i| data.db.set_holdout(&data.resolve(i)?, false))?,
                None => {}
            }
            let held = data.db.holdout()?;
            let secs: f64 = held.iter().filter_map(|i| data.db.recording(i).ok().flatten()).map(|r| r.duration).sum();
            for id in &held {
                println!("{id}  {}", data.db.recording(id)?.map(|r| r.title).unwrap_or_default());
            }
            println!("{} held out, {:.1} h", held.len(), secs / 3600.0);
        }
=======
        Cmd::Bench { media, comskip, model_dir } => bench::run(&media, &model_dir, tools::find(comskip.as_deref(), "comskip").as_deref())?,
>>>>>>> f332fa1 (adbreak bench: time the break scan against Comskip on the same file)
        Cmd::Dataset { cmd: DatasetCmd::Export { dir } } => {
            let n = data.db.export_dir(&dir)?;
            eprintln!("wrote {n} recordings to {}", dir.display());
        }
        Cmd::Dataset { cmd: DatasetCmd::Import { dir, replace } } => {
            let n = data.db.import_dir(&dir, replace)?;
            eprintln!("read {n} recordings from {} into {}", dir.display(), data.db.path.display());
        }
        Cmd::Dataset { cmd: DatasetCmd::Backup { dir, keep } } => {
            let dir = dir.unwrap_or_else(|| data.db.path.with_file_name("backups"));
            println!("{}", data.db.backup(&dir, keep)?.display());
        }
    }
    Ok(())
}

fn import(data: &Dataset, dir: &Path, height: u32) -> Result<String> {
    let mut rec: Recording = dataset::read(&dir.join("recording.json"))?.with_context(|| format!("no recording.json in {}", dir.display()))?;
    let media = dir.join("media.mp4");
    if !media.is_file() {
        bail!("no media.mp4 in {}", dir.display());
    }
    let id = rec.id();
    let out = data.media_dir(&id);
    std::fs::create_dir_all(&out)?;
    eprintln!("{id}: {} on {} {}, {:.0} min: making the {height}p copy", rec.title, rec.channel, rec.network, rec.duration / 60.0);
    rec.duration = tools::shrink(&media, &out, height)?;
    if dir.join("captions.vtt").is_file() {
        std::fs::copy(dir.join("captions.vtt"), out.join("captions.vtt"))?;
    }
    data.save_recording(&rec)?;
    if let Some(s) = dataset::read::<Suggestions>(&dir.join("suggestions.json"))? {
        eprintln!("  suggestions from the recorder: {}", s.sources.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", "));
        data.save_sources(&id, &rec, s.sources)?;
    }
    if let Some(mut l) = dataset::read::<Labels>(&dir.join("labels.json"))? {
        if data.entry(&id)?.labels.is_some() {
            eprintln!("  labels came with it, but the dataset already has labels for {id}: kept the dataset's");
        } else {
            l.recording = rec.clone();
            l.normalise().map_err(anyhow::Error::msg)?;
            data.save_labels(&id, &l)?;
        }
    }
    eprintln!("  added: {id}");
    Ok(id)
}

/// The model's draft of a recording's breaks, through the ML project: `uv run adbreak-ml evaluate --only <id>`.
fn draft(data: &Dataset, ml: &Path, id: &str) -> Result<()> {
    if !ml.join("pyproject.toml").is_file() {
        eprintln!("  no ML project at {} (--ml): no model draft", ml.display());
        return Ok(());
    }
    let uv = std::env::var("ADBREAK_UV").unwrap_or_else(|_| "uv".into());
    let ok = std::process::Command::new(uv)
        .args(["run", "-q", "adbreak-ml", "evaluate", "--only", id])
        .current_dir(ml)
        .env("ADBREAK_DB", &data.db.path)
        .env("ADBREAK_MEDIA", &data.media)
        .env("ADBREAK_RUN_ORIGIN", "ingest")
        .status()
        .context("running uv (ADBREAK_UV)")?
        .success();
    if !ok {
        bail!("the model's draft failed (adbreak-ml evaluate --only {id})");
    }
    Ok(())
}

fn suggest(data: &Dataset, id: &str, comskip: Option<PathBuf>, detector: bool) -> Result<()> {
    let e = data.entry(id)?;
    let rec = &e.recording;
    let dir = data.media_dir(id);
    eprintln!("{id}: {} on {} {}, {:.0} min", rec.title, rec.channel, rec.network, rec.duration / 60.0);
    let mut sources = vec![];
    if let Ok(vtt) = std::fs::read_to_string(dir.join("captions.vtt")) {
        let secs = adbreak_captions::seconds(&adbreak_captions::parse_vtt(&vtt), rec.duration);
        let q = adbreak_captions::Quiet::default();
        let breaks = adbreak_captions::uncaptioned(&secs, &q).into_iter().map(|(a, b)| Span::new(a, b).into()).collect();
        let detail = format!("uncaptioned: window {} s, coverage < {}, at least {} s", q.window, q.max_coverage, q.min_secs);
        sources.push(Source { name: "captions".into(), detail, breaks, trained_on: String::new() });
    }
    if comskip.is_some() || detector {
        let media = dir.join("media.mp4");
        if !media.is_file() {
            bail!("no media for {id} in {} (import it again)", dir.display());
        }
        let work = std::env::temp_dir().join(format!("adbreak-{id}"));
        std::fs::create_dir_all(&work)?;
        let started = std::time::Instant::now();
        let (m1, w1) = (media.clone(), work.clone());
        let cs = comskip.map(|c| std::thread::spawn(move || tools::comskip(&c, &m1, &w1)));
        let det = detector.then(|| std::thread::spawn(move || tools::detect(&media)));
        if let Some(t) = cs {
            let (breaks, version) = t.join().expect("comskip thread")?;
            sources.push(Source { name: "comskip".into(), detail: version, breaks: breaks.into_iter().map(Into::into).collect(), trained_on: String::new() });
        }
        if let Some(t) = det {
            let breaks = t.join().expect("detector thread")?;
            sources.push(Source {
                name: "logo-detector".into(),
                detail: format!("logo detector (crates/adbreak) {}, defaults", env!("CARGO_PKG_VERSION")),
                breaks: breaks.into_iter().map(Into::into).collect(),
                trained_on: String::new(),
            });
        }
        eprintln!("  sources ran in {:.0} s", started.elapsed().as_secs_f64());
        std::fs::remove_dir_all(&work)?;
    }
    for s in &sources {
        eprintln!("  {:9} {:3} breaks, {:5.0} s", s.name, s.breaks.len(), s.breaks.iter().map(|b| b.span().len()).fold(0.0, |a, b| a + b));
    }
    data.save_sources(id, rec, sources)?;
    Ok(())
}

fn eval(data: &Dataset, each: bool) -> Result<()> {
    let mut totals: std::collections::BTreeMap<String, Score> = Default::default();
    let mut labelled = 0;
    // Sources that tell the kinds apart somewhere: each kind is scored on every recording for them, found or not.
    let entries = data.all()?;
    let kinded: std::collections::BTreeSet<String> =
        entries.iter().flat_map(|e| e.suggestions.iter().flat_map(|s| &s.sources)).filter(|s| !s.kinds().is_empty()).map(|s| s.name.clone()).collect();
    // With a held-out set, only its recordings: the learned sources trained on the rest.
    let held = data.db.holdout()?;
    if !held.is_empty() {
        println!("scored on the {} held-out recordings (adbreak holdout)", held.len());
    }
    for e in entries {
        let (Some(labels), Some(sugg)) = (&e.labels, &e.suggestions) else { continue };
        if labels.reviewed.is_empty() || (!held.is_empty() && !held.contains(&e.id)) {
            continue;
        }
        labelled += 1;
        if each {
            println!("{}  ({} labelled breaks)", e.id, labels.breaks.len());
        }
        // "viewer": spots a recorder's viewer corrected a skip, to review; not breaks, so not scored.
        for s in sugg.sources.iter().filter(|s| s.name != "viewer") {
            let sc = score(labels, &s.spans(Kind::Ad));
            if each {
                line(&format!("  {}", s.name), &sc);
            }
            totals.entry(s.name.clone()).or_default().add(&sc);
            // A source that tells the kinds apart: each kind scored as its own row.
            let kinds: &[Kind] = if kinded.contains(&s.name) { &[Kind::DoubleBox, Kind::AdRead, Kind::Bumper, Kind::Credits] } else { &[] };
            for &k in kinds {
                let sc = adbreak_labels::score_kind(labels, k, &s.spans(k));
                let name = format!("{} · {}", s.name, kind_name(k));
                if each {
                    line(&format!("  {name}"), &sc);
                }
                totals.entry(name).or_default().add(&sc);
            }
        }
    }
    if labelled == 0 {
        bail!("no labelled recordings with suggestions in {}", data.db.path.display());
    }
    println!("{labelled} labelled recording(s)");
    println!(
        "  {:15} {:>9} {:>9} {:>9} {:>20} {:>9} {:>9} {:>11} {:>11} {:>12} {:>12}",
        "source", "precision", "recall", "F1", "breaks matched", "start err", "end err", "show cut/h", "ads left/h", "double box", "ad read"
    );
    for (name, sc) in &totals {
        line(&format!("  {name}"), sc);
    }
    Ok(())
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Ad => "break",
        Kind::DoubleBox => "double box",
        Kind::AdRead => "ad read",
        Kind::PaidProgramming => "paid",
        Kind::Bumper => "bumper",
        Kind::Credits => "credits",
    }
}

/// "40% of 120s": how much of some labelled time was marked as break.
fn marked(marked: f64, seconds: f64) -> String {
    if seconds > 0.0 { format!("{:.0}% of {:.0}s", 100.0 * marked / seconds, seconds) } else { "-".into() }
}

fn line(name: &str, s: &Score) {
    let pct = |x: f64| format!("{:.0}%", 100.0 * x);
    println!(
        "{:17} {:>9} {:>9} {:>9} {:>20} {:>8.1}s {:>8.1}s {:>10.0}s {:>10.0}s {:>12} {:>12}",
        name,
        pct(s.precision()),
        pct(s.recall()),
        pct(s.f1()),
        format!("{}/{} ({} found)", s.matched, s.breaks, s.found),
        s.mean_start_error(),
        s.mean_end_error(),
        s.wrongly_skipped_per_hour(),
        s.missed_per_hour(),
        // How much labelled double box and ad read time it marked as break.
        marked(s.double_box_marked, s.double_box_seconds),
        marked(s.ad_read_marked, s.ad_read_seconds),
    );
}
