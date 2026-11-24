//! `adbreak`: build a dataset of labelled commercial breaks and score detectors against it.
//!
//! Works on files only; any recorder can feed it by writing an import directory (see `import`).
//!
//! - `adbreak import <dir>` adds a recording: a low-resolution copy and its captions to the media store,
//!   its metadata (and any suggestions or labels it came with) to the dataset.
//! - `adbreak suggest <id>` asks the automatic sources (Comskip, the captions, the adbreak detector).
//! - `adbreak label` serves a page to review recordings against the suggestions and save labels.
//! - `adbreak eval` scores every source against the labels.

mod dataset;
mod label;
mod tools;

use adbreak_labels::{Labels, Recording, Score, Source, Span, Suggestions, score};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use dataset::Dataset;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(version, about = "Build a dataset of labelled commercial breaks and score detectors against it")]
struct Cli {
    /// The dataset directory: metadata, labels and suggestions, one directory per recording (kept in git).
    #[arg(long, env = "ADBREAK_DATASET", default_value = "dataset", global = true)]
    dataset: PathBuf,
    /// The media store: each recording's low-resolution copy and captions (never in git).
    #[arg(long, env = "ADBREAK_MEDIA", default_value_os_t = home().join("Movies/adbreak"), global = true)]
    media: PathBuf,
    /// ffmpeg binary (default: `ffmpeg` on PATH or Homebrew's).
    #[arg(long, env = "ADBREAK_FFMPEG", global = true)]
    ffmpeg: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Add a recording from an import directory: `recording.json` (adbreak_labels::Recording), the media
    /// (`media.*`, anything ffmpeg reads), and optionally `captions.vtt`, `suggestions.json` and `labels.json`.
    Import {
        dirs: Vec<PathBuf>,
        /// Height of the copy kept, in lines.
        #[arg(long, default_value_t = 360)]
        height: u32,
    },
    /// Ask Comskip, the captions and the adbreak detector where a recording's breaks are.
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
    /// Serve the labelling page.
    Label {
        /// Address to listen on. Anyone who can reach it can watch the recordings: keep it on loopback (publish it
        /// with something like `tailscale serve` if needed).
        #[arg(long, default_value = "127.0.0.1:8585")]
        bind: String,
    },
    /// Score each source against the labels.
    Eval {
        /// Show each recording, not just the totals.
        #[arg(long)]
        each: bool,
    },
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let data = Dataset::new(cli.dataset.clone(), cli.media.clone());
    let ffmpeg = || tools::find(cli.ffmpeg.as_deref(), "ffmpeg").context("no ffmpeg found (--ffmpeg or ADBREAK_FFMPEG)");
    match cli.cmd {
        Cmd::Import { dirs, height } => {
            let ff = ffmpeg()?;
            for d in dirs {
                import(&data, &ff, &d, height)?;
            }
        }
        Cmd::Suggest { ids, comskip, quick } => {
            let comskip = if quick { None } else { tools::find(comskip.as_deref(), "comskip") };
            let ff = if quick { None } else { Some(ffmpeg()?) };
            if !quick && comskip.is_none() {
                eprintln!("no Comskip found (--comskip or ADBREAK_COMSKIP); skipping it");
            }
            for name in ids {
                suggest(&data, &data.resolve(&name)?, comskip.clone(), ff.clone())?;
            }
        }
        Cmd::Label { bind } => label::serve(data, &bind).await?,
        Cmd::Eval { each } => eval(&data, each)?,
    }
    Ok(())
}

fn import(data: &Dataset, ffmpeg: &Path, dir: &Path, height: u32) -> Result<()> {
    let mut rec: Recording = dataset::read(&dir.join("recording.json"))?.with_context(|| format!("no recording.json in {}", dir.display()))?;
    let media = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.file_stem().is_some_and(|s| s == "media"))
        .with_context(|| format!("no media.* in {}", dir.display()))?;
    let id = rec.id();
    let out = data.media_dir(&id);
    std::fs::create_dir_all(&out)?;
    eprintln!("{id}: {} on {} {}, {:.0} min: making the {height}p copy", rec.title, rec.channel, rec.network, rec.duration / 60.0);
    let started = std::time::Instant::now();
    tools::transcode(ffmpeg, &media, &out.join("media.mp4"), height)?;
    rec.duration = tools::duration(ffmpeg, &out.join("media.mp4")).unwrap_or(rec.duration);
    let size = std::fs::metadata(out.join("media.mp4"))?.len();
    eprintln!("  {} MB in {:.0} s", size / 1_000_000, started.elapsed().as_secs_f64());
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
    eprintln!("  added: {}", data.root.join(&id).display());
    Ok(())
}

fn suggest(data: &Dataset, id: &str, comskip: Option<PathBuf>, ffmpeg: Option<PathBuf>) -> Result<()> {
    let e = data.entry(id)?;
    let rec = &e.recording;
    let dir = data.media_dir(id);
    eprintln!("{id}: {} on {} {}, {:.0} min", rec.title, rec.channel, rec.network, rec.duration / 60.0);
    let mut sources = vec![];
    if let Ok(vtt) = std::fs::read_to_string(dir.join("captions.vtt")) {
        let secs = adbreak_captions::seconds(&adbreak_captions::parse_vtt(&vtt), rec.duration);
        let q = adbreak_captions::Quiet::default();
        let breaks = adbreak_captions::uncaptioned(&secs, &q).into_iter().map(|(a, b)| Span::new(a, b)).collect();
        let detail = format!("uncaptioned: window {} s, coverage < {}, at least {} s", q.window, q.max_coverage, q.min_secs);
        sources.push(Source { name: "captions".into(), detail, breaks });
    }
    if comskip.is_some() || ffmpeg.is_some() {
        let media = dir.join("media.mp4");
        if !media.is_file() {
            bail!("no media for {id} in {} (import it again)", dir.display());
        }
        let work = std::env::temp_dir().join(format!("adbreak-{id}"));
        std::fs::create_dir_all(&work)?;
        let started = std::time::Instant::now();
        let (m1, w1) = (media.clone(), work.clone());
        let cs = comskip.map(|c| std::thread::spawn(move || tools::comskip(&c, &m1, &w1)));
        let det = ffmpeg.map(|f| std::thread::spawn(move || tools::detect(&f, &media)));
        if let Some(t) = cs {
            let (breaks, version) = t.join().expect("comskip thread")?;
            sources.push(Source { name: "comskip".into(), detail: version, breaks });
        }
        if let Some(t) = det {
            let breaks = t.join().expect("detector thread")?;
            sources.push(Source { name: "adbreak".into(), detail: format!("adbreak {} defaults", env!("CARGO_PKG_VERSION")), breaks });
        }
        eprintln!("  sources ran in {:.0} s", started.elapsed().as_secs_f64());
        std::fs::remove_dir_all(&work)?;
    }
    for s in &sources {
        eprintln!("  {:9} {:3} breaks, {:5.0} s", s.name, s.breaks.len(), s.breaks.iter().map(Span::len).fold(0.0, |a, b| a + b));
    }
    data.save_sources(id, rec, sources)?;
    Ok(())
}

fn eval(data: &Dataset, each: bool) -> Result<()> {
    let mut totals: std::collections::BTreeMap<String, Score> = Default::default();
    let mut labelled = 0;
    for e in data.all()? {
        let (Some(labels), Some(sugg)) = (&e.labels, &e.suggestions) else { continue };
        if labels.reviewed.is_empty() {
            continue;
        }
        labelled += 1;
        if each {
            println!("{}  ({} labelled breaks)", e.id, labels.breaks.len());
        }
        for s in &sugg.sources {
            let sc = score(labels, &s.breaks);
            if each {
                line(&format!("  {}", s.name), &sc);
            }
            totals.entry(s.name.clone()).or_default().add(&sc);
        }
    }
    if labelled == 0 {
        bail!("no labelled recordings with suggestions in {}", data.root.display());
    }
    println!("{labelled} labelled recording(s)");
    println!("  {:10} {:>9} {:>9} {:>9} {:>14} {:>9} {:>9}", "source", "precision", "recall", "F1", "breaks matched", "start err", "end err");
    for (name, sc) in &totals {
        line(&format!("  {name}"), sc);
    }
    Ok(())
}

fn line(name: &str, s: &Score) {
    let pct = |x: f64| format!("{:.0}%", 100.0 * x);
    println!(
        "{:12} {:>9} {:>9} {:>9} {:>14} {:>8.1}s {:>8.1}s",
        name,
        pct(s.precision()),
        pct(s.recall()),
        pct(s.f1()),
        format!("{}/{} ({} found)", s.matched, s.breaks, s.found),
        s.mean_start_error(),
        s.mean_end_error()
    );
}
