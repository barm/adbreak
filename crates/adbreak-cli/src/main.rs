//! `adbreak`: label commercial breaks in DVR recordings and score detectors against the labels.
//!
//! - `adbreak suggest <job>` asks the automatic sources (Comskip, the captions, the adbreak detector, the DVR's
//!   own scan) where the breaks are and keeps their answers in the dataset.
//! - `adbreak label` serves a page to review a recording against those suggestions and save labels.
//! - `adbreak eval` scores every source against the labels.

mod dataset;
mod dvr;
mod label;
mod tools;

use adbreak_labels::{Score, Source, Span, score};
use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use dataset::Dataset;
use dvr::Dvr;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Label commercial breaks in DVR recordings and score detectors against the labels")]
struct Cli {
    /// The DVR (barm/dvr) to read recordings from.
    #[arg(long, env = "ADBREAK_DVR", default_value = "http://127.0.0.1:8484", global = true)]
    dvr: String,
    /// File holding the DVR's agent token.
    #[arg(long, env = "ADBREAK_DVR_TOKEN_FILE", default_value_os_t = home().join("Movies/hdhr-dvr/.api_token"), global = true)]
    token_file: PathBuf,
    /// The dataset directory (labels and suggestions, one directory per recording).
    #[arg(long, env = "ADBREAK_DATASET", default_value = "dataset", global = true)]
    dataset: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Ask Comskip, the captions, the adbreak detector and the DVR where a recording's breaks are.
    Suggest {
        /// DVR job ids.
        #[arg(required = true)]
        jobs: Vec<i64>,
        /// Comskip binary (default: `comskip` on PATH; skipped if there is none).
        #[arg(long, env = "ADBREAK_COMSKIP")]
        comskip: Option<PathBuf>,
        /// ffmpeg binary, to decode for the detector (default: `ffmpeg` on PATH or Homebrew's).
        #[arg(long, env = "ADBREAK_FFMPEG")]
        ffmpeg: Option<PathBuf>,
        /// Only the sources that don't need the media (captions, the DVR's scan): no download.
        #[arg(long)]
        quick: bool,
        /// Where the media is downloaded to while the sources run.
        #[arg(long, default_value_os_t = home().join("Library/Caches/adbreak"))]
        cache: PathBuf,
        /// Keep the downloaded media afterwards.
        #[arg(long)]
        keep: bool,
    },
    /// Serve the labelling page.
    Label {
        /// Address to listen on. Anyone who can reach it can watch the DVR's recordings through it: keep it on
        /// loopback or a private network.
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
    let data = Dataset::new(cli.dataset.clone());
    match cli.cmd {
        Cmd::Suggest { jobs, comskip, ffmpeg, quick, cache, keep } => {
            let dvr = Dvr::new(&cli.dvr, &cli.token_file)?;
            let comskip = if quick { None } else { tools::find(comskip.as_deref(), "comskip") };
            let ffmpeg = if quick { None } else { tools::find(ffmpeg.as_deref(), "ffmpeg") };
            if !quick && comskip.is_none() {
                eprintln!("no Comskip found (--comskip or ADBREAK_COMSKIP); skipping it");
            }
            if !quick && ffmpeg.is_none() {
                eprintln!("no ffmpeg found (--ffmpeg or ADBREAK_FFMPEG); skipping the detector");
            }
            for id in jobs {
                suggest(&dvr, &data, id, comskip.clone(), ffmpeg.clone(), &cache, keep).await?;
            }
        }
        Cmd::Label { bind } => {
            let dvr = Dvr::new(&cli.dvr, &cli.token_file)?;
            label::serve(dvr, data, &bind).await?;
        }
        Cmd::Eval { each } => eval(&data, each)?,
    }
    Ok(())
}

async fn suggest(dvr: &Dvr, data: &Dataset, id: i64, comskip: Option<PathBuf>, ffmpeg: Option<PathBuf>, cache: &std::path::Path, keep: bool) -> Result<()> {
    let job = dvr.job(id).await?;
    if !job.usable() {
        bail!("job {id} can't be labelled: it needs to be completed (it is {}) with no commercials cut out", job.status);
    }
    let rec = dvr.recording(&job).await;
    eprintln!("job {id}: {} on {} {}, {:.0} min", rec.title, rec.channel, rec.network, rec.duration / 60.0);
    let mut sources = vec![];
    if job.breaks_done {
        sources.push(Source { name: "dvr".into(), detail: "the DVR's own scan".into(), breaks: job.breaks.clone() });
    }
    let vtt = dvr.captions(id).await?;
    if !vtt.is_empty() {
        let secs = adbreak_captions::seconds(&adbreak_captions::parse_vtt(&vtt), rec.duration);
        let q = adbreak_captions::Quiet::default();
        let breaks = adbreak_captions::uncaptioned(&secs, &q).into_iter().map(|(a, b)| Span::new(a, b)).collect();
        let detail = format!("uncaptioned: window {} s, coverage < {}, at least {} s", q.window, q.max_coverage, q.min_secs);
        sources.push(Source { name: "captions".into(), detail, breaks });
    }
    if comskip.is_some() || ffmpeg.is_some() {
        let work = cache.join(id.to_string());
        std::fs::create_dir_all(&work)?;
        let media = match work.join("media.mp4") {
            p if keep && p.exists() => p,
            _ => dvr.download(id, &work).await?,
        };
        let started = std::time::Instant::now();
        let (m1, w1) = (media.clone(), work.clone());
        let cs = comskip.map(|c| std::thread::spawn(move || tools::comskip(&c, &m1, &w1)));
        let (m2, f2) = (media.clone(), ffmpeg.clone());
        let det = f2.map(|f| std::thread::spawn(move || tools::detect(&f, &m2)));
        if let Some(t) = cs {
            let (breaks, version) = t.join().expect("comskip thread")?;
            sources.push(Source { name: "comskip".into(), detail: version, breaks });
        }
        if let Some(t) = det {
            let breaks = t.join().expect("detector thread")?;
            sources.push(Source { name: "adbreak".into(), detail: format!("adbreak {} defaults", env!("CARGO_PKG_VERSION")), breaks });
        }
        eprintln!("  sources ran in {:.0} s", started.elapsed().as_secs_f64());
        if !keep {
            std::fs::remove_dir_all(&work)?;
        }
    }
    for s in &sources {
        eprintln!("  {:9} {:3} breaks, {:5.0} s", s.name, s.breaks.len(), s.breaks.iter().map(Span::len).fold(0.0, |a, b| a + b));
    }
    data.save_sources(&rec, sources)?;
    eprintln!("  saved to {}", data.dir(&rec).join("suggestions.json").display());
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
            println!("{}  ({} labelled breaks)", e.dir.file_name().unwrap_or_default().to_string_lossy(), labels.breaks.len());
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
