//! The app's Model page: the installed model (`model.json`, `model-info.json` from `adbreak-ml export`), its
//! leave-one-out scores on the labelled recordings, and running the model again (`adbreak-ml` in `ml/`).

use crate::dataset::Dataset;
use adbreak_labels::{Kind, Score, score_kind};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub struct Model {
    pub data: Arc<Dataset>,
    /// The installed model's directory (what a recorder loads).
    pub dir: PathBuf,
    /// The Python project (`ml/`) that trains it, run through uv.
    pub ml: PathBuf,
    job: Mutex<Job>,
}

/// The last run of `adbreak-ml` from the page.
#[derive(Serialize, Default, Clone)]
struct Job {
    running: bool,
    action: String,
    only: Option<String>,
    started: String,
    finished: String,
    ok: Option<bool>,
    log: Vec<String>,
}

impl Model {
    pub fn new(data: Arc<Dataset>, dir: PathBuf, ml: PathBuf) -> Self {
        Model { data, dir, ml, job: Mutex::default() }
    }
}

pub fn router(model: Arc<Model>) -> Router {
    Router::new().route("/api/model", get(show)).route("/api/model/run", get(job).post(run)).with_state(model)
}

#[derive(Serialize)]
struct KindScore {
    source: String,
    kind: &'static str,
    #[serde(flatten)]
    score: Scored,
}

#[derive(Serialize, Default)]
struct Scored {
    precision: f64,
    recall: f64,
    f1: f64,
    breaks: usize,
    found: usize,
    matched: usize,
    start_error: f64,
    end_error: f64,
}

impl From<&Score> for Scored {
    fn from(s: &Score) -> Self {
        Scored {
            precision: s.precision(),
            recall: s.recall(),
            f1: s.f1(),
            breaks: s.breaks,
            found: s.found,
            matched: s.matched,
            start_error: s.mean_start_error(),
            end_error: s.mean_end_error(),
        }
    }
}

#[derive(Serialize)]
struct RecordingScore {
    id: String,
    title: String,
    /// The model's commercial breaks against the labels, and Comskip's for comparison.
    model: Option<Scored>,
    comskip: Option<Scored>,
    /// Per-second log loss and AUC of its commercial-break probability (`features/model-out.json`).
    log_loss: Option<f64>,
    auc: Option<f64>,
    /// How the model's suggestions were made (leave-one-out, or trained on all), and whether they predate the labels.
    how: String,
    stale: bool,
}

#[derive(Serialize)]
struct Overview {
    dir: String,
    /// `model-info.json` as written (absent for a model exported before it existed).
    info: Option<serde_json::Value>,
    /// Bytes on disk: model.json, and the image model's package.
    model_bytes: u64,
    image_model_bytes: u64,
    image_model: String,
    kinds: Vec<String>,
    /// `learning-curve.json` beside the model (`adbreak-ml learning-curve`), less its individual runs.
    learning_curve: Option<serde_json::Value>,
    totals: Vec<KindScore>,
    recordings: Vec<RecordingScore>,
    can_run: bool,
}

const KINDS: [(Kind, &str); 3] = [(Kind::Ad, "commercial break"), (Kind::DoubleBox, "double box"), (Kind::AdRead, "ad read")];

fn du(p: &Path) -> u64 {
    match std::fs::metadata(p) {
        Ok(m) if m.is_dir() => std::fs::read_dir(p).map(|d| d.flatten().map(|e| du(&e.path())).sum()).unwrap_or(0),
        Ok(m) => m.len(),
        Err(_) => 0,
    }
}

async fn show(State(m): State<Arc<Model>>) -> Response {
    match overview(&m) {
        Ok(o) => Json(o).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

fn overview(m: &Model) -> anyhow::Result<Overview> {
    let json: Option<serde_json::Value> = std::fs::read_to_string(m.dir.join("model.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
    let image_model = json.as_ref().and_then(|j| j["image_model"].as_str()).unwrap_or_default().to_string();
    let kinds = json.as_ref().and_then(|j| j["kinds"].as_object()).map(|k| k.keys().cloned().collect()).unwrap_or_default();
    let info = std::fs::read_to_string(m.dir.join("model-info.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
    let fingerprint = crate::label::fingerprint(&m.data);
    let mut totals: Vec<(String, Kind, &'static str, Score)> = vec![];
    let mut add = |source: &str, k: Kind, name: &'static str, s: &Score| match totals.iter_mut().find(|t| t.0 == source && t.1 == k) {
        Some(t) => t.3.add(s),
        None => totals.push((source.into(), k, name, *s)),
    };
    let mut recordings = vec![];
    for e in m.data.all()? {
        let (Some(labels), Some(sugg)) = (&e.labels, &e.suggestions) else { continue };
        if labels.reviewed.is_empty() {
            continue;
        }
        let src = |n: &str| sugg.sources.iter().find(|s| s.name == n);
        let mut row = RecordingScore {
            id: e.id.clone(),
            title: e.recording.title.clone(),
            model: None,
            comskip: None,
            log_loss: None,
            auc: None,
            how: String::new(),
            stale: false,
        };
        if let Some(s) = src("model") {
            for (k, name) in KINDS {
                let sc = score_kind(labels, k, &s.spans(k));
                if k == Kind::Ad {
                    row.model = Some((&sc).into());
                }
                add("model", k, name, &sc);
            }
            row.how = s.detail.clone();
            row.stale = !s.trained_on.is_empty() && s.trained_on != fingerprint;
        }
        if let Some(s) = src("comskip") {
            let sc = score_kind(labels, Kind::Ad, &s.spans(Kind::Ad));
            row.comskip = Some((&sc).into());
            add("comskip", Kind::Ad, "commercial break", &sc);
        }
        let out: Option<serde_json::Value> =
            std::fs::read_to_string(m.data.media_dir(&e.id).join("features/model-out.json")).ok().and_then(|t| serde_json::from_str(&t).ok());
        if let Some(o) = out {
            row.log_loss = o["metrics"]["log_loss"].as_f64();
            row.auc = o["metrics"]["auc"].as_f64();
        }
        recordings.push(row);
    }
    let image = m.dir.join(format!("{image_model}-image.mlpackage"));
    Ok(Overview {
        dir: m.dir.display().to_string(),
        info,
        model_bytes: du(&m.dir.join("model.json")),
        image_model_bytes: du(&image),
        image_model,
        kinds,
        learning_curve: std::fs::read_to_string(m.dir.join("learning-curve.json")).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()).map(
            |mut v| {
                if let Some(o) = v.as_object_mut() {
                    o.remove("runs");
                }
                v
            },
        ),
        totals: totals.into_iter().map(|(source, _, kind, s)| KindScore { source, kind, score: (&s).into() }).collect(),
        recordings,
        can_run: m.ml.join("pyproject.toml").is_file(),
    })
}

async fn job(State(m): State<Arc<Model>>) -> Json<Job> {
    Json(m.job.lock().unwrap().clone())
}

#[derive(Deserialize)]
struct Run {
    /// "evaluate" (suggestions again, leave-one-out), "retrain" (evaluate, then export and install the model), or
    /// "learning-curve" (`adbreak-ml learning-curve` into the model's directory).
    action: String,
    /// Just this recording (evaluate only).
    only: Option<String>,
}

async fn run(State(m): State<Arc<Model>>, Json(r): Json<Run>) -> Response {
    if !matches!(r.action.as_str(), "evaluate" | "retrain" | "learning-curve") {
        return (StatusCode::BAD_REQUEST, "action is evaluate, retrain or learning-curve").into_response();
    }
    if !m.ml.join("pyproject.toml").is_file() {
        return (StatusCode::CONFLICT, format!("no ML project at {} (adbreak label --ml)", m.ml.display())).into_response();
    }
    {
        let mut j = m.job.lock().unwrap();
        if j.running {
            return (StatusCode::CONFLICT, format!("already running: {}", j.action)).into_response();
        }
        *j = Job { running: true, action: r.action.clone(), only: r.only.clone(), started: now(), ..Default::default() };
    }
    let mut steps: Vec<Vec<String>> = vec![];
    if r.action == "learning-curve" {
        steps.push(vec!["run".into(), "-q".into(), "adbreak-ml".into(), "learning-curve".into(), m.dir.display().to_string()]);
    }
    let mut evaluate = vec!["run".to_string(), "-q".into(), "adbreak-ml".into(), "evaluate".into()];
    if let Some(id) = &r.only {
        evaluate.extend(["--only".into(), id.clone()]);
    }
    if r.action != "learning-curve" {
        steps.push(evaluate);
    }
    if r.action == "retrain" {
        steps.push(vec!["run".into(), "-q".into(), "adbreak-ml".into(), "export".into(), m.dir.display().to_string()]);
    }
    let model = m.clone();
    tokio::spawn(async move {
        let mut ok = true;
        for args in steps {
            model.job.lock().unwrap().log.push(format!("$ uv {}", args.join(" ")));
            match exec(&model, &args).await {
                Ok(true) => {}
                Ok(false) => {
                    ok = false;
                    break;
                }
                Err(e) => {
                    model.job.lock().unwrap().log.push(format!("{e:#}"));
                    ok = false;
                    break;
                }
            }
        }
        let mut j = model.job.lock().unwrap();
        j.running = false;
        j.ok = Some(ok);
        j.finished = now();
    });
    (StatusCode::ACCEPTED, Json(m.job.lock().unwrap().clone())).into_response()
}

/// Runs `uv <args>` in the ML project, keeping the last lines of its output in the job's log.
async fn exec(m: &Model, args: &[String]) -> anyhow::Result<bool> {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let uv = std::env::var("ADBREAK_UV").unwrap_or_else(|_| "uv".into());
    let mut child = tokio::process::Command::new(uv)
        .args(args)
        .current_dir(&m.ml)
        // The same dataset and media store as this server.
        .env("ADBREAK_DB", &m.data.db.path)
        .env("ADBREAK_MEDIA", &m.data.media)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;
    let (out, err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let push = |line: String| {
        let mut j = m.job.lock().unwrap();
        j.log.push(line);
        let n = j.log.len();
        if n > 200 {
            j.log.drain(..n - 200);
        }
    };
    let mut a = BufReader::new(out).lines();
    let mut b = BufReader::new(err).lines();
    let (mut a_done, mut b_done) = (false, false);
    while !(a_done && b_done) {
        tokio::select! {
            l = a.next_line(), if !a_done => match l? { Some(l) => push(l), None => a_done = true },
            l = b.next_line(), if !b_done => match l? { Some(l) if !l.contains("Warning") => push(l), Some(_) => {}, None => b_done = true },
        }
    }
    Ok(child.wait().await?.success())
}

fn now() -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    format!("{t}")
}
