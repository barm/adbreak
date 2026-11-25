//! The labelling page: a recording's copy from the media store, the sources' suggestions on a timeline, and the
//! labels.

use crate::dataset::Dataset;
use adbreak_labels::{Labels, Suggestions};
use anyhow::Result;
use axum::{
    Json, Router,
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use serde::Serialize;
use std::sync::Arc;
use tower::ServiceExt;
use tower_http::services::ServeFile;

type Shared = Arc<Dataset>;

/// The page shown when the web app hasn't been built: how to build it.
fn unbuilt(web: &std::path::Path) -> String {
    format!(
        "<!doctype html><meta charset=utf-8><title>adbreak labelling</title><body style=\"font: 15px system-ui; margin: 2em\">\
         <h1>The labelling page isn't built</h1><p>No <code>{}</code>. Build it (needs Node), then reload:</p>\
         <pre>cd web &amp;&amp; npm ci &amp;&amp; npm run build</pre><p>Or point <code>adbreak label --web</code> at a build.</p>",
        web.join("index.html").display()
    )
}

/// Serves the labelling page (the web app's build in `web`, from `web/`: `npm run build`) and its API.
pub async fn serve(data: Dataset, bind: &str, web: &std::path::Path) -> Result<()> {
    let root = std::fs::canonicalize(&data.root).unwrap_or_else(|_| data.root.clone());
    let web = std::fs::canonicalize(web).unwrap_or_else(|_| web.to_path_buf());
    let page = if web.join("index.html").is_file() {
        tower_http::services::ServeDir::new(&web).append_index_html_on_directories(true)
    } else {
        eprintln!("no web app build in {}: run  cd web && npm ci && npm run build", web.display());
        tower_http::services::ServeDir::new(&web)
    };
    let missing = unbuilt(&web);
    let router = Router::new()
        .route(
            "/",
            get(move || {
                let missing = missing.clone();
                let web = web.clone();
                async move {
                    match tokio::fs::read_to_string(web.join("index.html")).await {
                        Ok(page) => Html(page),
                        Err(_) => Html(missing),
                    }
                }
            }),
        )
        .route("/api/recordings", get(list))
        .route("/api/recordings/{id}", get(show))
        .route("/api/recordings/{id}/labels", axum::routing::put(save))
        .route("/api/recordings/{id}/inputs", get(inputs))
        .route("/files/{id}/{file}", get(file))
        .fallback_service(page)
        .with_state(Arc::new(data));
    let listener = tokio::net::TcpListener::bind(bind).await?;
    eprintln!("labelling on http://{}  (dataset {})", listener.local_addr()?, root.display());
    axum::serve(listener, router).await?;
    Ok(())
}

struct Error(StatusCode, anyhow::Error);

impl<E: Into<anyhow::Error>> From<E> for Error {
    fn from(e: E) -> Self {
        Error(StatusCode::BAD_REQUEST, e.into())
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.0, format!("{:#}", self.1)).into_response()
    }
}

#[derive(Serialize)]
struct Row {
    id: String,
    title: String,
    channel: String,
    network: String,
    start: String,
    duration: f64,
    /// Seconds reviewed (less any excluded), seconds there are to review (the length less any excluded), and
    /// labelled breaks.
    reviewed: f64,
    usable: f64,
    breaks: usize,
    sources: Vec<String>,
    /// Whether the media store has its copy.
    media: bool,
}

async fn list(State(data): State<Shared>) -> Result<Json<Vec<Row>>, Error> {
    let mut rows = vec![];
    for e in data.all()? {
        let r = e.recording;
        rows.push(Row {
            media: data.media_dir(&e.id).join("media.mp4").is_file(),
            id: e.id,
            title: r.title,
            channel: r.channel,
            network: r.network,
            start: r.start,
            usable: e.labels.as_ref().map_or(r.duration, Labels::usable_seconds),
            duration: r.duration,
            reviewed: e.labels.as_ref().map_or(0.0, |l| l.usable_reviewed().iter().map(|s| s.len()).sum()),
            breaks: e.labels.as_ref().map_or(0, |l| l.breaks.len()),
            sources: e.suggestions.map(|s| s.sources.into_iter().map(|s| s.name).collect()).unwrap_or_default(),
        });
    }
    rows.sort_by(|a, b| b.start.cmp(&a.start));
    Ok(Json(rows))
}

#[derive(Serialize)]
struct Detail {
    labels: Labels,
    suggestions: Option<Suggestions>,
    /// Each source scored against the labels (when some of the recording has been reviewed).
    scores: std::collections::BTreeMap<String, RowScore>,
    /// The labels file's version; a save must name it (see `save`).
    version: String,
    /// The whole dataset's labels fingerprint: learned sources trained on a different one are stale.
    fingerprint: String,
}

/// FNV-1a 64 over every recording's labels.json, in id order; `adbreak_ml.data.labels_fingerprint` computes the same.
fn fingerprint(data: &Dataset) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for id in data.ids() {
        if let Ok(bytes) = std::fs::read(data.root.join(&id).join("labels.json")) {
            for b in bytes {
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    format!("{h:016x}")
}

/// A version for a labels file: a hash of its bytes ("" when there is none yet).
fn version(data: &Dataset, id: &str) -> String {
    match std::fs::read(data.root.join(id).join("labels.json")) {
        Ok(bytes) => format!("{:016x}", bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3))),
        Err(_) => String::new(),
    }
}

#[derive(Serialize)]
struct RowScore {
    precision: f64,
    recall: f64,
    f1: f64,
    matched: usize,
    breaks: usize,
    found: usize,
    double_box_seconds: f64,
    double_box_marked: f64,
    ad_read_seconds: f64,
    ad_read_marked: f64,
}

async fn show(State(data): State<Shared>, Path(id): Path<String>) -> Result<Json<Detail>, Error> {
    let id = data.resolve(&id)?;
    let e = data.entry(&id)?;
    let mut labels = e.labels.unwrap_or_else(|| Labels::new(e.recording.clone()));
    labels.recording = e.recording;
    let mut scores = std::collections::BTreeMap::new();
    if !labels.reviewed.is_empty() {
        for src in e.suggestions.iter().flat_map(|s| &s.sources) {
            let mut kinds = vec![(adbreak_labels::Kind::Ad, src.name.clone())];
            kinds.extend(src.kinds().into_iter().map(|k| (k, format!("{}/{}", src.name, serde_json::to_value(k).unwrap().as_str().unwrap_or_default()))));
            for (k, name) in kinds {
                let sc = adbreak_labels::score_kind(&labels, k, &src.spans(k));
                scores.insert(
                    name,
                    RowScore {
                        precision: sc.precision(),
                        recall: sc.recall(),
                        f1: sc.f1(),
                        matched: sc.matched,
                        breaks: sc.breaks,
                        found: sc.found,
                        double_box_seconds: sc.double_box_seconds,
                        double_box_marked: sc.double_box_marked,
                        ad_read_seconds: sc.ad_read_seconds,
                        ad_read_marked: sc.ad_read_marked,
                    },
                );
            }
        }
    }
    Ok(Json(Detail { labels, suggestions: e.suggestions, scores, version: version(&data, &id), fingerprint: fingerprint(&data) }))
}

/// Saves labels, if they are an edit of the current file: the page sends the version it loaded (`If-Match`), and a
/// save from a page that loaded an older version (another tab or browser, or before a change made outside the
/// page) is refused rather than writing over the newer labels.
async fn save(State(data): State<Shared>, Path(id): Path<String>, headers: HeaderMap, Json(mut labels): Json<Labels>) -> Result<Response, Error> {
    let id = data.resolve(&id)?;
    let current = version(&data, &id);
    let sent = headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()).unwrap_or("-");
    if sent != current {
        return Err(Error(
            StatusCode::CONFLICT,
            anyhow::anyhow!("these labels were changed since this page loaded them (another tab or browser, or an update): reload the page"),
        ));
    }
    labels.recording = data.entry(&id)?.recording;
    labels.normalise().map_err(anyhow::Error::msg)?;
    data.save_labels(&id, &labels)?;
    let mut r = Json(labels).into_response();
    r.headers_mut().insert("x-labels-version", version(&data, &id).parse()?);
    Ok(r)
}

/// The model's inputs and output for one recording, per second, for the timeline: what `adbreak features` computed
/// (captions, black, silence, edges; `features/rust-v1`) and what the last `adbreak-ml evaluate` gave
/// (`features/model-out.json`: the picture score and the final probability). Missing parts are absent.
#[derive(Serialize, Default)]
struct Tracks {
    /// 0/1 per second.
    caption_visible: Vec<u8>,
    caption_roll_up: Vec<u8>,
    caption_new: Vec<u8>,
    /// Share of each second, 0–100.
    black: Vec<u8>,
    silent: Vec<u8>,
    edges: Vec<f64>,
    /// 0–100 per second, and which labels they were trained against.
    picture: Vec<u8>,
    probability: Vec<u8>,
    /// The other kinds' outputs (`double_box`, `ad_read`), 0–100 per second.
    kinds: std::collections::BTreeMap<String, Vec<u8>>,
    trained_on: String,
}

async fn inputs(State(data): State<Shared>, Path(id): Path<String>) -> Result<Json<Tracks>, Error> {
    let id = data.resolve(&id)?;
    let dir = data.media_dir(&id).join("features");
    let mut t = Tracks::default();
    if let (Ok(meta), Ok(raw)) = (std::fs::read_to_string(dir.join("rust-v1.json")), std::fs::read(dir.join("rust-v1.f32"))) {
        let meta: serde_json::Value = serde_json::from_str(&meta)?;
        let dim = meta["dim"].as_u64().unwrap_or(0) as usize;
        let cols = dim + 6;
        let floats: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        for row in floats.chunks_exact(cols) {
            let c = &row[dim..];
            t.caption_visible.push(u8::from(c[1] > 0.5));
            t.caption_roll_up.push(u8::from(c[2] > 0.5));
            t.caption_new.push(u8::from(c[3] > 0.5));
            t.black.push((c[4].clamp(0.0, 1.0) * 100.0).round() as u8);
            t.silent.push((c[5].clamp(0.0, 1.0) * 100.0).round() as u8);
        }
        t.edges = meta["edges"].as_array().map(|a| a.iter().filter_map(|v| v.as_f64()).collect()).unwrap_or_default();
    }
    if let Ok(text) = std::fs::read_to_string(dir.join("model-out.json")) {
        let m: serde_json::Value = serde_json::from_str(&text)?;
        let pct = |v: &serde_json::Value| -> Vec<u8> {
            v.as_array().map(|a| a.iter().map(|v| (v.as_f64().unwrap_or(0.0).clamp(0.0, 1.0) * 100.0).round() as u8).collect()).unwrap_or_default()
        };
        t.picture = pct(&m["picture"]);
        t.probability = pct(&m["probability"]);
        if let Some(k) = m["kinds"].as_object() {
            t.kinds = k.iter().map(|(name, v)| (name.clone(), pct(v))).collect();
        }
        t.trained_on = m["trained_on"].as_str().unwrap_or_default().to_string();
    }
    Ok(Json(t))
}

/// The copy (its playlist and media, with range requests) and its captions.
async fn file(State(data): State<Shared>, Path((id, name)): Path<(String, String)>, req: Request) -> Result<Response, Error> {
    if !["index.m3u8", "media.mp4", "captions.vtt"].contains(&name.as_str()) {
        return Err(Error(StatusCode::NOT_FOUND, anyhow::anyhow!("no such file")));
    }
    let id = data.resolve(&id)?;
    Ok(ServeFile::new(data.media_dir(&id).join(name)).oneshot(req).await?.into_response())
}
