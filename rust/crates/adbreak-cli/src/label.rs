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

pub async fn serve(data: Dataset, bind: &str) -> Result<()> {
    let root = std::fs::canonicalize(&data.root).unwrap_or_else(|_| data.root.clone());
    let router = Router::new()
        .route("/", get(|| async { Html(include_str!("../../../../web/label.html")) }))
        .route("/api/recordings", get(list))
        .route("/api/recordings/{id}", get(show))
        .route("/api/recordings/{id}/labels", axum::routing::put(save))
        .route("/files/{id}/{file}", get(file))
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
    /// Seconds reviewed, and labelled breaks.
    reviewed: f64,
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
            duration: r.duration,
            reviewed: e.labels.as_ref().map_or(0.0, |l| l.reviewed.iter().map(|s| s.len()).fold(0.0, |a, b| a + b)),
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
    sponsor_seconds: f64,
    sponsor_skipped: f64,
}

async fn show(State(data): State<Shared>, Path(id): Path<String>) -> Result<Json<Detail>, Error> {
    let id = data.resolve(&id)?;
    let e = data.entry(&id)?;
    let mut labels = e.labels.unwrap_or_else(|| Labels::new(e.recording.clone()));
    labels.recording = e.recording;
    let mut scores = std::collections::BTreeMap::new();
    if !labels.reviewed.is_empty() {
        for src in e.suggestions.iter().flat_map(|s| &s.sources) {
            let sc = adbreak_labels::score(&labels, &src.breaks);
            scores.insert(
                src.name.clone(),
                RowScore {
                    precision: sc.precision(),
                    recall: sc.recall(),
                    f1: sc.f1(),
                    matched: sc.matched,
                    breaks: sc.breaks,
                    found: sc.found,
                    sponsor_seconds: sc.sponsor_seconds,
                    sponsor_skipped: sc.sponsor_skipped,
                },
            );
        }
    }
    Ok(Json(Detail { labels, suggestions: e.suggestions, scores, version: version(&data, &id) }))
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

/// The copy (its playlist and media, with range requests) and its captions.
async fn file(State(data): State<Shared>, Path((id, name)): Path<(String, String)>, req: Request) -> Result<Response, Error> {
    if !["index.m3u8", "media.mp4", "captions.vtt"].contains(&name.as_str()) {
        return Err(Error(StatusCode::NOT_FOUND, anyhow::anyhow!("no such file")));
    }
    let id = data.resolve(&id)?;
    Ok(ServeFile::new(data.media_dir(&id).join(name)).oneshot(req).await?.into_response())
}
