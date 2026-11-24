//! The labelling page: a recording from the DVR, the sources' suggestions on a timeline, and the labels.
//!
//! The page plays the recording through this server, which forwards to the DVR with the agent token, so the
//! browser needs no DVR login and the token never reaches it.

use crate::{dataset::Dataset, dvr::Dvr};
use adbreak_labels::{Labels, Suggestions};
use anyhow::Result;
use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use serde::Serialize;
use std::sync::Arc;

struct App {
    dvr: Dvr,
    data: Dataset,
}

type Shared = Arc<App>;

pub async fn serve(dvr: Dvr, data: Dataset, bind: &str) -> Result<()> {
    let root = std::fs::canonicalize(&data.root).unwrap_or_else(|_| data.root.clone());
    let app = Arc::new(App { dvr, data });
    let router = Router::new()
        .route("/", get(|| async { Html(include_str!("label.html")) }))
        .route("/api/recordings", get(list))
        .route("/api/recordings/{id}", get(show))
        .route("/api/recordings/{id}/labels", axum::routing::put(save))
        .route("/dvr/hls/{id}/{file}", get(proxy))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    eprintln!("labelling on http://{}  (dataset {})", listener.local_addr()?, root.display());
    axum::serve(listener, router).with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok().unwrap_or_default() }).await?;
    Ok(())
}

struct Error(anyhow::Error);

impl<E: Into<anyhow::Error>> From<E> for Error {
    fn from(e: E) -> Self {
        Error(e.into())
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, format!("{:#}", self.0)).into_response()
    }
}

#[derive(Serialize)]
struct Row {
    job: i64,
    title: String,
    channel: String,
    start: String,
    duration: f64,
    /// Seconds reviewed, and labelled breaks.
    reviewed: f64,
    breaks: usize,
    sources: Vec<String>,
}

async fn list(State(app): State<Shared>) -> Result<Json<Vec<Row>>, Error> {
    let mut rows = vec![];
    for j in app.dvr.jobs().await?.into_iter().filter(|j| j.usable()) {
        let rec = adbreak_labels::Recording { source: "dvr".into(), job: j.id, ..Default::default() };
        let e = app.data.entry(&rec)?;
        rows.push(Row {
            job: j.id,
            title: j.title,
            channel: j.channel,
            start: j.window_start,
            duration: j.media_secs.unwrap_or(0.0),
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
}

async fn show(State(app): State<Shared>, Path(id): Path<i64>) -> Result<Json<Detail>, Error> {
    let job = app.dvr.job(id).await?;
    let rec = app.dvr.recording(&job).await;
    let e = app.data.entry(&rec)?;
    let mut labels = e.labels.unwrap_or_else(|| Labels::new(rec.clone()));
    labels.recording = rec;
    Ok(Json(Detail { labels, suggestions: e.suggestions }))
}

async fn save(State(app): State<Shared>, Path(id): Path<i64>, Json(mut labels): Json<Labels>) -> Result<Json<Labels>, Error> {
    if labels.recording.job != id {
        return Err(anyhow::anyhow!("labels are for job {}, not {id}", labels.recording.job).into());
    }
    labels.normalise().map_err(anyhow::Error::msg)?;
    app.data.save_labels(&labels)?;
    Ok(Json(labels))
}

/// Forwards playlist, media and caption requests to the DVR, ranges included.
async fn proxy(State(app): State<Shared>, Path((id, file)): Path<(i64, String)>, headers: HeaderMap) -> Result<Response, Error> {
    let mut req = app.dvr.get(&format!("/hls/{id}/{file}"));
    if let Some(r) = headers.get(header::RANGE) {
        req = req.header(header::RANGE, r);
    }
    let r = req.send().await?;
    let mut out = Response::builder().status(r.status().as_u16());
    for h in [header::CONTENT_TYPE, header::CONTENT_LENGTH, header::CONTENT_RANGE, header::ACCEPT_RANGES, header::CONTENT_ENCODING] {
        if let Some(v) = r.headers().get(h.as_str()) {
            out = out.header(h.as_str(), v.as_bytes());
        }
    }
    Ok(out.header(header::CACHE_CONTROL, "no-store").body(Body::from_stream(r.bytes_stream()))?)
}
