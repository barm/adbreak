//! The parts of barm/dvr's HTTP API this tool uses: recordings, their captions and media.

use adbreak_labels::{Recording, Span};
use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// A DVR job, as much of it as is needed here.
#[derive(Clone, Debug, Deserialize)]
pub struct Job {
    pub id: i64,
    pub channel: String,
    pub title: String,
    /// The recording's actual start: when the tuner stream began.
    pub window_start: String,
    pub status: String,
    #[serde(default)]
    pub media_secs: Option<f64>,
    /// The DVR's own break scan.
    #[serde(default)]
    pub breaks: Vec<Span>,
    #[serde(default)]
    pub breaks_done: bool,
    /// Stretches "Remove commercials" took out. The media no longer matches the broadcast if there are any.
    #[serde(default)]
    pub cuts: Vec<serde_json::Value>,
}

impl Job {
    /// Finished, with media to look at and nothing cut out of it.
    pub fn usable(&self) -> bool {
        self.status == "completed" && self.media_secs.is_some_and(|s| s > 0.0) && self.cuts.is_empty()
    }
}

#[derive(Deserialize)]
struct Channel {
    number: String,
    name: String,
}

#[derive(Clone)]
pub struct Dvr {
    pub base: String,
    token: String,
    http: reqwest::Client,
}

impl Dvr {
    pub fn new(base: &str, token_file: &Path) -> Result<Self> {
        let token = std::fs::read_to_string(token_file)
            .with_context(|| format!("reading the DVR's agent token from {} (generate one on the DVR's Connect-an-agent page)", token_file.display()))?
            .trim()
            .to_string();
        Ok(Dvr { base: base.trim_end_matches('/').to_string(), token, http: reqwest::Client::new() })
    }

    pub fn get(&self, path: &str) -> reqwest::RequestBuilder {
        self.http.get(format!("{}{path}", self.base)).bearer_auth(&self.token)
    }

    async fn json<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let r = self.get(path).send().await.with_context(|| format!("reaching the DVR at {}", self.base))?;
        if !r.status().is_success() {
            bail!("DVR {path}: {}", r.status());
        }
        Ok(r.json().await?)
    }

    pub async fn jobs(&self) -> Result<Vec<Job>> {
        self.json("/recordings").await
    }

    pub async fn job(&self, id: i64) -> Result<Job> {
        self.json(&format!("/recordings/{id}")).await
    }

    /// The station's name for a channel (e.g. `KTVU-HD` for 2.1), if the tuner lists it.
    pub async fn network(&self, channel: &str) -> String {
        let chans: Vec<Channel> = self.json("/channels").await.unwrap_or_default();
        chans.into_iter().find(|c| c.number == channel).map(|c| c.name).unwrap_or_default()
    }

    pub async fn recording(&self, job: &Job) -> Recording {
        Recording {
            source: "dvr".into(),
            job: job.id,
            channel: job.channel.clone(),
            network: self.network(&job.channel).await,
            title: job.title.clone(),
            start: job.window_start.clone(),
            duration: job.media_secs.unwrap_or(0.0),
        }
    }

    /// The recording's captions (WebVTT), or empty if it has none.
    pub async fn captions(&self, id: i64) -> Result<String> {
        let r = self.get(&format!("/hls/{id}/captions.vtt")).send().await?;
        Ok(if r.status().is_success() { r.text().await? } else { String::new() })
    }

    /// Downloads the recording's media (one fragmented MP4) to `dir/media.mp4`, reporting progress.
    pub async fn download(&self, id: i64, dir: &Path) -> Result<PathBuf> {
        let path = dir.join("media.mp4");
        let r = self.get(&format!("/media/{id}")).send().await?;
        if !r.status().is_success() {
            bail!("DVR /media/{id}: {}", r.status());
        }
        let total = r.content_length().unwrap_or(0);
        let mut f = tokio::fs::File::create(&path).await?;
        let (mut got, mut shown, started) = (0u64, 0u64, std::time::Instant::now());
        let mut body = r.bytes_stream();
        while let Some(chunk) = body.next().await {
            let chunk = chunk.context("downloading the media (the DVR's storage may be busy; try again)")?;
            f.write_all(&chunk).await?;
            got += chunk.len() as u64;
            if got - shown >= 256 << 20 || got == total {
                shown = got;
                let mb = |b: u64| b / 1_000_000;
                eprintln!("  media {} / {} MB, {:.1} MB/s", mb(got), mb(total), got as f64 / 1e6 / started.elapsed().as_secs_f64().max(0.001));
            }
        }
        f.flush().await?;
        Ok(path)
    }
}
