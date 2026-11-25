//! The dataset, in two places:
//!
//! - the database (`adbreak-db`, SQLite, not in git): each recording's metadata, its labels (a person's) and the
//!   automatic sources' suggestions; `adbreak dataset export` writes a snapshot in the dataset-directory layout to
//!   version or publish;
//! - the media store (never in git: broadcast content): one directory per recording (`Recording::id`) with its
//!   low-resolution copy `media.mp4`, captions `captions.vtt` and computed features.

use adbreak_db::Db;
use adbreak_labels::{Labels, Recording, Source, Suggestions};
use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};

pub struct Dataset {
    pub db: Db,
    pub media: PathBuf,
}

pub struct Entry {
    pub id: String,
    pub recording: Recording,
    pub labels: Option<Labels>,
    pub suggestions: Option<Suggestions>,
}

impl Dataset {
    pub fn open(db: &Path, media: PathBuf) -> Result<Self> {
        Ok(Dataset { db: Db::open(db)?, media })
    }

    pub fn ids(&self) -> Vec<String> {
        self.db.ids().unwrap_or_default()
    }

    /// The recording `name` refers to: its full id, or a unique part of one (such as the recorder's job number).
    pub fn resolve(&self, name: &str) -> Result<String> {
        let ids = self.ids();
        if ids.iter().any(|i| i == name) {
            return Ok(name.to_string());
        }
        let suffix = format!("_{name}");
        let hits: Vec<&String> = ids.iter().filter(|i| i.ends_with(&suffix)).collect();
        let hits = if hits.is_empty() { ids.iter().filter(|i| i.contains(name)).collect() } else { hits };
        match hits.as_slice() {
            [one] => Ok(one.to_string()),
            [] => bail!("no recording {name} in {} (import it first)", self.db.path.display()),
            many => bail!("{name} is ambiguous: {}", many.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")),
        }
    }

    pub fn entry(&self, id: &str) -> Result<Entry> {
        let recording = self.db.recording(id)?.with_context(|| format!("no recording {id}"))?;
        Ok(Entry { id: id.to_string(), recording, labels: self.db.labels(id)?.map(|(l, _)| l), suggestions: self.db.suggestions(id)? })
    }

    pub fn all(&self) -> Result<Vec<Entry>> {
        self.ids().iter().map(|id| self.entry(id)).collect()
    }

    /// The recording's directory in the media store.
    pub fn media_dir(&self, id: &str) -> PathBuf {
        self.media.join(id)
    }

    pub fn save_recording(&self, rec: &Recording) -> Result<String> {
        self.db.save_recording(rec)
    }

    /// Saves labels; returns their new version.
    pub fn save_labels(&self, id: &str, labels: &Labels) -> Result<String> {
        self.db.save_labels(id, labels)
    }

    /// Adds or replaces sources (by name) in the recording's suggestions.
    pub fn save_sources(&self, id: &str, _rec: &Recording, sources: Vec<Source>) -> Result<()> {
        self.db.save_sources(id, &sources)
    }
}

pub fn read<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(Some(serde_json::from_str(&t).with_context(|| format!("reading {}", path.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_ids_and_merges_sources() {
        let root = std::env::temp_dir().join(format!("adbreak-ds-{}", std::process::id()));
        let d = Dataset::open(&root.join("adbreak.db"), root.join("media")).unwrap();
        let rec =
            Recording { source: "recorder".into(), job: 32, channel: "2.1".into(), start: "2026-11-24T07:06:41Z".into(), duration: 60.0, ..Default::default() };
        let id = d.save_recording(&rec).unwrap();
        assert_eq!(id, "2026-11-24_2.1_32");
        assert_eq!(d.resolve("32").unwrap(), id);
        assert_eq!(d.resolve(&id).unwrap(), id);
        assert!(d.resolve("3").is_err() || d.resolve("3").unwrap() == id);
        assert!(d.resolve("99").is_err());
        let src = |name: &str, n: usize| Source {
            name: name.into(),
            detail: String::new(),
            breaks: vec![adbreak_labels::Span::new(0.0, 1.0).into(); n],
            trained_on: String::new(),
        };
        d.save_sources(&id, &rec, vec![src("recorder", 1), src("captions", 2)]).unwrap();
        d.save_sources(&id, &rec, vec![src("captions", 3)]).unwrap();
        let e = d.entry(&id).unwrap();
        let s = e.suggestions.unwrap();
        assert_eq!(s.sources.iter().map(|s| (s.name.as_str(), s.breaks.len())).collect::<Vec<_>>(), vec![("captions", 3), ("recorder", 1)]);
        std::fs::remove_dir_all(root).unwrap();
    }
}
