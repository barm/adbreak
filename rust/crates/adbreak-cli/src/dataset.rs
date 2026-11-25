//! The dataset on disk, in two places:
//!
//! - the dataset directory (in git): one directory per recording (`Recording::id`) with `recording.json`,
//!   `labels.json` (a person's) and `suggestions.json` (the automatic sources');
//! - the media store (never in git: broadcast content): the same directory names, each with the recording's
//!   low-resolution copy `media.mp4` and its captions `captions.vtt`.

use adbreak_labels::{Labels, Recording, Source, Suggestions};
use anyhow::{Context, Result, bail};
use serde::{Serialize, de::DeserializeOwned};
use std::path::{Path, PathBuf};

pub struct Dataset {
    pub root: PathBuf,
    pub media: PathBuf,
}

pub struct Entry {
    pub id: String,
    pub recording: Recording,
    pub labels: Option<Labels>,
    pub suggestions: Option<Suggestions>,
}

impl Dataset {
    pub fn new(root: PathBuf, media: PathBuf) -> Self {
        Dataset { root, media }
    }

    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = std::fs::read_dir(&self.root)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().join("recording.json").is_file())
            .filter_map(|e| e.file_name().to_str().map(String::from))
            .collect();
        ids.sort();
        ids
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
            [] => bail!("no recording {name} in {} (import it first)", self.root.display()),
            many => bail!("{name} is ambiguous: {}", many.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")),
        }
    }

    pub fn entry(&self, id: &str) -> Result<Entry> {
        let dir = self.root.join(id);
        let recording = read(&dir.join("recording.json"))?.with_context(|| format!("no recording.json in {}", dir.display()))?;
        Ok(Entry { id: id.to_string(), recording, labels: read(&dir.join("labels.json"))?, suggestions: read(&dir.join("suggestions.json"))? })
    }

    pub fn all(&self) -> Result<Vec<Entry>> {
        self.ids().iter().map(|id| self.entry(id)).collect()
    }

    /// The recording's directory in the media store.
    pub fn media_dir(&self, id: &str) -> PathBuf {
        self.media.join(id)
    }

    pub fn save_recording(&self, rec: &Recording) -> Result<String> {
        let id = rec.id();
        write(&self.root.join(&id).join("recording.json"), rec)?;
        Ok(id)
    }

    pub fn save_labels(&self, id: &str, labels: &Labels) -> Result<()> {
        write(&self.root.join(id).join("labels.json"), labels)
    }

    /// Adds or replaces sources (by name) in the recording's suggestions.
    pub fn save_sources(&self, id: &str, rec: &Recording, sources: Vec<Source>) -> Result<()> {
        let path = self.root.join(id).join("suggestions.json");
        let mut s = read::<Suggestions>(&path)?.unwrap_or(Suggestions { format: adbreak_labels::FORMAT, recording: rec.clone(), sources: vec![] });
        s.recording = rec.clone();
        for new in sources {
            s.sources.retain(|o| o.name != new.name);
            s.sources.push(new);
        }
        s.sources.sort_by(|a, b| a.name.cmp(&b.name));
        write(&path, &s)
    }
}

pub fn read<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(Some(serde_json::from_str(&t).with_context(|| format!("reading {}", path.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// Pretty JSON, written to a temporary file and renamed so a crash never leaves half a file.
fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let dir = path.parent().context("no directory")?;
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(value)? + "\n")?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_ids_and_merges_sources() {
        let root = std::env::temp_dir().join(format!("adbreak-ds-{}", std::process::id()));
        let d = Dataset::new(root.join("dataset"), root.join("media"));
        let rec =
            Recording { source: "recorder".into(), job: 32, channel: "2.1".into(), start: "2026-11-24T07:06:41Z".into(), duration: 60.0, ..Default::default() };
        let id = d.save_recording(&rec).unwrap();
        assert_eq!(id, "2026-11-24_2.1_32");
        assert_eq!(d.resolve("32").unwrap(), id);
        assert_eq!(d.resolve(&id).unwrap(), id);
        assert!(d.resolve("3").is_err() || d.resolve("3").unwrap() == id);
        assert!(d.resolve("99").is_err());
        let src = |name: &str, n: usize| Source { name: name.into(), detail: String::new(), breaks: vec![adbreak_labels::Span::new(0.0, 1.0); n] };
        d.save_sources(&id, &rec, vec![src("recorder", 1), src("captions", 2)]).unwrap();
        d.save_sources(&id, &rec, vec![src("captions", 3)]).unwrap();
        let e = d.entry(&id).unwrap();
        let s = e.suggestions.unwrap();
        assert_eq!(s.sources.iter().map(|s| (s.name.as_str(), s.breaks.len())).collect::<Vec<_>>(), vec![("captions", 3), ("recorder", 1)]);
        std::fs::remove_dir_all(root).unwrap();
    }
}
