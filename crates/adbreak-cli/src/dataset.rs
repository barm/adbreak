//! The dataset on disk: one directory per recording (`Recording::id`), holding `labels.json` (a person's) and
//! `suggestions.json` (the automatic sources').

use adbreak_labels::{Labels, Recording, Source, Suggestions};
use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};
use std::path::{Path, PathBuf};

pub struct Dataset {
    pub root: PathBuf,
}

pub struct Entry {
    pub dir: PathBuf,
    pub labels: Option<Labels>,
    pub suggestions: Option<Suggestions>,
}

impl Dataset {
    pub fn new(root: PathBuf) -> Self {
        Dataset { root }
    }

    /// The directory for a DVR job: an existing one ending in `_<job>`, else a new name from the recording.
    pub fn dir(&self, rec: &Recording) -> PathBuf {
        let suffix = format!("_{}", rec.job);
        let existing = std::fs::read_dir(&self.root).into_iter().flatten().flatten().map(|e| e.path()).find(|p| {
            p.is_dir()
                && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(&suffix))
                && read::<Labels>(&p.join("labels.json")).ok().flatten().is_none_or(|l| l.recording.source == rec.source)
        });
        existing.unwrap_or_else(|| self.root.join(rec.id()))
    }

    pub fn entry(&self, rec: &Recording) -> Result<Entry> {
        let dir = self.dir(rec);
        Ok(Entry { labels: read(&dir.join("labels.json"))?, suggestions: read(&dir.join("suggestions.json"))?, dir })
    }

    pub fn all(&self) -> Result<Vec<Entry>> {
        let mut out = vec![];
        let Ok(rd) = std::fs::read_dir(&self.root) else { return Ok(out) };
        let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        for dir in dirs {
            out.push(Entry { labels: read(&dir.join("labels.json"))?, suggestions: read(&dir.join("suggestions.json"))?, dir });
        }
        Ok(out)
    }

    pub fn save_labels(&self, labels: &Labels) -> Result<()> {
        write(&self.dir(&labels.recording).join("labels.json"), labels)
    }

    /// Adds or replaces sources (by name) in the recording's suggestions.
    pub fn save_sources(&self, rec: &Recording, sources: Vec<Source>) -> Result<Suggestions> {
        let path = self.dir(rec).join("suggestions.json");
        let mut s = read::<Suggestions>(&path)?.unwrap_or(Suggestions { format: adbreak_labels::FORMAT, recording: rec.clone(), sources: vec![] });
        s.recording = rec.clone();
        for new in sources {
            s.sources.retain(|o| o.name != new.name);
            s.sources.push(new);
        }
        s.sources.sort_by(|a, b| a.name.cmp(&b.name));
        write(&path, &s)?;
        Ok(s)
    }
}

fn read<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
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
