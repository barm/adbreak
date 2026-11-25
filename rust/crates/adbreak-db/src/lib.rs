//! The adbreak dataset in one SQLite database: each recording's metadata, its labels (a person's; every save is
//! kept), and the automatic sources' suggestions. The labelling app writes labels here as you go, and training reads
//! and writes the same database, so nothing needs committing day to day; `export_dir` writes a snapshot in the
//! dataset-directory layout (one directory per recording: `recording.json`, `labels.json`, `suggestions.json`) to
//! version or publish, and `import_dir` reads one.
//!
//! Labels and sources are kept as JSON (`adbreak_labels` types). A recording's labels **version** is the FNV-1a 64
//! hash of its stored labels JSON, and the dataset's **fingerprint** the same hash over every recording's labels JSON
//! in id order; `ml/adbreak_ml/data.py` computes both the same way.

#![forbid(unsafe_code)]

use adbreak_labels::{FORMAT, Labels, Recording, Source, Suggestions};
use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS recordings (
    id        TEXT PRIMARY KEY,
    recording TEXT NOT NULL,             -- adbreak_labels::Recording, JSON
    added_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE TABLE IF NOT EXISTS labels (
    id         TEXT PRIMARY KEY REFERENCES recordings(id) ON DELETE CASCADE,
    labels     TEXT NOT NULL,            -- adbreak_labels::Labels, JSON
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE TABLE IF NOT EXISTS label_history (
    id       TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    labels   TEXT NOT NULL,
    saved_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS label_history_id ON label_history(id, saved_at);
CREATE TABLE IF NOT EXISTS sources (
    id         TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    source     TEXT NOT NULL,            -- adbreak_labels::Source, JSON
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (id, name)
);
";

/// The database. Safe to share between threads; other processes (training) may use it at the same time.
pub struct Db {
    conn: Mutex<Connection>,
    pub path: PathBuf,
}

/// FNV-1a 64 of some bytes, in hex.
pub fn fnv(bytes: &[u8]) -> String {
    format!("{:016x}", bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)))
}

/// Where the database lives by default: `~/Library/Application Support/adbreak/adbreak.db`.
pub fn default_path() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    home.join("Library/Application Support/adbreak/adbreak.db")
}

impl Db {
    /// Opens (creating if needed) the database at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        // WAL: the app and training read and write at the same time.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(10))?;
        conn.execute_batch(SCHEMA)?;
        Ok(Db { conn: Mutex::new(conn), path: path.to_path_buf() })
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Every recording's id, sorted.
    pub fn ids(&self) -> Result<Vec<String>> {
        let c = self.conn();
        let mut q = c.prepare("SELECT id FROM recordings ORDER BY id")?;
        let ids = q.query_map([], |r| r.get(0))?.collect::<rusqlite::Result<Vec<String>>>()?;
        Ok(ids)
    }

    pub fn recording(&self, id: &str) -> Result<Option<Recording>> {
        let t: Option<String> = self.conn().query_row("SELECT recording FROM recordings WHERE id = ?1", [id], |r| r.get(0)).optional()?;
        t.map(|t| serde_json::from_str(&t).with_context(|| format!("recording {id}"))).transpose()
    }

    /// Adds or updates a recording; returns its id.
    pub fn save_recording(&self, rec: &Recording) -> Result<String> {
        let id = rec.id();
        self.conn().execute(
            "INSERT INTO recordings (id, recording) VALUES (?1, ?2) ON CONFLICT(id) DO UPDATE SET recording = excluded.recording",
            params![id, serde_json::to_string(rec)?],
        )?;
        Ok(id)
    }

    /// The recording's labels and their version, if any have been saved.
    pub fn labels(&self, id: &str) -> Result<Option<(Labels, String)>> {
        let t: Option<String> = self.conn().query_row("SELECT labels FROM labels WHERE id = ?1", [id], |r| r.get(0)).optional()?;
        t.map(|t| Ok((serde_json::from_str(&t).with_context(|| format!("labels for {id}"))?, fnv(t.as_bytes())))).transpose()
    }

    /// The version of the recording's labels ("" when there are none yet).
    pub fn labels_version(&self, id: &str) -> Result<String> {
        let t: Option<String> = self.conn().query_row("SELECT labels FROM labels WHERE id = ?1", [id], |r| r.get(0)).optional()?;
        Ok(t.map(|t| fnv(t.as_bytes())).unwrap_or_default())
    }

    /// Saves the recording's labels (keeping the save in its history); returns their new version.
    pub fn save_labels(&self, id: &str, labels: &Labels) -> Result<String> {
        let t = serde_json::to_string(labels)?;
        let mut c = self.conn();
        let tx = c.transaction()?;
        tx.execute(
            "INSERT INTO labels (id, labels) VALUES (?1, ?2)
             ON CONFLICT(id) DO UPDATE SET labels = excluded.labels, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')",
            params![id, t],
        )?;
        tx.execute("INSERT INTO label_history (id, labels) VALUES (?1, ?2)", params![id, t])?;
        tx.commit()?;
        Ok(fnv(t.as_bytes()))
    }

    /// The recording's sources, by name.
    pub fn sources(&self, id: &str) -> Result<Vec<Source>> {
        let c = self.conn();
        let mut q = c.prepare("SELECT source FROM sources WHERE id = ?1 ORDER BY name")?;
        let rows = q.query_map([id], |r| r.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
        rows.iter().map(|t| serde_json::from_str(t).with_context(|| format!("a source for {id}"))).collect()
    }

    /// The recording's suggestions (its sources), if it has any.
    pub fn suggestions(&self, id: &str) -> Result<Option<Suggestions>> {
        let sources = self.sources(id)?;
        if sources.is_empty() {
            return Ok(None);
        }
        let recording = self.recording(id)?.with_context(|| format!("no recording {id}"))?;
        Ok(Some(Suggestions { format: FORMAT, recording, sources }))
    }

    /// Adds or replaces sources (by name) for the recording.
    pub fn save_sources(&self, id: &str, sources: &[Source]) -> Result<()> {
        let mut c = self.conn();
        let tx = c.transaction()?;
        for s in sources {
            tx.execute(
                "INSERT INTO sources (id, name, source) VALUES (?1, ?2, ?3)
                 ON CONFLICT(id, name) DO UPDATE SET source = excluded.source, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')",
                params![id, s.name, serde_json::to_string(s)?],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Which labels a learned source was trained against: FNV-1a 64 over every recording's labels JSON in id order.
    pub fn fingerprint(&self) -> Result<String> {
        let c = self.conn();
        let mut q = c.prepare("SELECT labels FROM labels ORDER BY id")?;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for t in q.query_map([], |r| r.get::<_, String>(0))? {
            for b in t?.bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        Ok(format!("{h:016x}"))
    }

    /// Reads a dataset directory (one directory per recording with `recording.json`, and `labels.json` and
    /// `suggestions.json` if present) into the database. Labels already in the database are kept unless `replace`.
    /// Returns how many recordings it read.
    pub fn import_dir(&self, dir: &Path, replace: bool) -> Result<usize> {
        let mut n = 0;
        let mut entries: Vec<_> = std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let d = e.path();
            let Some(rec) = read::<Recording>(&d.join("recording.json"))? else { continue };
            let id = self.save_recording(&rec)?;
            if let Some(l) = read::<Labels>(&d.join("labels.json"))?
                && (replace || self.labels(&id)?.is_none())
            {
                self.save_labels(&id, &l)?;
            }
            if let Some(s) = read::<Suggestions>(&d.join("suggestions.json"))? {
                self.save_sources(&id, &s.sources)?;
            }
            n += 1;
        }
        Ok(n)
    }

    /// Writes the whole dataset to `dir` in the dataset-directory layout (pretty JSON, as in git). Returns how many
    /// recordings it wrote.
    pub fn export_dir(&self, dir: &Path) -> Result<usize> {
        let ids = self.ids()?;
        for id in &ids {
            let d = dir.join(id);
            std::fs::create_dir_all(&d)?;
            let rec = self.recording(id)?.with_context(|| format!("no recording {id}"))?;
            write(&d.join("recording.json"), &rec)?;
            match self.labels(id)? {
                Some((l, _)) => write(&d.join("labels.json"), &l)?,
                None => remove(&d.join("labels.json"))?,
            }
            match self.suggestions(id)? {
                Some(s) => write(&d.join("suggestions.json"), &s)?,
                None => remove(&d.join("suggestions.json"))?,
            }
        }
        Ok(ids.len())
    }

    /// Writes a consistent copy of the whole database, label history included, to
    /// `dir/adbreak-<UTC time>.db` (`VACUUM INTO`: safe while the app and training are writing), then deletes all but
    /// the newest `keep` copies there. Returns the new copy's path.
    pub fn backup(&self, dir: &Path, keep: usize) -> Result<PathBuf> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let c = self.conn();
        let stamp: String = c.query_row("SELECT strftime('%Y-%m-%dT%H%M%SZ', 'now')", [], |r| r.get(0))?;
        let path = dir.join(format!("adbreak-{stamp}.db"));
        let tmp = dir.join(format!(".adbreak-{stamp}.db.tmp"));
        remove(&tmp)?;
        c.execute("VACUUM INTO ?1", [tmp.to_str().context("backup path is not UTF-8")?])?;
        drop(c);
        std::fs::rename(&tmp, &path)?;
        let mut copies: Vec<PathBuf> = std::fs::read_dir(dir)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("adbreak-") && n.ends_with(".db")))
            .collect();
        copies.sort();
        for old in copies.iter().rev().skip(keep.max(1)) {
            std::fs::remove_file(old)?;
        }
        Ok(path)
    }
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(Some(serde_json::from_str(&t).with_context(|| format!("reading {}", path.display()))?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

fn write<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(value)? + "\n")?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn remove(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adbreak_labels::{Break, Kind, Span};

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("adbreak-db-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn backups_are_complete_and_pruned() {
        let dir = temp("backup");
        let db = Db::open(&dir.join("a.db")).unwrap();
        let rec =
            Recording { source: "recorder".into(), job: 1, channel: "2.1".into(), start: "2026-11-24T07:06:41Z".into(), duration: 60.0, ..Default::default() };
        let id = db.save_recording(&rec).unwrap();
        db.save_labels(&id, &Labels::new(rec)).unwrap();
        let out = dir.join("backups");
        std::fs::create_dir_all(&out).unwrap();
        for i in 0..3 {
            std::fs::write(out.join(format!("adbreak-2020-01-0{i}T000000Z.db")), "old").unwrap();
        }
        let path = db.backup(&out, 2).unwrap();
        let copy = Db::open(&path).unwrap();
        assert_eq!(copy.ids().unwrap(), vec![id.clone()]);
        assert_eq!(copy.fingerprint().unwrap(), db.fingerprint().unwrap());
        drop(copy);
        let mut left: Vec<_> = std::fs::read_dir(&out).unwrap().flatten().map(|e| e.file_name().into_string().unwrap()).collect();
        left.sort();
        assert_eq!(left.len(), 2, "{left:?}");
        assert_eq!(left[0], "adbreak-2020-01-02T000000Z.db");
        assert_eq!(out.join(&left[1]), path);
    }

    #[test]
    fn round_trips_labels_sources_and_snapshots() {
        let dir = temp("rt");
        let db = Db::open(&dir.join("a.db")).unwrap();
        let rec =
            Recording { source: "recorder".into(), job: 7, channel: "2.1".into(), start: "2026-11-24T07:06:41Z".into(), duration: 600.0, ..Default::default() };
        let id = db.save_recording(&rec).unwrap();
        assert_eq!(db.ids().unwrap(), vec![id.clone()]);
        assert!(db.labels(&id).unwrap().is_none() && db.labels_version(&id).unwrap().is_empty());

        let mut l = Labels::new(rec.clone());
        l.reviewed = vec![Span::new(0.0, 600.0)];
        l.breaks = vec![Break { start: 100.0, end: 200.0, kind: Kind::Ad }];
        let v1 = db.save_labels(&id, &l).unwrap();
        let (back, v) = db.labels(&id).unwrap().unwrap();
        assert_eq!((back.breaks.len(), v.clone()), (1, v1.clone()));
        let fp1 = db.fingerprint().unwrap();
        l.breaks[0].end = 210.0;
        let v2 = db.save_labels(&id, &l).unwrap();
        assert_ne!(v1, v2);
        assert_ne!(fp1, db.fingerprint().unwrap(), "a label change changes the fingerprint");
        let saves: i64 = db.conn().query_row("SELECT count(*) FROM label_history WHERE id = ?1", [&id], |r| r.get(0)).unwrap();
        assert_eq!(saves, 2, "every save is kept");

        let s =
            |name: &str, end: f64| Source { name: name.into(), detail: String::new(), breaks: vec![Span::new(100.0, end).into()], trained_on: String::new() };
        db.save_sources(&id, &[s("comskip", 200.0), s("model", 190.0)]).unwrap();
        db.save_sources(&id, &[s("model", 195.0)]).unwrap();
        let got = db.suggestions(&id).unwrap().unwrap();
        assert_eq!(got.sources.iter().map(|s| (s.name.as_str(), s.breaks[0].end)).collect::<Vec<_>>(), vec![("comskip", 200.0), ("model", 195.0)]);

        // A snapshot and back gives the same dataset, labels version included.
        let snap = dir.join("snap");
        assert_eq!(db.export_dir(&snap).unwrap(), 1);
        let db2 = Db::open(&dir.join("b.db")).unwrap();
        assert_eq!(db2.import_dir(&snap, false).unwrap(), 1);
        assert_eq!(db2.labels(&id).unwrap().unwrap().1, v2);
        assert_eq!(db2.fingerprint().unwrap(), db.fingerprint().unwrap());
        assert_eq!(db2.suggestions(&id).unwrap().unwrap().sources.len(), 2);
    }

    #[test]
    fn import_keeps_newer_labels_unless_replacing() {
        let dir = temp("keep");
        let db = Db::open(&dir.join("a.db")).unwrap();
        let rec = Recording { job: 1, channel: "4.1".into(), start: "2026-11-24T00:00:00Z".into(), duration: 60.0, ..Default::default() };
        let id = db.save_recording(&rec).unwrap();
        let mut l = Labels::new(rec.clone());
        l.notes = "from the app".into();
        db.save_labels(&id, &l).unwrap();
        let snap = dir.join("snap");
        db.export_dir(&snap).unwrap();
        l.notes = "newer".into();
        db.save_labels(&id, &l).unwrap();
        db.import_dir(&snap, false).unwrap();
        assert_eq!(db.labels(&id).unwrap().unwrap().0.notes, "newer");
        db.import_dir(&snap, true).unwrap();
        assert_eq!(db.labels(&id).unwrap().unwrap().0.notes, "from the app");
    }
}
