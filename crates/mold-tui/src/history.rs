//! Prompt history, backed by the `prompt_history` table in the metadata DB.
//!
//! The in-memory shape (cursor + draft for prev/next navigation) is the
//! same as the previous JSONL-backed implementation. Only the storage
//! moved: [`Self::load`] reads from the DB, [`Self::save`] writes there,
//! and the first launch on an upgraded install imports
//! `~/.mold/prompt-history.jsonl` via [`import_legacy_jsonl`].

use mold_db::{HistoryEntry as DbEntry, MetadataDb, PromptHistory as DbHistory};
use serde::{Deserialize, Serialize};

/// Matches the legacy limit from the JSONL writer.
const MAX_ENTRIES: usize = 500;

/// A single prompt history entry. Kept `Serialize`/`Deserialize` so the
/// legacy JSONL importer can parse old files byte-for-byte.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative: Option<String>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub timestamp: u64,
}

/// Prompt history with navigation and fuzzy search.
pub struct PromptHistory {
    /// In-memory cache; source of truth for prev/next navigation.
    /// Ordered oldest-first to match legacy semantics.
    entries: Vec<HistoryEntry>,
    /// Current position in history (None = not navigating, 0 = oldest).
    cursor: Option<usize>,
    /// The prompt text before the user started navigating (to restore on cancel).
    draft: Option<String>,
}

fn open_db() -> Option<MetadataDb> {
    match mold_db::open_default() {
        Ok(Some(db)) => Some(db),
        _ => None,
    }
}

impl PromptHistory {
    /// Load history from the DB (newest first in storage, flipped to
    /// oldest-first for the in-memory cache so `prev()`/`next()` keep
    /// their existing semantics). Returns empty when the DB is
    /// unavailable.
    pub fn load() -> Self {
        let mut entries = Vec::new();
        if let Some(db) = open_db() {
            let h = DbHistory::new(&db);
            if let Ok(rows) = h.recent(MAX_ENTRIES) {
                // DB returns newest-first; flip to oldest-first.
                entries = rows
                    .into_iter()
                    .rev()
                    .map(|e: DbEntry| HistoryEntry {
                        prompt: e.prompt,
                        negative: e.negative,
                        model: e.model,
                        timestamp: (e.created_at_ms / 1000).max(0) as u64,
                    })
                    .collect();
            }
        }
        Self {
            entries,
            cursor: None,
            draft: None,
        }
    }

    /// Append an entry and persist to the DB (best-effort).
    pub fn push(&mut self, entry: HistoryEntry) {
        if self.push_entry(entry.clone()) {
            self.persist(&entry);
        }
    }

    /// Append an entry without persisting. Returns true if entry was added.
    pub(crate) fn push_entry(&mut self, entry: HistoryEntry) -> bool {
        if entry.prompt.trim().is_empty() {
            return false;
        }
        if let Some(last) = self.entries.last() {
            if last.prompt == entry.prompt {
                return false;
            }
        }
        self.entries.push(entry);
        if self.entries.len() > MAX_ENTRIES {
            let excess = self.entries.len() - MAX_ENTRIES;
            self.entries.drain(..excess);
        }
        true
    }

    /// Persist a single entry to the DB and trim the table to MAX_ENTRIES.
    fn persist(&self, entry: &HistoryEntry) {
        let Some(db) = open_db() else {
            return;
        };
        let h = DbHistory::new(&db);
        let db_entry = DbEntry {
            prompt: entry.prompt.clone(),
            negative: entry.negative.clone(),
            model: entry.model.clone(),
            // `timestamp` is seconds; the DB wants ms. 0 means "stamp now".
            created_at_ms: if entry.timestamp == 0 {
                0
            } else {
                (entry.timestamp as i64) * 1000
            },
        };
        if let Err(e) = h.push(&db_entry) {
            tracing::warn!(error = %e, "prompt history: push failed");
        }
        if let Err(e) = h.trim_to(MAX_ENTRIES) {
            tracing::warn!(error = %e, "prompt history: trim failed");
        }
    }

    /// Start or continue navigating backward through history.
    pub fn prev(&mut self, current_prompt: &str) -> Option<&str> {
        if self.entries.is_empty() {
            return None;
        }
        let new_cursor = match self.cursor {
            None => {
                self.draft = Some(current_prompt.to_string());
                self.entries.len().saturating_sub(1)
            }
            Some(pos) => {
                if pos == 0 {
                    return None;
                }
                pos - 1
            }
        };
        self.cursor = Some(new_cursor);
        Some(&self.entries[new_cursor].prompt)
    }

    /// Navigate forward through history toward the draft.
    pub fn next(&mut self, _current_prompt: &str) -> Option<&str> {
        match self.cursor {
            None => None,
            Some(pos) => {
                if pos + 1 >= self.entries.len() {
                    self.cursor = None;
                    self.draft.as_deref()
                } else {
                    self.cursor = Some(pos + 1);
                    Some(&self.entries[pos + 1].prompt)
                }
            }
        }
    }

    pub fn reset_cursor(&mut self) {
        self.cursor = None;
        self.draft = None;
    }

    /// Search history entries by substring (case-insensitive).
    pub fn search(&self, query: &str) -> Vec<&HistoryEntry> {
        let query_lower = query.to_lowercase();
        self.entries
            .iter()
            .rev()
            .filter(|e| e.prompt.to_lowercase().contains(&query_lower))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn recent(&self, max: usize) -> impl Iterator<Item = &HistoryEntry> {
        self.entries.iter().rev().take(max)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn is_navigating(&self) -> bool {
        self.cursor.is_some()
    }
}

/// Import `~/.mold/prompt-history.jsonl` into the DB exactly once.
/// Called from `session::import_legacy_json_once`; no-op if the file is
/// missing or the DB already holds rows.
pub(crate) fn import_legacy_jsonl(db: &MetadataDb) {
    let path = match mold_core::Config::mold_dir().map(|d| d.join("prompt-history.jsonl")) {
        Some(p) if p.exists() => p,
        _ => return,
    };
    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e,
                "prompt history: legacy JSONL read failed");
            return;
        }
    };
    let entries: Vec<HistoryEntry> = contents
        .lines()
        .filter_map(|line| serde_json::from_str::<HistoryEntry>(line).ok())
        .collect();
    if entries.is_empty() {
        // Still rename the file so we don't reparse on every launch.
        rename_to_migrated(&path);
        return;
    }

    let h = DbHistory::new(db);
    let mut imported = 0;
    for e in &entries {
        let db_entry = DbEntry {
            prompt: e.prompt.clone(),
            negative: e.negative.clone(),
            model: e.model.clone(),
            created_at_ms: if e.timestamp == 0 {
                0
            } else {
                (e.timestamp as i64) * 1000
            },
        };
        if h.push(&db_entry).is_ok() {
            imported += 1;
        }
    }
    let _ = h.trim_to(MAX_ENTRIES);
    rename_to_migrated(&path);
    tracing::info!(
        path = %path.display(),
        imported,
        "imported legacy prompt-history.jsonl into metadata DB"
    );
}

fn rename_to_migrated(path: &std::path::Path) {
    if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
        if let Some(parent) = path.parent() {
            let dst = parent.join(format!("{fname}.migrated"));
            if let Err(e) = std::fs::rename(path, &dst) {
                tracing::warn!(src = %path.display(), dst = %dst.display(), error = %e,
                    "rename legacy history file to .migrated failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(prompt: &str) -> HistoryEntry {
        HistoryEntry {
            prompt: prompt.to_string(),
            negative: None,
            model: "test".to_string(),
            timestamp: 0,
        }
    }

    // ---------- in-memory navigation + trim behaviour ----------
    // These tests exercise the cache logic and don't touch the DB, so they
    // don't need env isolation.

    #[test]
    fn push_and_len() {
        let mut history = PromptHistory {
            entries: Vec::new(),
            cursor: None,
            draft: None,
        };
        history.entries.push(make_entry("first"));
        history.entries.push(make_entry("second"));
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn push_deduplicates_consecutive() {
        let mut history = PromptHistory {
            entries: Vec::new(),
            cursor: None,
            draft: None,
        };
        history.push_entry(make_entry("hello"));
        history.push_entry(make_entry("hello"));
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn push_skips_empty_prompts() {
        let mut history = PromptHistory {
            entries: Vec::new(),
            cursor: None,
            draft: None,
        };
        assert!(!history.push_entry(make_entry("")));
        assert!(!history.push_entry(make_entry("   ")));
        assert!(history.push_entry(make_entry("real prompt")));
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn push_trims_oldest() {
        let mut history = PromptHistory {
            entries: (0..MAX_ENTRIES + 10)
                .map(|i| make_entry(&format!("prompt {i}")))
                .collect(),
            cursor: None,
            draft: None,
        };
        history.push_entry(make_entry("new"));
        assert!(history.len() <= MAX_ENTRIES);
        assert_eq!(history.entries.last().unwrap().prompt, "new");
    }

    #[test]
    fn prev_navigates_backward() {
        let mut history = PromptHistory {
            entries: vec![
                make_entry("first"),
                make_entry("second"),
                make_entry("third"),
            ],
            cursor: None,
            draft: None,
        };
        assert_eq!(history.prev("draft"), Some("third"));
        assert_eq!(history.prev("draft"), Some("second"));
        assert_eq!(history.prev("draft"), Some("first"));
        assert_eq!(history.prev("draft"), None);
    }

    #[test]
    fn next_navigates_forward_to_draft() {
        let mut history = PromptHistory {
            entries: vec![make_entry("old"), make_entry("new")],
            cursor: None,
            draft: None,
        };
        history.prev("my draft");
        history.prev("my draft");
        assert_eq!(history.next(""), Some("new"));
        assert_eq!(history.next(""), Some("my draft"));
        assert!(!history.is_navigating());
    }

    #[test]
    fn search_case_insensitive() {
        let history = PromptHistory {
            entries: vec![
                make_entry("a Cat in a hat"),
                make_entry("sunset mountains"),
                make_entry("CATS everywhere"),
            ],
            cursor: None,
            draft: None,
        };
        let results = history.search("cat");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].prompt, "CATS everywhere");
        assert_eq!(results[1].prompt, "a Cat in a hat");
    }

    #[test]
    fn reset_cursor_clears_state() {
        let mut history = PromptHistory {
            entries: vec![make_entry("test")],
            cursor: None,
            draft: None,
        };
        history.prev("draft");
        assert!(history.is_navigating());
        history.reset_cursor();
        assert!(!history.is_navigating());
    }

    #[test]
    fn entry_serialization() {
        let entry = HistoryEntry {
            prompt: "a cat".to_string(),
            negative: Some("blurry".to_string()),
            model: "flux:q8".to_string(),
            timestamp: 12345,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let restored: HistoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.prompt, "a cat");
        assert_eq!(restored.negative, Some("blurry".to_string()));
    }

    #[test]
    fn prev_does_not_navigate_when_empty() {
        let mut history = PromptHistory {
            entries: Vec::new(),
            cursor: None,
            draft: None,
        };
        assert!(history.prev("current").is_none());
        assert!(!history.is_navigating());
    }

    #[test]
    fn next_without_prev_returns_none() {
        let mut history = PromptHistory {
            entries: vec![make_entry("test")],
            cursor: None,
            draft: None,
        };
        assert!(history.next("current").is_none());
    }

    #[test]
    fn push_entry_returns_false_for_duplicates() {
        let mut history = PromptHistory {
            entries: Vec::new(),
            cursor: None,
            draft: None,
        };
        assert!(history.push_entry(make_entry("hello")));
        assert!(!history.push_entry(make_entry("hello")));
        assert!(history.push_entry(make_entry("world")));
    }

    #[test]
    fn recent_yields_newest_first_up_to_max() {
        let mut history = PromptHistory {
            entries: vec![
                make_entry("oldest"),
                make_entry("middle"),
                make_entry("newest"),
            ],
            cursor: None,
            draft: None,
        };
        let prompts: Vec<&str> = history.recent(5).map(|e| e.prompt.as_str()).collect();
        assert_eq!(prompts, vec!["newest", "middle", "oldest"]);

        let capped: Vec<&str> = history.recent(1).map(|e| e.prompt.as_str()).collect();
        assert_eq!(capped, vec!["newest"]);

        history.entries.clear();
        assert_eq!(history.recent(3).count(), 0);
    }

    // ---------- DB round-trip ----------

    use crate::test_env::with_isolated_env;

    #[test]
    fn push_then_load_roundtrips_through_db() {
        with_isolated_env(|_home| {
            let mut h = PromptHistory::load();
            h.push(HistoryEntry {
                prompt: "first prompt".into(),
                negative: None,
                model: "flux-dev:q4".into(),
                timestamp: 1_000,
            });
            h.push(HistoryEntry {
                prompt: "second prompt".into(),
                negative: Some("ugly".into()),
                model: "sdxl:fp16".into(),
                timestamp: 2_000,
            });

            let reloaded = PromptHistory::load();
            // Cache is oldest-first.
            let prompts: Vec<_> = reloaded.entries.iter().map(|e| &e.prompt).collect();
            assert_eq!(
                prompts,
                vec![&"first prompt".to_string(), &"second prompt".to_string()]
            );
        });
    }

    #[test]
    fn legacy_jsonl_import_populates_db_and_renames_file() {
        with_isolated_env(|home| {
            let src = home.join("prompt-history.jsonl");
            std::fs::write(
                &src,
                r#"{"prompt":"cat","model":"m","timestamp":1000}
{"prompt":"dog","model":"m","timestamp":2000}
{"prompt":"bird","model":"m","timestamp":3000}"#,
            )
            .unwrap();

            // Trigger import via TuiSession::load (the unified entry point).
            let _ = super::super::session::TuiSession::load();

            assert!(!src.exists());
            assert!(home.join("prompt-history.jsonl.migrated").exists());

            let h = PromptHistory::load();
            let prompts: Vec<_> = h.entries.iter().map(|e| e.prompt.as_str()).collect();
            assert_eq!(prompts, vec!["cat", "dog", "bird"]);
        });
    }

    #[test]
    fn db_disabled_keeps_history_in_memory_only() {
        with_isolated_env(|_home| {
            std::env::set_var("MOLD_DB_DISABLE", "1");
            let mut h = PromptHistory::load();
            h.push(HistoryEntry {
                prompt: "in memory".into(),
                negative: None,
                model: "m".into(),
                timestamp: 0,
            });
            assert_eq!(h.len(), 1);

            // A fresh load returns empty because nothing was persisted.
            let fresh = PromptHistory::load();
            assert!(fresh.is_empty());
            std::env::remove_var("MOLD_DB_DISABLE");
        });
    }
}
