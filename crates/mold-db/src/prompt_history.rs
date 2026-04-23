//! Prompt history backed by the `prompt_history` table (v5 schema).
//!
//! Replaces `~/.mold/prompt-history.jsonl`. Still bounded by the caller —
//! callers use `trim_to(N)` after a `push` to cap size (the legacy limit
//! was 500).

use anyhow::Result;
use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::db::MetadataDb;

/// One recorded prompt. Kept structurally identical to the legacy
/// `HistoryEntry` in `mold-tui/src/history.rs` so imports are trivial.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HistoryEntry {
    pub prompt: String,
    pub negative: Option<String>,
    pub model: String,
    /// Unix epoch milliseconds. Set by `push` if omitted by caller.
    pub created_at_ms: i64,
}

impl HistoryEntry {
    pub fn new(prompt: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            negative: None,
            model: model.into(),
            created_at_ms: now_ms(),
        }
    }
}

/// Borrowed view onto the prompt-history table.
pub struct PromptHistory<'a> {
    db: &'a MetadataDb,
}

impl<'a> PromptHistory<'a> {
    pub fn new(db: &'a MetadataDb) -> Self {
        Self { db }
    }

    /// Append a new entry. If `entry.created_at_ms` is 0, stamps `now()`.
    pub fn push(&self, entry: &HistoryEntry) -> Result<i64> {
        let ts = if entry.created_at_ms > 0 {
            entry.created_at_ms
        } else {
            now_ms()
        };
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO prompt_history (prompt, negative, model, created_at_ms)
                 VALUES (?1, ?2, ?3, ?4)",
                params![entry.prompt, entry.negative, entry.model, ts],
            )?;
            let id: i64 = conn.query_row("SELECT last_insert_rowid()", [], |r| r.get(0))?;
            Ok(id)
        })
    }

    /// Most-recent `limit` entries, newest first.
    pub fn recent(&self, limit: usize) -> Result<Vec<HistoryEntry>> {
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT prompt, negative, model, created_at_ms
                 FROM prompt_history
                 ORDER BY created_at_ms DESC, id DESC
                 LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit as i64], |r| {
                Ok(HistoryEntry {
                    prompt: r.get(0)?,
                    negative: r.get(1)?,
                    model: r.get(2)?,
                    created_at_ms: r.get(3)?,
                })
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        })
    }

    /// Substring search across `prompt` (case-insensitive via SQLite's
    /// `LIKE`, which is case-insensitive for ASCII by default).
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<HistoryEntry>> {
        let pat = format!("%{query}%");
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT prompt, negative, model, created_at_ms
                 FROM prompt_history
                 WHERE prompt LIKE ?1
                 ORDER BY created_at_ms DESC, id DESC
                 LIMIT ?2",
            )?;
            let rows = stmt.query_map(params![pat, limit as i64], |r| {
                Ok(HistoryEntry {
                    prompt: r.get(0)?,
                    negative: r.get(1)?,
                    model: r.get(2)?,
                    created_at_ms: r.get(3)?,
                })
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        })
    }

    /// Cap the table at `keep` rows by dropping the oldest entries.
    /// Returns the number of rows deleted.
    pub fn trim_to(&self, keep: usize) -> Result<usize> {
        self.db.with_conn(|conn| {
            let n = conn.execute(
                "DELETE FROM prompt_history
                 WHERE id IN (
                    SELECT id FROM prompt_history
                    ORDER BY created_at_ms DESC, id DESC
                    LIMIT -1 OFFSET ?1
                 )",
                params![keep as i64],
            )?;
            Ok(n)
        })
    }

    pub fn count(&self) -> Result<i64> {
        self.db.with_conn(|conn| {
            let n: i64 = conn.query_row("SELECT COUNT(*) FROM prompt_history", [], |r| r.get(0))?;
            Ok(n)
        })
    }

    pub fn clear(&self) -> Result<usize> {
        self.db.with_conn(|conn| {
            let n = conn.execute("DELETE FROM prompt_history", [])?;
            Ok(n)
        })
    }
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> MetadataDb {
        MetadataDb::open_in_memory().unwrap()
    }

    fn entry(prompt: &str, model: &str, ts: i64) -> HistoryEntry {
        HistoryEntry {
            prompt: prompt.into(),
            negative: None,
            model: model.into(),
            created_at_ms: ts,
        }
    }

    #[test]
    fn push_and_recent_roundtrip() {
        let db = db();
        let h = PromptHistory::new(&db);
        h.push(&entry("first", "flux-dev:q4", 1_000)).unwrap();
        h.push(&entry("second", "flux-dev:q4", 2_000)).unwrap();
        h.push(&entry("third", "sdxl:fp16", 3_000)).unwrap();
        let got = h.recent(10).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].prompt, "third");
        assert_eq!(got[1].prompt, "second");
        assert_eq!(got[2].prompt, "first");
    }

    #[test]
    fn recent_respects_limit() {
        let db = db();
        let h = PromptHistory::new(&db);
        // Start ts at 1 — ts=0 triggers the "stamp now()" path and would
        // shuffle p0 to the top of the list.
        for i in 0..10 {
            h.push(&entry(&format!("p{i}"), "m", (i as i64 + 1) * 100))
                .unwrap();
        }
        let got = h.recent(3).unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].prompt, "p9");
    }

    #[test]
    fn push_stamps_now_when_ts_zero() {
        let db = db();
        let h = PromptHistory::new(&db);
        let e = HistoryEntry {
            prompt: "now".into(),
            negative: None,
            model: "m".into(),
            created_at_ms: 0,
        };
        h.push(&e).unwrap();
        let got = h.recent(1).unwrap();
        assert!(
            got[0].created_at_ms > 0,
            "push must stamp a real timestamp when ts=0"
        );
    }

    #[test]
    fn trim_to_keeps_newest_entries() {
        let db = db();
        let h = PromptHistory::new(&db);
        for i in 0..10 {
            h.push(&entry(&format!("p{i}"), "m", (i as i64 + 1) * 100))
                .unwrap();
        }
        assert_eq!(h.count().unwrap(), 10);
        let dropped = h.trim_to(3).unwrap();
        assert_eq!(dropped, 7);
        assert_eq!(h.count().unwrap(), 3);
        let remaining: Vec<_> = h
            .recent(10)
            .unwrap()
            .into_iter()
            .map(|e| e.prompt)
            .collect();
        assert_eq!(remaining, vec!["p9", "p8", "p7"]);
    }

    #[test]
    fn trim_to_bigger_than_count_is_noop() {
        let db = db();
        let h = PromptHistory::new(&db);
        h.push(&entry("only", "m", 100)).unwrap();
        let dropped = h.trim_to(500).unwrap();
        assert_eq!(dropped, 0);
        assert_eq!(h.count().unwrap(), 1);
    }

    #[test]
    fn search_matches_substring_case_insensitive() {
        let db = db();
        let h = PromptHistory::new(&db);
        h.push(&entry("A Sunny Day", "m", 1)).unwrap();
        h.push(&entry("cloudy morning", "m", 2)).unwrap();
        h.push(&entry("SUNSET over sea", "m", 3)).unwrap();
        let hits = h.search("sun", 10).unwrap();
        let prompts: Vec<_> = hits.iter().map(|e| e.prompt.as_str()).collect();
        assert!(prompts.contains(&"A Sunny Day"));
        assert!(prompts.contains(&"SUNSET over sea"));
        assert!(!prompts.contains(&"cloudy morning"));
    }

    #[test]
    fn clear_empties_table() {
        let db = db();
        let h = PromptHistory::new(&db);
        h.push(&entry("a", "m", 1)).unwrap();
        h.push(&entry("b", "m", 2)).unwrap();
        assert_eq!(h.clear().unwrap(), 2);
        assert_eq!(h.count().unwrap(), 0);
    }

    #[test]
    fn negative_prompt_roundtrips() {
        let db = db();
        let h = PromptHistory::new(&db);
        h.push(&HistoryEntry {
            prompt: "with neg".into(),
            negative: Some("bad".into()),
            model: "m".into(),
            created_at_ms: 1,
        })
        .unwrap();
        let got = h.recent(1).unwrap();
        assert_eq!(got[0].negative.as_deref(), Some("bad"));
    }
}
