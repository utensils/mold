use std::path::PathBuf;

use serde::{Deserialize, Serialize};

const MAX_ENTRIES: usize = 500;

/// A single prompt history entry.
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
    entries: Vec<HistoryEntry>,
    /// Current position in history (-1 = not navigating, 0 = most recent).
    cursor: Option<usize>,
    /// The prompt text before the user started navigating (to restore on cancel).
    draft: Option<String>,
}

fn history_path() -> Option<PathBuf> {
    mold_core::Config::mold_dir().map(|d| d.join("prompt-history.jsonl"))
}

impl PromptHistory {
    /// Load history from disk. Returns empty if file missing.
    pub fn load() -> Self {
        let entries = match history_path() {
            Some(path) => match std::fs::read_to_string(&path) {
                Ok(contents) => contents
                    .lines()
                    .filter_map(|line| serde_json::from_str::<HistoryEntry>(line).ok())
                    .collect(),
                Err(_) => Vec::new(),
            },
            None => Vec::new(),
        };
        Self {
            entries,
            cursor: None,
            draft: None,
        }
    }

    /// Append an entry and save to disk.
    pub fn push(&mut self, entry: HistoryEntry) {
        if self.push_entry(entry) {
            self.save();
        }
    }

    /// Append an entry without saving. Returns true if entry was added.
    pub(crate) fn push_entry(&mut self, entry: HistoryEntry) -> bool {
        // Skip empty/whitespace-only prompts
        if entry.prompt.trim().is_empty() {
            return false;
        }
        // Don't add duplicate of the most recent entry
        if let Some(last) = self.entries.last() {
            if last.prompt == entry.prompt {
                return false;
            }
        }
        self.entries.push(entry);
        // Trim oldest entries beyond the limit
        if self.entries.len() > MAX_ENTRIES {
            let excess = self.entries.len() - MAX_ENTRIES;
            self.entries.drain(..excess);
        }
        true
    }

    /// Save history to disk (best-effort).
    fn save(&self) {
        let path = match history_path() {
            Some(p) => p,
            None => return,
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let lines: Vec<String> = self
            .entries
            .iter()
            .filter_map(|e| serde_json::to_string(e).ok())
            .collect();
        let _ = std::fs::write(&path, lines.join("\n") + "\n");
    }

    /// Start or continue navigating backward through history.
    /// Returns the prompt at the new cursor position, or None if at the end.
    pub fn prev(&mut self, current_prompt: &str) -> Option<&str> {
        if self.entries.is_empty() {
            return None;
        }
        let new_cursor = match self.cursor {
            None => {
                // First navigation — save the current draft
                self.draft = Some(current_prompt.to_string());
                self.entries.len().saturating_sub(1)
            }
            Some(pos) => {
                if pos == 0 {
                    return None; // already at oldest
                }
                pos - 1
            }
        };
        self.cursor = Some(new_cursor);
        Some(&self.entries[new_cursor].prompt)
    }

    /// Navigate forward through history toward the draft.
    /// Returns the prompt, or the original draft if we're past the newest.
    pub fn next(&mut self, _current_prompt: &str) -> Option<&str> {
        match self.cursor {
            None => None, // not navigating
            Some(pos) => {
                if pos + 1 >= self.entries.len() {
                    // Past the newest — return to draft
                    self.cursor = None;
                    self.draft.as_deref()
                } else {
                    self.cursor = Some(pos + 1);
                    Some(&self.entries[pos + 1].prompt)
                }
            }
        }
    }

    /// Reset navigation state (e.g., when user types something).
    pub fn reset_cursor(&mut self) {
        self.cursor = None;
        self.draft = None;
    }

    /// Search history entries by substring (case-insensitive).
    /// Returns matches most-recent-first.
    pub fn search(&self, query: &str) -> Vec<&HistoryEntry> {
        let query_lower = query.to_lowercase();
        self.entries
            .iter()
            .rev()
            .filter(|e| e.prompt.to_lowercase().contains(&query_lower))
            .collect()
    }

    /// Total number of history entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether we're actively navigating history.
    pub fn is_navigating(&self) -> bool {
        self.cursor.is_some()
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
        // Newest should be "new"
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
        assert_eq!(history.prev("draft"), None); // at oldest
    }

    #[test]
    fn next_navigates_forward_to_draft() {
        let mut history = PromptHistory {
            entries: vec![make_entry("old"), make_entry("new")],
            cursor: None,
            draft: None,
        };
        history.prev("my draft"); // cursor at "new" (index 1)
        history.prev("my draft"); // cursor at "old" (index 0)
        assert_eq!(history.next(""), Some("new"));
        assert_eq!(history.next(""), Some("my draft")); // back to draft
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
        // Most recent first
        assert_eq!(results[0].prompt, "CATS everywhere");
        assert_eq!(results[1].prompt, "a Cat in a hat");
    }

    #[test]
    fn search_empty_query_returns_all() {
        let history = PromptHistory {
            entries: vec![make_entry("a"), make_entry("b")],
            cursor: None,
            draft: None,
        };
        assert_eq!(history.search("").len(), 2);
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
        // next without prior prev should return None
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
        assert!(!history.push_entry(make_entry("hello"))); // duplicate
        assert!(history.push_entry(make_entry("world"))); // different
    }
}
