//! Tab-completion candidates the shell can offer without a network call.
//!
//! Tags, collections, job ids, gallery filenames and the machines you talk to
//! all live on a SERVER, and until now none of them completed: `mold library
//! tag add cat.png --tag <TAB>` offered nothing, so every one of them had to
//! be copied out of a previous listing by hand.
//!
//! A completer cannot ask the server. `CompleteEnv::complete()` runs inside
//! the `#[tokio::main]` runtime, so a `block_on` there deadlocks, and a fresh
//! thread with its own runtime would put an HTTP round trip behind every Tab
//! (`commands::runpod::complete_pod_id` carries the same note). So the
//! commands that ALREADY fetch these lists write what they saw to
//! `$MOLD_HOME/completion-cache.json`, and the completers read that file.
//!
//! Three properties make it safe to read on a keystroke:
//!
//! * **It never fails.** A missing, unreadable, or corrupt file completes
//!   nothing, exactly as `commands::config::complete_profile_name` treats an
//!   absent `mold.db`.
//! * **It never creates anything.** The file's existence is the gate. Pressing
//!   Tab on a machine that has never run mold must not leave state behind.
//! * **It is written atomically** (tmp then rename, the `runpod-state.json`
//!   pattern), so a Tab racing a `mold library list` reads one whole document
//!   or the previous one, never a half-written mixture.
//!
//! It is a CACHE, not an authority: the entries are what some earlier command
//! saw on some machine. A stale tag completes and the server then says it does
//! not exist, which is the same answer typing it by hand gets.

use clap_complete::engine::CompletionCandidate;
use mold_core::Config;
use serde::{Deserialize, Serialize};

const CACHE_FILE: &str = "completion-cache.json";

/// How many entries of each kind are kept. Enough that a working set of tags
/// or a day's prints all complete, small enough that the file stays a single
/// cheap read on a keystroke.
const CAP: usize = 200;

/// The completion candidates this machine has seen, newest first.
///
/// Every field defaults, so a file written by an older or newer mold — one
/// that knows fewer or more lists — still loads.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionCache {
    /// Base URLs of servers a command reached successfully.
    #[serde(default)]
    pub hosts: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Collection display names AND slugs — the `NAME-OR-SLUG` positionals
    /// take either.
    #[serde(default)]
    pub collections: Vec<String>,
    #[serde(default)]
    pub job_ids: Vec<String>,
    #[serde(default)]
    pub filenames: Vec<String>,
    #[serde(default)]
    pub download_ids: Vec<String>,
    #[serde(default)]
    pub workflow_ids: Vec<String>,
}

impl CompletionCache {
    /// Move `values` to the front, dropping duplicates and anything past the
    /// cap.
    ///
    /// Newest-first because that is the order a completion list is useful in:
    /// the print you just made, the tag you just applied, the machine you are
    /// working against.
    fn merge(list: &mut Vec<String>, values: impl IntoIterator<Item = String>) {
        let mut merged: Vec<String> = values
            .into_iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .collect();
        merged.append(list);
        let mut seen = std::collections::HashSet::new();
        merged.retain(|value| seen.insert(value.clone()));
        merged.truncate(CAP);
        *list = merged;
    }

    pub fn record_hosts(&mut self, values: impl IntoIterator<Item = String>) {
        Self::merge(&mut self.hosts, values);
    }

    pub fn record_tags(&mut self, values: impl IntoIterator<Item = String>) {
        Self::merge(&mut self.tags, values);
    }

    pub fn record_collections(&mut self, values: impl IntoIterator<Item = String>) {
        Self::merge(&mut self.collections, values);
    }

    pub fn record_job_ids(&mut self, values: impl IntoIterator<Item = String>) {
        Self::merge(&mut self.job_ids, values);
    }

    pub fn record_filenames(&mut self, values: impl IntoIterator<Item = String>) {
        Self::merge(&mut self.filenames, values);
    }

    /// Recorded by the download verbs. Declared here with the rest so the
    /// document's shape is one decision rather than one per command.
    #[allow(dead_code)]
    pub fn record_download_ids(&mut self, values: impl IntoIterator<Item = String>) {
        Self::merge(&mut self.download_ids, values);
    }

    /// Recorded by the durable 3-D workflow verbs. See
    /// [`CompletionCache::record_download_ids`].
    #[allow(dead_code)]
    pub fn record_workflow_ids(&mut self, values: impl IntoIterator<Item = String>) {
        Self::merge(&mut self.workflow_ids, values);
    }
}

/// Where the cache lives. Never creates the directory — see
/// [`load_if_present`].
fn cache_path() -> Option<std::path::PathBuf> {
    Config::mold_dir().map(|dir| dir.join(CACHE_FILE))
}

/// Read the cache, or nothing at all.
///
/// `None` for a machine with no Mold home, no cache file, or a file that does
/// not parse. A completer must never answer an error, and must never create
/// the file it is reading.
pub fn load_if_present() -> Option<CompletionCache> {
    let path = cache_path()?;
    if !path.exists() {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Read the cache for a WRITER: a machine with no file yet gets an empty one.
fn load_for_update() -> CompletionCache {
    load_if_present().unwrap_or_default()
}

/// Write the cache atomically. Errors are swallowed by [`record`]; a cache
/// that cannot be written is a completion that does not improve, never a
/// command that fails.
fn save(cache: &CompletionCache) -> anyhow::Result<()> {
    let path = cache_path().ok_or_else(|| anyhow::anyhow!("no mold home"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(cache)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Fold what a command just saw into the cache.
///
/// Deliberately infallible: refreshing completions is a side effect of a
/// command that has already succeeded, and a read-only `$MOLD_HOME` must not
/// turn `mold library list` into a failure.
pub fn record(update: impl FnOnce(&mut CompletionCache)) {
    let mut cache = load_for_update();
    update(&mut cache);
    let _ = save(&cache);
}

/// Record the machine a command just reached, plus whatever it listed.
///
/// The host is recorded HERE rather than where the client is built, because
/// construction proves nothing: a URL that was never reachable is not a
/// machine worth completing.
pub fn record_reached_host(host: &str, update: impl FnOnce(&mut CompletionCache)) {
    let host = host.trim_end_matches('/').to_string();
    record(|cache| {
        cache.record_hosts([host]);
        update(cache);
    });
}

fn candidates(select: impl FnOnce(&CompletionCache) -> &Vec<String>) -> Vec<CompletionCandidate> {
    let Some(cache) = load_if_present() else {
        return Vec::new();
    };
    select(&cache)
        .iter()
        .map(CompletionCandidate::new)
        .collect()
}

pub fn complete_host() -> Vec<CompletionCandidate> {
    candidates(|cache| &cache.hosts)
}

pub fn complete_tag() -> Vec<CompletionCandidate> {
    candidates(|cache| &cache.tags)
}

pub fn complete_collection() -> Vec<CompletionCandidate> {
    candidates(|cache| &cache.collections)
}

pub fn complete_job_id() -> Vec<CompletionCandidate> {
    candidates(|cache| &cache.job_ids)
}

pub fn complete_filename() -> Vec<CompletionCandidate> {
    candidates(|cache| &cache.filenames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ENV_LOCK;

    /// Point `$MOLD_HOME` at a fresh directory for the body of one test.
    fn with_home<T>(body: impl FnOnce(&std::path::Path) -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let home = tempfile::tempdir().expect("temp home");
        let previous = std::env::var_os("MOLD_HOME");
        std::env::set_var("MOLD_HOME", home.path());
        let result = body(home.path());
        match previous {
            Some(value) => std::env::set_var("MOLD_HOME", value),
            None => std::env::remove_var("MOLD_HOME"),
        }
        result
    }

    #[test]
    fn a_recorded_cache_round_trips_through_the_file() {
        with_home(|home| {
            record_reached_host("http://plato:7680/", |cache| {
                cache.record_tags(["cat".into(), "village".into()]);
                cache.record_collections(["Winter Scenes".into(), "winter-scenes".into()]);
                cache.record_job_ids(["job-abc".into()]);
                cache.record_filenames(["a.png".into()]);
                cache.record_download_ids(["dl-1".into()]);
                cache.record_workflow_ids(["wf-1".into()]);
            });
            assert!(home.join(CACHE_FILE).exists(), "the cache file is written");

            let loaded = load_if_present().expect("a written cache loads");
            assert_eq!(loaded.hosts, vec!["http://plato:7680".to_string()]);
            assert_eq!(loaded.tags, vec!["cat".to_string(), "village".to_string()]);
            assert_eq!(
                loaded.collections,
                vec!["Winter Scenes".to_string(), "winter-scenes".to_string()]
            );
            assert_eq!(loaded.job_ids, vec!["job-abc".to_string()]);
            assert_eq!(loaded.filenames, vec!["a.png".to_string()]);
            assert_eq!(loaded.download_ids, vec!["dl-1".to_string()]);
            assert_eq!(loaded.workflow_ids, vec!["wf-1".to_string()]);

            let offered: Vec<String> = complete_tag()
                .iter()
                .map(|candidate| candidate.get_value().to_string_lossy().into_owned())
                .collect();
            assert_eq!(offered, vec!["cat".to_string(), "village".to_string()]);
        });
    }

    /// A second command's answers go in FRONT, duplicates collapse to one
    /// entry, and the list never outgrows the cap.
    #[test]
    fn recording_puts_the_newest_first_and_caps_the_list() {
        with_home(|_| {
            record(|cache| cache.record_tags(["old".into(), "shared".into()]));
            record(|cache| cache.record_tags(["new".into(), "shared".into()]));
            let loaded = load_if_present().unwrap();
            assert_eq!(
                loaded.tags,
                vec!["new".to_string(), "shared".to_string(), "old".to_string()],
                "newest first, one entry per value"
            );

            record(|cache| {
                cache.record_filenames((0..CAP + 50).map(|index| format!("print-{index}.png")))
            });
            let loaded = load_if_present().unwrap();
            assert_eq!(loaded.filenames.len(), CAP);
            assert_eq!(loaded.filenames[0], "print-0.png");
        });
    }

    /// Blank values are not candidates, and a host's trailing slash is
    /// dropped so `--host http://plato:7680` completes once, not twice.
    #[test]
    fn blank_values_and_trailing_slashes_never_reach_the_cache() {
        with_home(|_| {
            record_reached_host("http://plato:7680/", |cache| {
                cache.record_tags(["  ".into(), " spaced ".into()]);
            });
            record_reached_host("http://plato:7680", |_| {});
            let loaded = load_if_present().unwrap();
            assert_eq!(loaded.hosts, vec!["http://plato:7680".to_string()]);
            assert_eq!(loaded.tags, vec!["spaced".to_string()]);
        });
    }

    /// Tab on a machine that has never run mold completes nothing AND leaves
    /// nothing behind. The file's own absence is the gate.
    #[test]
    fn an_absent_cache_completes_nothing_and_is_not_created() {
        with_home(|home| {
            assert!(load_if_present().is_none());
            assert!(complete_tag().is_empty());
            assert!(complete_host().is_empty());
            assert!(complete_collection().is_empty());
            assert!(complete_job_id().is_empty());
            assert!(complete_filename().is_empty());
            assert!(
                !home.join(CACHE_FILE).exists(),
                "completing must not create the cache"
            );
        });
    }

    /// A truncated or hand-edited file is treated as absent rather than
    /// aborting the shell's completion.
    #[test]
    fn a_corrupt_cache_completes_nothing() {
        with_home(|home| {
            std::fs::write(home.join(CACHE_FILE), "{ not json").unwrap();
            assert!(load_if_present().is_none());
            assert!(complete_tag().is_empty());
        });
    }

    /// A file written by a mold that knew fewer lists still loads, and
    /// recording into it keeps what was there.
    #[test]
    fn a_cache_missing_newer_fields_still_loads() {
        with_home(|home| {
            std::fs::write(home.join(CACHE_FILE), r#"{"tags":["cat"]}"#).unwrap();
            let loaded = load_if_present().expect("partial documents load");
            assert_eq!(loaded.tags, vec!["cat".to_string()]);
            assert!(loaded.hosts.is_empty());

            record(|cache| cache.record_hosts(["http://plato:7680".into()]));
            let loaded = load_if_present().unwrap();
            assert_eq!(loaded.tags, vec!["cat".to_string()]);
            assert_eq!(loaded.hosts, vec!["http://plato:7680".to_string()]);
        });
    }
}
