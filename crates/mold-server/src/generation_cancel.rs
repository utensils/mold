//! Cooperative inference cancellation, shared by durable chain jobs and
//! ordinary singleton generations.
//!
//! Two independent registries exist on `AppState`: one owned by the chain-job
//! runner, one for public singletons. Keeping them separate keeps a chain
//! cancel from reaching an unrelated print and vice versa; the shutdown path
//! signals both.
//!
//! The register/shutdown race is ordered by the token map's own lock:
//! [`CancelRegistry::register`] reads `shutting_down` while holding it, so a
//! claim racing shutdown is either visible to `request_all` or cancelled on
//! insert.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

pub struct CancelRegistry {
    tokens: Mutex<HashMap<String, CancelEntry>>,
    shutting_down: AtomicBool,
}

struct CancelEntry {
    token: mold_inference::InferenceCancellationToken,
    user_requested: bool,
}

impl CancelRegistry {
    pub fn new() -> Self {
        Self {
            tokens: Mutex::new(HashMap::new()),
            shutting_down: AtomicBool::new(false),
        }
    }

    pub fn register(&self, job_id: &str) {
        let mut tokens = self
            .tokens
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = tokens
            .entry(job_id.to_string())
            .or_insert_with(|| CancelEntry {
                token: mold_inference::InferenceCancellationToken::default(),
                user_requested: false,
            });
        // The load happens while the token map is locked. If shutdown races
        // after this load, request_all() must acquire the same lock and will
        // observe/cancel this token; if shutdown won first, cancel it here.
        if self.shutting_down.load(Ordering::Acquire) {
            entry.token.cancel();
        }
    }

    pub fn unregister(&self, job_id: &str) {
        self.tokens
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(job_id);
    }

    pub fn request(&self, job_id: &str) -> bool {
        let mut tokens = self
            .tokens
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(entry) = tokens.get_mut(job_id) else {
            return false;
        };
        entry.user_requested = true;
        entry.token.cancel();
        true
    }

    /// Revoke a running job even when dispatch marked it running just before
    /// its owner thread registered the inference token.
    pub fn request_or_register(&self, job_id: &str) {
        let mut tokens = self
            .tokens
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = tokens
            .entry(job_id.to_string())
            .or_insert_with(|| CancelEntry {
                token: mold_inference::InferenceCancellationToken::default(),
                user_requested: false,
            });
        entry.user_requested = true;
        entry.token.cancel();
    }

    pub fn request_all(&self) -> usize {
        // Fence future registrations before snapshotting current attempts.
        // register() checks this flag while holding the token-map lock, making
        // a claim racing shutdown either visible here or cancelled on insert.
        self.shutting_down.store(true, Ordering::Release);
        let tokens = self
            .tokens
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .map(|entry| entry.token.clone())
            .collect::<Vec<_>>();
        for token in &tokens {
            token.cancel();
        }
        tokens.len()
    }

    pub fn is_cancelled(&self, job_id: &str) -> bool {
        self.tokens
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(job_id)
            .is_some_and(|entry| entry.token.is_cancelled())
    }

    pub fn was_user_requested(&self, job_id: &str) -> bool {
        self.tokens
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(job_id)
            .is_some_and(|entry| entry.user_requested)
    }

    pub fn token(&self, job_id: &str) -> mold_inference::InferenceCancellationToken {
        let mut tokens = self
            .tokens
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = tokens
            .entry(job_id.to_string())
            .or_insert_with(|| CancelEntry {
                token: mold_inference::InferenceCancellationToken::default(),
                user_requested: false,
            });
        if self.shutting_down.load(Ordering::Acquire) {
            entry.token.cancel();
        }
        entry.token.clone()
    }
}

impl CancelRegistry {
    /// Whether a token is currently registered for this job. Test-facing:
    /// the invariant callers care about is that a finished attempt leaves no
    /// token behind.
    pub fn is_registered(&self, job_id: &str) -> bool {
        self.tokens
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(job_id)
    }
}

impl Default for CancelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_all_cancels_current_and_future_registrations() {
        let registry = CancelRegistry::new();
        let running = registry.token("job-1");
        assert!(!running.is_cancelled());

        assert_eq!(registry.request_all(), 1);
        assert!(running.is_cancelled());

        // A job admitted after the fence went up is cancelled on insert, not
        // silently allowed to start.
        assert!(registry.token("job-2").is_cancelled());
    }

    #[test]
    fn unregistering_leaves_no_token_to_cancel() {
        let registry = CancelRegistry::new();
        let _ = registry.token("job-1");
        assert!(registry.request("job-1"));
        assert!(registry.is_cancelled("job-1"));

        registry.unregister("job-1");
        assert!(!registry.request("job-1"));
        assert!(!registry.is_cancelled("job-1"));
    }

    #[test]
    fn request_before_worker_registration_cancels_the_future_owner_token() {
        let registry = CancelRegistry::new();
        registry.request_or_register("job-1");

        let worker_token = registry.token("job-1");
        assert!(worker_token.is_cancelled());
        assert!(registry.was_user_requested("job-1"));
    }
}
