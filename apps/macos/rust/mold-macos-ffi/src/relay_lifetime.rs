//! An alive flag may go false and true between two connector polls. Retain a
//! monotonic generation so a restarted engine cannot revive its predecessor.
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) struct EngineLifetime(AtomicU64);
impl EngineLifetime {
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }
    pub fn snapshot(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
    pub fn invalidate(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
    pub fn matches(&self, snapshot: u64, alive: bool) -> bool {
        alive && self.snapshot() == snapshot
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exit_invalidates_a_start_even_while_connector_ownership_is_locked() {
        let lifetime = EngineLifetime::new();
        let connector = std::sync::Mutex::new(());
        let start = lifetime.snapshot();
        let _starting = connector.lock().unwrap();
        lifetime.invalidate(); // exit cannot take the connector lock
        assert!(!lifetime.matches(start, false));
        assert!(
            !lifetime.matches(start, true),
            "rapid restart must not revive the old connector"
        );
        assert!(lifetime.matches(lifetime.snapshot(), true));
    }
    #[test]
    fn a_dead_engine_cannot_publish_even_before_generation_withdrawal() {
        let lifetime = EngineLifetime::new();
        assert!(!lifetime.matches(lifetime.snapshot(), false));
    }
}
