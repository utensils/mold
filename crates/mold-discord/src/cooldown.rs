use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Per-user cooldown tracker. Thread-safe via `std::sync::Mutex`.
#[derive(Debug)]
pub struct CooldownTracker {
    last_use: Mutex<HashMap<u64, Instant>>,
}

impl CooldownTracker {
    pub fn new() -> Self {
        Self {
            last_use: Mutex::new(HashMap::new()),
        }
    }

    /// Check if the user is within cooldown.
    /// Returns `Ok(())` if they can proceed, or `Err(remaining)` with the time left.
    pub fn check(&self, user_id: u64, cooldown: Duration) -> Result<(), Duration> {
        let map = self.last_use.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(last) = map.get(&user_id) {
            let elapsed = last.elapsed();
            if elapsed < cooldown {
                return Err(cooldown - elapsed);
            }
        }
        Ok(())
    }

    /// Record that a user just performed a generation.
    pub fn record(&self, user_id: u64) {
        let mut map = self.last_use.lock().unwrap_or_else(|e| e.into_inner());
        map.insert(user_id, Instant::now());
    }
}

impl Default for CooldownTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_first_use() {
        let tracker = CooldownTracker::new();
        assert!(tracker.check(1, Duration::from_secs(10)).is_ok());
    }

    #[test]
    fn blocks_within_cooldown() {
        let tracker = CooldownTracker::new();
        tracker.record(1);
        let result = tracker.check(1, Duration::from_secs(60));
        assert!(result.is_err());
        let remaining = result.unwrap_err();
        assert!(remaining.as_secs() > 0);
        assert!(remaining.as_secs() <= 60);
    }

    #[test]
    fn allows_after_cooldown_expires() {
        let tracker = CooldownTracker::new();
        tracker.record(1);
        // Zero-duration cooldown should always pass
        assert!(tracker.check(1, Duration::ZERO).is_ok());
    }

    #[test]
    fn tracks_users_independently() {
        let tracker = CooldownTracker::new();
        tracker.record(1);
        // User 2 should not be affected by user 1's cooldown
        assert!(tracker.check(2, Duration::from_secs(60)).is_ok());
        // User 1 should be on cooldown
        assert!(tracker.check(1, Duration::from_secs(60)).is_err());
    }

    #[test]
    fn record_resets_cooldown() {
        let tracker = CooldownTracker::new();
        tracker.record(1);
        // Immediately record again — cooldown restarts
        tracker.record(1);
        let result = tracker.check(1, Duration::from_secs(60));
        assert!(result.is_err());
    }
}
