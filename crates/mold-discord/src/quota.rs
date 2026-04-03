use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Usage record for a single user on a single UTC day.
#[derive(Debug, Clone)]
struct DayUsage {
    /// UTC day ordinal (days since Unix epoch).
    day: u32,
    /// Number of generations performed on that day.
    count: u32,
}

/// Per-user daily quota tracker. Thread-safe via `std::sync::Mutex`.
///
/// Quotas reset lazily at midnight UTC — when the stored day differs from
/// the current day, the count is treated as zero.
#[derive(Debug)]
pub struct QuotaTracker {
    usage: Mutex<HashMap<u64, DayUsage>>,
}

impl QuotaTracker {
    pub fn new() -> Self {
        Self {
            usage: Mutex::new(HashMap::new()),
        }
    }

    /// Check if the user has quota remaining.
    /// Returns `Some(remaining)` if allowed, `None` if exhausted.
    /// When `max_daily` is `None`, quota is unlimited and always returns `Some`.
    pub fn check(&self, user_id: u64, max_daily: Option<u32>) -> Option<u32> {
        let max = match max_daily {
            Some(m) => m,
            None => return Some(u32::MAX),
        };
        let used = self.usage_for_day(user_id, current_utc_day());
        if used >= max {
            None
        } else {
            Some(max - used)
        }
    }

    /// Increment the user's generation count for today.
    pub fn record(&self, user_id: u64) {
        let today = current_utc_day();
        let mut map = self.usage.lock().unwrap_or_else(|e| e.into_inner());
        let entry = map.entry(user_id).or_insert(DayUsage {
            day: today,
            count: 0,
        });
        if entry.day != today {
            entry.day = today;
            entry.count = 0;
        }
        entry.count += 1;
    }

    /// Get the number of generations the user has performed today.
    pub fn usage(&self, user_id: u64) -> u32 {
        self.usage_for_day(user_id, current_utc_day())
    }

    /// Reset a user's quota count to zero for today.
    pub fn reset(&self, user_id: u64) {
        let mut map = self.usage.lock().unwrap_or_else(|e| e.into_inner());
        map.remove(&user_id);
    }

    fn usage_for_day(&self, user_id: u64, day: u32) -> u32 {
        let map = self.usage.lock().unwrap_or_else(|e| e.into_inner());
        match map.get(&user_id) {
            Some(entry) if entry.day == day => entry.count,
            _ => 0,
        }
    }

    /// Record usage for a specific day (test helper).
    #[cfg(test)]
    fn record_for_day(&self, user_id: u64, day: u32) {
        let mut map = self.usage.lock().unwrap_or_else(|e| e.into_inner());
        let entry = map.entry(user_id).or_insert(DayUsage { day, count: 0 });
        if entry.day != day {
            entry.day = day;
            entry.count = 0;
        }
        entry.count += 1;
    }

    /// Check quota against a specific day (test helper).
    #[cfg(test)]
    fn check_for_day(&self, user_id: u64, max_daily: Option<u32>, day: u32) -> Option<u32> {
        let max = match max_daily {
            Some(m) => m,
            None => return Some(u32::MAX),
        };
        let used = self.usage_for_day(user_id, day);
        if used >= max {
            None
        } else {
            Some(max - used)
        }
    }
}

impl Default for QuotaTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Current UTC day as days-since-epoch.
fn current_utc_day() -> u32 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    (secs / 86400) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_when_unlimited() {
        let tracker = QuotaTracker::new();
        assert!(tracker.check(1, None).is_some());
        // Even after many records, unlimited always passes
        for _ in 0..100 {
            tracker.record(1);
        }
        assert!(tracker.check(1, None).is_some());
    }

    #[test]
    fn allows_within_quota() {
        let tracker = QuotaTracker::new();
        tracker.record(1);
        tracker.record(1);
        let result = tracker.check(1, Some(5));
        assert_eq!(result, Some(3));
    }

    #[test]
    fn blocks_when_exhausted() {
        let tracker = QuotaTracker::new();
        for _ in 0..5 {
            tracker.record(1);
        }
        assert_eq!(tracker.check(1, Some(5)), None);
    }

    #[test]
    fn resets_on_new_day() {
        let tracker = QuotaTracker::new();
        let yesterday = current_utc_day() - 1;
        let today = current_utc_day();

        // Record 5 uses "yesterday"
        for _ in 0..5 {
            tracker.record_for_day(1, yesterday);
        }
        // Yesterday's quota should be exhausted
        assert_eq!(tracker.check_for_day(1, Some(5), yesterday), None);
        // Today's quota should be fresh
        assert_eq!(tracker.check_for_day(1, Some(5), today), Some(5));
    }

    #[test]
    fn reset_clears_count() {
        let tracker = QuotaTracker::new();
        tracker.record(1);
        tracker.record(1);
        assert_eq!(tracker.usage(1), 2);
        tracker.reset(1);
        assert_eq!(tracker.usage(1), 0);
    }

    #[test]
    fn tracks_users_independently() {
        let tracker = QuotaTracker::new();
        for _ in 0..3 {
            tracker.record(1);
        }
        assert_eq!(tracker.usage(1), 3);
        assert_eq!(tracker.usage(2), 0);
        assert_eq!(tracker.check(2, Some(5)), Some(5));
    }

    #[test]
    fn usage_returns_zero_for_unknown_user() {
        let tracker = QuotaTracker::new();
        assert_eq!(tracker.usage(999), 0);
    }

    #[test]
    fn quota_of_zero_always_blocks() {
        let tracker = QuotaTracker::new();
        assert_eq!(tracker.check(1, Some(0)), None);
    }
}
