//! Epoch-timestamp helpers shared by every crate that stamps outputs or DB rows.

use std::time::{SystemTime, UNIX_EPOCH};

/// Milliseconds since the UNIX epoch. Returns 0 if the clock reads pre-epoch.
pub fn now_epoch_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Milliseconds since the UNIX epoch as `u64` (filename builders). Saturates at 0.
pub fn now_epoch_ms_u64() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-01-01T00:00:00Z in ms — a floor that any correct clock exceeds.
    const JAN_2026_MS: i64 = 1_767_225_600_000;

    #[test]
    fn now_epoch_ms_is_after_2026() {
        assert!(now_epoch_ms() > JAN_2026_MS);
    }

    #[test]
    fn ms_variants_agree_within_a_second() {
        let a = now_epoch_ms();
        let b = now_epoch_ms_u64() as i64;
        assert!(
            (b - a).abs() < 1_000,
            "i64/u64 variants diverged: {a} vs {b}"
        );
    }
}
