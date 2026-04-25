//! Politeness layer for the scanner stages: pre-request sleep + 429 retry
//! with `Retry-After` parsing.
//!
//! A `mold catalog refresh` makes thousands of requests against HF and
//! Civitai. Without throttling the scanner fires them as fast as the
//! sockets allow, which is the easiest way to get IP-banned. The two
//! knobs here are deliberately conservative defaults:
//!
//! - **HF**: ~250 ms between requests (≈4 req/s, well under the
//!   anonymous `/api/models` rate ceiling).
//! - **Civitai**: ~1500 ms between requests (≈40 req/min, just inside
//!   the documented anonymous limit).
//!
//! On a 429, the helper honours the server's `Retry-After` header when
//! present (delta-seconds form) and otherwise falls back to an
//! exponential backoff seeded by `default_429_backoff`. After
//! `max_429_retries` exhausted retries the caller sees
//! `ScanError::RateLimited`, exactly as before.

use std::time::Duration;

use reqwest::header::{HeaderMap, RETRY_AFTER};
use reqwest::{Request, Response, StatusCode};

use crate::scanner::ScanError;

/// Outcome of one retry-aware request: either the body bytes (already
/// drained) or a `ScanError` derived from the response status.
pub struct RequestOutcome {
    pub body: String,
}

/// Sleep helper used by the stages before each request. Lifted out so a
/// future test can swap in `tokio::time::pause()` for deterministic
/// scheduling — production callers always hit the real clock.
pub async fn pre_request_delay(delay: Duration) {
    if delay.is_zero() {
        return;
    }
    tokio::time::sleep(delay).await;
}

/// Wait based on a 429 response. Prefers the server's `Retry-After`
/// (delta-seconds form, per RFC 7231 §7.1.3). HTTP-date form is
/// uncommon for rate limit replies and is treated as missing — we
/// fall back to the exponential default.
pub async fn after_429_delay(headers: &HeaderMap, default: Duration, attempt: u8) {
    let from_header = headers
        .get(RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs);

    let wait = from_header.unwrap_or_else(|| {
        // Exponential backoff: default × 2^attempt. attempt counts from 0,
        // so 5s → 10s → 20s with the default.
        let factor = 1u64 << attempt.min(6);
        default.saturating_mul(factor as u32)
    });
    if !wait.is_zero() {
        tokio::time::sleep(wait).await;
    }
}

/// Translate a non-2xx response into the appropriate `ScanError`.
/// Stages call this for the *final* attempt only — earlier 429s are
/// retried by the caller via [`after_429_delay`].
pub fn classify_status(status: StatusCode, host: &'static str) -> Option<ScanError> {
    if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
        return Some(ScanError::AuthRequired { host });
    }
    if status == StatusCode::TOO_MANY_REQUESTS {
        return Some(ScanError::RateLimited { host });
    }
    if !status.is_success() {
        // Bubble non-success up via reqwest's error_for_status semantics so
        // the orchestrator can record it as a NetworkError.
        // Note: stages currently propagate via `reqwest::Error`; classify_status
        // is only consulted on the explicit auth/rate-limit branches.
        None
    } else {
        None
    }
}

/// Send `req` with the politeness layer applied: pre-request sleep,
/// then up to `max_retries` retries on HTTP 429 with `Retry-After`-aware
/// backoff. The first attempt uses the configured `pre_delay`; retries
/// use `after_429_delay`.
///
/// Note: `req.try_clone()` is used to clone the request between
/// retries. For the GET requests the scanner uses this is always
/// `Some(_)`.
pub async fn polite_send(
    client: &reqwest::Client,
    req: Request,
    pre_delay: Duration,
    default_backoff: Duration,
    max_retries: u8,
    host: &'static str,
) -> Result<RequestOutcome, ScanError> {
    pre_request_delay(pre_delay).await;

    let mut attempt: u8 = 0;
    loop {
        let cloned = match req.try_clone() {
            Some(c) => c,
            None => {
                // Should never trigger for the GET requests we make, but
                // fail loudly if a future caller introduces a streaming
                // body.
                return Err(ScanError::Network(
                    client
                        .execute(req)
                        .await
                        .err()
                        .unwrap_or_else(|| panic!("non-cloneable request")),
                ));
            }
        };
        let resp: Response = client.execute(cloned).await?;
        let status = resp.status();

        if status == StatusCode::TOO_MANY_REQUESTS && attempt < max_retries {
            after_429_delay(resp.headers(), default_backoff, attempt).await;
            attempt += 1;
            continue;
        }

        if let Some(err) = classify_status(status, host) {
            return Err(err);
        }
        if !status.is_success() {
            // Convert non-success into a Network error via reqwest's
            // error_for_status helper for a consistent error chain.
            return Err(resp.error_for_status().unwrap_err().into());
        }

        let body = resp.text().await?;
        return Ok(RequestOutcome { body });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};
    use std::time::Instant;

    #[tokio::test]
    async fn after_429_uses_retry_after_when_present() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("0"));
        let start = Instant::now();
        after_429_delay(&headers, Duration::from_secs(5), 0).await;
        // Retry-After=0 should not stall us beyond a few ms.
        assert!(
            start.elapsed() < Duration::from_millis(50),
            "honoured Retry-After=0: {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn after_429_falls_back_to_default_when_header_missing() {
        let headers = HeaderMap::new();
        // Use Duration::from_millis(1) so the test stays fast.
        let start = Instant::now();
        after_429_delay(&headers, Duration::from_millis(1), 0).await;
        // attempt=0 with 1ms default = 1ms. Allow generous slack.
        assert!(start.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn after_429_doubles_default_per_attempt() {
        let headers = HeaderMap::new();
        // attempt=2 → 1ms × 4 = 4ms. Just verify it doesn't underflow or
        // skip the sleep entirely.
        let start = Instant::now();
        after_429_delay(&headers, Duration::from_millis(1), 2).await;
        assert!(start.elapsed() >= Duration::from_millis(2));
    }

    #[test]
    fn classify_status_maps_auth_and_rate_limit() {
        assert!(matches!(
            classify_status(StatusCode::UNAUTHORIZED, "h"),
            Some(ScanError::AuthRequired { host: "h" })
        ));
        assert!(matches!(
            classify_status(StatusCode::FORBIDDEN, "h"),
            Some(ScanError::AuthRequired { host: "h" })
        ));
        assert!(matches!(
            classify_status(StatusCode::TOO_MANY_REQUESTS, "h"),
            Some(ScanError::RateLimited { host: "h" })
        ));
        assert!(classify_status(StatusCode::OK, "h").is_none());
        assert!(classify_status(StatusCode::INTERNAL_SERVER_ERROR, "h").is_none());
    }
}
