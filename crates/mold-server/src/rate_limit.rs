use axum::{
    extract::{ConnectInfo, Request},
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use serde::Serialize;
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::warn;

type IpRateLimiter = RateLimiter<NotKeyed, InMemoryState, DefaultClock>;

/// Maximum number of per-IP limiter entries before eviction.
/// At ~200 bytes per entry, 10,000 entries ≈ 2 MB — bounded and safe.
pub(crate) const MAX_LIMITER_ENTRIES: usize = 10_000;

/// Per-IP rate limiter state. Each IP gets its own limiter instance.
/// Maps are bounded to [`MAX_LIMITER_ENTRIES`] — when full, the entire map
/// is cleared to reclaim memory (simple but effective against OOM from
/// IP-spoofing attacks; legitimate clients just get a fresh bucket).
pub struct RateLimitState {
    pub generation_quota: Quota,
    pub read_quota: Quota,
    pub(crate) generation_limiters: Mutex<HashMap<IpAddr, Arc<IpRateLimiter>>>,
    pub(crate) read_limiters: Mutex<HashMap<IpAddr, Arc<IpRateLimiter>>>,
}

impl RateLimitState {
    pub(crate) fn new(generation_quota: Quota, read_quota: Quota) -> Self {
        Self {
            generation_quota,
            read_quota,
            generation_limiters: Mutex::new(HashMap::new()),
            read_limiters: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn get_generation_limiter(&self, ip: IpAddr) -> Arc<IpRateLimiter> {
        let mut map = self.generation_limiters.lock().unwrap();
        if map.len() >= MAX_LIMITER_ENTRIES {
            warn!(
                entries = map.len(),
                "generation rate limiter map exceeded cap, evicting all entries"
            );
            map.clear();
        }
        map.entry(ip)
            .or_insert_with(|| Arc::new(RateLimiter::direct(self.generation_quota)))
            .clone()
    }

    fn get_read_limiter(&self, ip: IpAddr) -> Arc<IpRateLimiter> {
        let mut map = self.read_limiters.lock().unwrap();
        if map.len() >= MAX_LIMITER_ENTRIES {
            warn!(
                entries = map.len(),
                "read rate limiter map exceeded cap, evicting all entries"
            );
            map.clear();
        }
        map.entry(ip)
            .or_insert_with(|| Arc::new(RateLimiter::direct(self.read_quota)))
            .clone()
    }
}

/// Shared rate limit state — `None` means rate limiting is disabled.
pub type RateLimitConfig = Option<Arc<RateLimitState>>;

#[derive(Debug, Serialize)]
struct RateLimitError {
    error: String,
    code: String,
}

/// Parse the `MOLD_RATE_LIMIT` env var and build rate limiter state.
///
/// Format: `N/period` where period is `sec`, `min`, or `hour`.
/// Examples: `10/min`, `5/sec`, `100/hour`
///
/// Returns `None` when unset (rate limiting disabled).
pub fn load_rate_limit_config() -> anyhow::Result<RateLimitConfig> {
    let raw = match std::env::var("MOLD_RATE_LIMIT") {
        Ok(v) if !v.is_empty() => v,
        _ => return Ok(None),
    };

    let (count, period) = parse_rate_spec(&raw)?;

    let burst = match std::env::var("MOLD_RATE_LIMIT_BURST") {
        Ok(v) if !v.is_empty() => v
            .parse::<u32>()
            .map_err(|_| anyhow::anyhow!("MOLD_RATE_LIMIT_BURST must be a positive integer"))?,
        _ => (count * 2).min(100), // Default: 2x rate, capped at 100
    };

    let generation_quota = build_quota(count, burst, period)?;

    // Read endpoints get 10x the generation limit.
    let read_count = (count * 10).min(1000);
    let read_burst = (burst * 10).min(1000);
    let read_quota = build_quota(read_count, read_burst, period)?;

    tracing::info!(
        rate = %raw,
        burst,
        read_multiplier = 10,
        "rate limiting enabled"
    );

    Ok(Some(Arc::new(RateLimitState::new(
        generation_quota,
        read_quota,
    ))))
}

fn parse_rate_spec(spec: &str) -> anyhow::Result<(u32, Duration)> {
    let parts: Vec<&str> = spec.splitn(2, '/').collect();
    if parts.len() != 2 {
        anyhow::bail!("MOLD_RATE_LIMIT must be in the format N/period (e.g., 10/min)");
    }

    let count: u32 = parts[0]
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("MOLD_RATE_LIMIT count must be a positive integer"))?;
    if count == 0 {
        anyhow::bail!("MOLD_RATE_LIMIT count must be greater than 0");
    }

    let period = match parts[1].trim() {
        "sec" | "second" | "s" => Duration::from_secs(1),
        "min" | "minute" | "m" => Duration::from_secs(60),
        "hour" | "hr" | "h" => Duration::from_secs(3600),
        other => anyhow::bail!("unknown MOLD_RATE_LIMIT period: {other} (use sec, min, or hour)"),
    };

    Ok((count, period))
}

fn build_quota(count: u32, burst: u32, period: Duration) -> anyhow::Result<Quota> {
    let replenish_interval = period / count;
    let burst_nz = NonZeroU32::new(burst)
        .ok_or_else(|| anyhow::anyhow!("rate limit burst must be greater than 0"))?;
    Ok(Quota::with_period(replenish_interval)
        .ok_or_else(|| anyhow::anyhow!("invalid rate limit period"))?
        .allow_burst(burst_nz))
}

/// Route tier for rate limiting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteTier {
    /// Expensive operations: generate, expand, model load/pull/unload.
    Generation,
    /// Cheap reads: list models, status, gallery.
    Read,
}

/// Paths and their rate limit tiers.
pub fn classify_route(path: &str, method: &axum::http::Method) -> Option<RouteTier> {
    // Health/docs are not rate limited.
    if path == "/health" || path == "/api/docs" || path == "/api/openapi.json" {
        return None;
    }

    match (method.as_str(), path) {
        ("POST", "/api/generate" | "/api/generate/stream" | "/api/expand") => {
            Some(RouteTier::Generation)
        }
        ("POST", "/api/models/load" | "/api/models/pull") => Some(RouteTier::Generation),
        ("DELETE", "/api/models/unload") => Some(RouteTier::Generation),
        ("DELETE", _) if path.starts_with("/api/gallery/") => Some(RouteTier::Generation),
        ("GET", _) => Some(RouteTier::Read),
        _ => Some(RouteTier::Read), // Default unknown routes to read tier
    }
}

/// Axum middleware that enforces per-IP rate limiting.
pub async fn rate_limit_middleware(
    connect_info: ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    let rl_config = request.extensions().get::<RateLimitConfig>().cloned();

    let state = match rl_config.as_ref().and_then(|c| c.as_ref()) {
        Some(s) => s,
        None => return next.run(request).await, // Rate limiting disabled
    };

    let ip = connect_info.0.ip();
    let tier = classify_route(request.uri().path(), request.method());

    let tier = match tier {
        Some(t) => t,
        None => return next.run(request).await, // Exempt path
    };

    let limiter = match tier {
        RouteTier::Generation => state.get_generation_limiter(ip),
        RouteTier::Read => state.get_read_limiter(ip),
    };

    match limiter.check() {
        Ok(_) => next.run(request).await,
        Err(not_until) => {
            let retry_after = not_until.wait_time_from(governor::clock::Clock::now(
                &governor::clock::DefaultClock::default(),
            ));
            let retry_secs = retry_after.as_secs().max(1);
            warn!(
                ip = %ip,
                retry_after_secs = retry_secs,
                tier = ?match tier { RouteTier::Generation => "generation", RouteTier::Read => "read" },
                "rate limit exceeded"
            );
            rate_limited_response(retry_secs)
        }
    }
}

fn rate_limited_response(retry_after_secs: u64) -> Response {
    let body = RateLimitError {
        error: "rate limit exceeded".to_string(),
        code: "RATE_LIMITED".to_string(),
    };
    let mut response = (StatusCode::TOO_MANY_REQUESTS, Json(body)).into_response();
    if let Ok(val) = HeaderValue::from_str(&retry_after_secs.to_string()) {
        response.headers_mut().insert("retry-after", val);
    }
    response
}

/// Injects the `RateLimitConfig` as a request extension.
pub async fn inject_rate_limit_state(
    axum::extract::State(rl): axum::extract::State<RateLimitConfig>,
    mut request: Request,
    next: Next,
) -> Response {
    request.extensions_mut().insert(rl);
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rate_per_minute() {
        let (count, period) = parse_rate_spec("10/min").unwrap();
        assert_eq!(count, 10);
        assert_eq!(period, Duration::from_secs(60));
    }

    #[test]
    fn parse_rate_per_second() {
        let (count, period) = parse_rate_spec("5/sec").unwrap();
        assert_eq!(count, 5);
        assert_eq!(period, Duration::from_secs(1));
    }

    #[test]
    fn parse_rate_per_hour() {
        let (count, period) = parse_rate_spec("100/hour").unwrap();
        assert_eq!(count, 100);
        assert_eq!(period, Duration::from_secs(3600));
    }

    #[test]
    fn parse_rate_short_aliases() {
        assert!(parse_rate_spec("1/s").is_ok());
        assert!(parse_rate_spec("1/m").is_ok());
        assert!(parse_rate_spec("1/h").is_ok());
    }

    #[test]
    fn parse_rate_invalid_format() {
        assert!(parse_rate_spec("10").is_err());
        assert!(parse_rate_spec("10/xyz").is_err());
        assert!(parse_rate_spec("0/min").is_err());
        assert!(parse_rate_spec("abc/min").is_err());
    }

    #[test]
    fn classify_generation_routes() {
        use axum::http::Method;
        assert_eq!(
            classify_route("/api/generate", &Method::POST),
            Some(RouteTier::Generation)
        );
        assert_eq!(
            classify_route("/api/generate/stream", &Method::POST),
            Some(RouteTier::Generation)
        );
        assert_eq!(
            classify_route("/api/expand", &Method::POST),
            Some(RouteTier::Generation)
        );
        assert_eq!(
            classify_route("/api/models/load", &Method::POST),
            Some(RouteTier::Generation)
        );
        assert_eq!(
            classify_route("/api/models/pull", &Method::POST),
            Some(RouteTier::Generation)
        );
        assert_eq!(
            classify_route("/api/models/unload", &Method::DELETE),
            Some(RouteTier::Generation)
        );
    }

    #[test]
    fn classify_read_routes() {
        use axum::http::Method;
        assert_eq!(
            classify_route("/api/models", &Method::GET),
            Some(RouteTier::Read)
        );
        assert_eq!(
            classify_route("/api/status", &Method::GET),
            Some(RouteTier::Read)
        );
        assert_eq!(
            classify_route("/api/gallery", &Method::GET),
            Some(RouteTier::Read)
        );
    }

    #[test]
    fn classify_exempt_routes() {
        use axum::http::Method;
        assert_eq!(classify_route("/health", &Method::GET), None);
        assert_eq!(classify_route("/api/docs", &Method::GET), None);
        assert_eq!(classify_route("/api/openapi.json", &Method::GET), None);
    }

    #[test]
    fn build_quota_valid() {
        let q = build_quota(10, 20, Duration::from_secs(60));
        assert!(q.is_ok());
    }

    #[test]
    fn rate_limiter_rejects_after_burst() {
        let quota = build_quota(1, 2, Duration::from_secs(60)).unwrap();
        let limiter = RateLimiter::direct(quota);
        assert!(limiter.check().is_ok()); // 1st
        assert!(limiter.check().is_ok()); // 2nd (burst)
        assert!(limiter.check().is_err()); // 3rd → rejected
    }
}
