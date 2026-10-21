//! Unsupported builds expose no admission route and cannot strand a parent.
pub(crate) fn router() -> axum::Router<crate::state::AppState> {
    axum::Router::new()
}
pub(crate) fn now_ms() -> i64 {
    i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    )
    .unwrap_or(i64::MAX)
}
