//! Static web UI serving.
//!
//! The web gallery SPA is **baked into the binary** at compile time via
//! `rust-embed`. `build.rs` stages the `dist/` output of the Vue app (from
//! `$MOLD_WEB_DIST`, `<repo>/web/dist`, or a generated stub) into a directory
//! that `EmbeddedWeb` points at, so `mold serve` works with zero runtime
//! filesystem dependency.
//!
//! Runtime overrides still exist so a developer can iterate on the SPA
//! without recompiling Rust — resolution order:
//!
//! 1. `$MOLD_WEB_DIR` env var (filesystem)
//! 2. `$XDG_DATA_HOME/mold/web` (or `~/.local/share/mold/web`)
//! 3. `~/.mold/web`
//! 4. `<binary dir>/web` or `<binary>/../share/mold/web` (legacy layouts)
//! 5. `./web/dist` (for `cargo run` in the repo)
//! 6. **Embedded bundle** baked in at compile time (always present).
//!
//! When the embedded bundle is the placeholder stub (marker file
//! `__mold_placeholder` is present), the server serves the existing inline
//! "mold is running" HTML instead of the stub's empty index — this preserves
//! the UX of a bare `cargo build` checkout where no SPA was staged.
//!
//! The returned router is composed via `Router::fallback_service(...)` so
//! `/api/*` handlers take priority — only unmatched requests reach the SPA.

use axum::{
    body::Body,
    http::{header, HeaderMap, HeaderValue, Method, Request, Response, StatusCode, Uri},
    response::{Html, IntoResponse},
    routing::any_service,
    Router,
};
use rust_embed::RustEmbed;
use std::convert::Infallible;
use std::path::PathBuf;
use tower::service_fn;
use tower_http::services::{ServeDir, ServeFile};

const PLACEHOLDER_MARKER: &str = "__mold_placeholder";

#[derive(RustEmbed)]
#[folder = "$MOLD_EMBED_WEB_DIR"]
struct EmbeddedWeb;

const FALLBACK_INDEX_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>mold</title>
<style>
  :root { color-scheme: dark; }
  html, body { margin: 0; height: 100%; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
  body {
    background: radial-gradient(ellipse at top, #1e293b, #020617 60%);
    color: #e2e8f0;
    display: grid; place-items: center; padding: 2rem;
  }
  .card {
    max-width: 32rem; padding: 2rem 2.25rem; border-radius: 1.25rem;
    background: rgba(15,23,42,0.6); backdrop-filter: blur(16px);
    border: 1px solid rgba(148,163,184,0.15);
    box-shadow: 0 24px 60px -28px rgba(15,23,42,0.9);
  }
  h1 { margin: 0 0 0.5rem; font-size: 1.4rem; letter-spacing: -0.02em; }
  p { margin: 0.5rem 0; line-height: 1.55; color: #94a3b8; }
  a { color: #7dd3fc; text-decoration: none; }
  a:hover { color: #bae6fd; }
  code {
    font: 13px ui-monospace, SFMono-Regular, Menlo, monospace;
    background: rgba(56,189,248,0.08); color: #7dd3fc;
    padding: 0.1rem 0.4rem; border-radius: 0.35rem;
  }
</style>
</head>
<body>
<div class="card">
  <h1>mold is running</h1>
  <p>This binary was built without the web gallery UI bundled.</p>
  <p>Build the SPA with <code>cd web && bun run build</code> and rebuild mold, or point <code>MOLD_WEB_DIR</code> at an existing <code>dist/</code> to serve it without rebuilding.</p>
  <p>In the meantime: <a href="/api/docs">API docs</a> · <a href="/api/status">/api/status</a> · <a href="/api/gallery">/api/gallery</a></p>
</div>
</body>
</html>
"##;

/// Build a router whose fallback serves the bundled web UI (or a small
/// placeholder page when the embedded bundle is the build-time stub).
pub fn router() -> Router {
    if let Some(dir) = resolve_web_dir() {
        tracing::info!(path = %dir.display(), "serving web UI from filesystem override");
        let index = dir.join("index.html");
        // `ServeDir::fallback` makes missing paths fall through to the
        // SPA's index.html so client-side routes resolve cleanly.
        let service = ServeDir::new(&dir)
            .append_index_html_on_directories(true)
            .fallback(ServeFile::new(index));
        return Router::new().fallback_service(any_service(service));
    }

    if is_embed_stub() {
        tracing::info!("no web UI bundle found; serving inline placeholder");
        return Router::new().fallback(fallback_placeholder);
    }

    tracing::info!("serving embedded web UI bundle");
    // service_fn adapts our async closure into a Tower service; Router's
    // fallback_service threads unmatched requests through it.
    let service = service_fn(|req: Request<Body>| async move {
        let (parts, _body) = req.into_parts();
        Ok::<_, Infallible>(serve_embedded(&parts.method, &parts.uri, &parts.headers).await)
    });
    Router::new().fallback_service(any_service(service))
}

/// Serve a request from the embedded bundle. Falls back to `index.html` for
/// any path that does not match a real asset, so SPA client-side routes
/// (e.g. `/gallery/item/abc`) render the app shell. Matches `ServeDir`
/// semantics: only GET and HEAD are served; other methods get 405 with an
/// `Allow` header, and HEAD strips the body but keeps the headers.
async fn serve_embedded(method: &Method, uri: &Uri, headers: &HeaderMap) -> Response<Body> {
    if method != Method::GET && method != Method::HEAD {
        let mut resp = (StatusCode::METHOD_NOT_ALLOWED, "method not allowed").into_response();
        resp.headers_mut()
            .insert(header::ALLOW, HeaderValue::from_static("GET, HEAD"));
        return resp;
    }

    let raw = uri.path().trim_start_matches('/');
    let candidate = if raw.is_empty() { "index.html" } else { raw };

    if let Some(resp) = embedded_response(candidate, headers, method) {
        return resp;
    }

    // SPA deep-link fallback.
    embedded_response("index.html", headers, method).unwrap_or_else(|| {
        let mut resp = (StatusCode::NOT_FOUND, "not found").into_response();
        resp.headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        resp
    })
}

fn embedded_response(path: &str, headers: &HeaderMap, method: &Method) -> Option<Response<Body>> {
    let file = EmbeddedWeb::get(path)?;
    let mime = mime_guess::from_path(path).first_or_octet_stream();
    let etag_value = format!("\"{}\"", hex_short(&file.metadata.sha256_hash()));

    // Honor `If-None-Match` so conditional revalidations short-circuit with
    // 304 Not Modified — matches `ServeDir`'s behavior for the filesystem path.
    if let Some(inm) = headers.get(header::IF_NONE_MATCH) {
        if let Ok(inm_str) = inm.to_str() {
            if inm_str.split(',').any(|v| v.trim() == etag_value) {
                let mut resp = Response::new(Body::empty());
                *resp.status_mut() = StatusCode::NOT_MODIFIED;
                if let Ok(value) = HeaderValue::from_str(&etag_value) {
                    resp.headers_mut().insert(header::ETAG, value);
                }
                return Some(resp);
            }
        }
    }

    // HEAD responses share everything with GET except the body.
    let body = if method == Method::HEAD {
        Body::empty()
    } else {
        Body::from(file.data.into_owned())
    };

    let mut resp = Response::new(body);
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(mime.as_ref()).unwrap_or(HeaderValue::from_static("text/plain")),
    );
    // Hashed bundle assets (e.g. `assets/index-abc123.js`) are safe to cache
    // aggressively; everything else we leave to the client to revalidate.
    let cache = if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
    if let Ok(value) = HeaderValue::from_str(&etag_value) {
        resp.headers_mut().insert(header::ETAG, value);
    }
    Some(resp)
}

fn hex_short(hash: &[u8; 32]) -> String {
    let mut out = String::with_capacity(16);
    for byte in &hash[..8] {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{:02x}", byte);
    }
    out
}

fn is_embed_stub() -> bool {
    EmbeddedWeb::get(PLACEHOLDER_MARKER).is_some()
}

async fn fallback_placeholder() -> impl IntoResponse {
    let mut resp = Html(FALLBACK_INDEX_HTML).into_response();
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    *resp.status_mut() = StatusCode::OK;
    resp
}

fn resolve_web_dir() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(dir) = std::env::var("MOLD_WEB_DIR") {
        candidates.push(PathBuf::from(dir));
    }

    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        candidates.push(PathBuf::from(xdg).join("mold").join("web"));
    } else if let Ok(home) = std::env::var("HOME") {
        candidates.push(
            PathBuf::from(&home)
                .join(".local")
                .join("share")
                .join("mold")
                .join("web"),
        );
    }

    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(home).join(".mold").join("web"));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("web"));
            candidates.push(parent.join("..").join("share").join("mold").join("web"));
        }
    }

    candidates.push(PathBuf::from("web").join("dist"));

    candidates
        .into_iter()
        .find(|p| p.join("index.html").is_file())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use tower::ServiceExt;

    /// `MOLD_WEB_DIR` is process-global, so tests that read or mutate it
    /// must serialize. Cargo runs tests in parallel by default and a
    /// concurrent `set_var` in `filesystem_override_beats_embed` would
    /// otherwise leak into the other `router()` tests and make them flake.
    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// With nothing staged at build time, the binary should still serve the
    /// inline "mold is running" placeholder — confirms the stub plumbing.
    #[tokio::test]
    async fn router_handles_root_request() {
        let _guard = env_lock();
        let app = router();
        let resp = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert!(
            ct.starts_with("text/html"),
            "expected HTML content-type, got {ct}"
        );
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let text = String::from_utf8_lossy(&body);
        // Either the real SPA (has "<div id=\"app\"") or the inline
        // placeholder ("mold is running") — both are acceptable here.
        assert!(
            text.contains("mold is running") || text.contains("id=\"app\""),
            "unexpected root body: {}",
            &text[..text.len().min(200)]
        );
    }

    /// The compile-time embed is either the real SPA or the placeholder stub;
    /// `EmbeddedWeb::get("index.html")` must always succeed because `build.rs`
    /// guarantees at least the stub `index.html` is staged.
    #[test]
    fn embed_always_contains_index() {
        assert!(
            EmbeddedWeb::get("index.html").is_some(),
            "build.rs must stage an index.html (stub or real)"
        );
    }

    /// `MOLD_WEB_DIR` must win over the embedded bundle so developers can
    /// hot-reload the SPA without recompiling Rust.
    #[tokio::test]
    async fn filesystem_override_beats_embed() {
        let _guard = env_lock();
        let tmp = tempdir();
        std::fs::write(tmp.join("index.html"), b"<html>override</html>").unwrap();

        // Scope the env var to this test so parallel runs don't race.
        let prev = std::env::var_os("MOLD_WEB_DIR");
        // SAFETY: tests run serially when they need mutable env; set_var is
        // unsafe in Rust 2024 but single-thread-scoped here by design.
        unsafe {
            std::env::set_var("MOLD_WEB_DIR", &tmp);
        }
        let app = router();
        let resp = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        assert_eq!(&body[..], b"<html>override</html>");

        unsafe {
            match prev {
                Some(v) => std::env::set_var("MOLD_WEB_DIR", v),
                None => std::env::remove_var("MOLD_WEB_DIR"),
            }
        }
    }

    fn tempdir() -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "mold-web-ui-test-{}-{}",
            std::process::id(),
            rand_u64()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn rand_u64() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    /// Non-GET/HEAD requests against the embedded fallback must return 405
    /// with an `Allow: GET, HEAD` header — matching `ServeDir` semantics.
    /// Skipped when the filesystem override or the stub path is active
    /// because those code paths are exercised by different tests (and the
    /// stub path serves an inline placeholder for all methods by design).
    #[tokio::test]
    async fn embedded_rejects_non_get_head() {
        let _guard = env_lock();
        if resolve_web_dir().is_some() || is_embed_stub() {
            eprintln!("skipping: embedded bundle not active in this test build");
            return;
        }
        let app = router();
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
        let allow = resp
            .headers()
            .get(header::ALLOW)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default();
        assert!(
            allow.contains("GET") && allow.contains("HEAD"),
            "expected Allow: GET, HEAD, got {allow}"
        );
    }

    /// A conditional GET whose `If-None-Match` matches the asset's ETag must
    /// return 304 with an empty body. Skipped when the embedded bundle isn't
    /// the active path (filesystem override / stub).
    #[tokio::test]
    async fn embedded_conditional_get_returns_304() {
        let _guard = env_lock();
        if resolve_web_dir().is_some() || is_embed_stub() {
            eprintln!("skipping: embedded bundle not active in this test build");
            return;
        }
        let app = router();
        let first = app
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let etag = first
            .headers()
            .get(header::ETAG)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert!(!etag.is_empty(), "first response should set ETag");
        let second = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(header::IF_NONE_MATCH, &etag)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::NOT_MODIFIED);
        let body = to_bytes(second.into_body(), 64 * 1024).await.unwrap();
        assert!(body.is_empty(), "304 must have empty body");
    }
}
