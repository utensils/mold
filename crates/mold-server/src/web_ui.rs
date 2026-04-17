//! Static web UI serving.
//!
//! Locates the built SPA (`index.html` + `assets/`) at one of a short list of
//! conventional paths and returns an axum fallback service that serves the
//! assets directly. SPA deep links (e.g. `/gallery/item/abc`) fall back to
//! `index.html` so the client router can resolve them.
//!
//! Resolution order (first hit wins):
//! 1. `$MOLD_WEB_DIR` env var
//! 2. `$XDG_DATA_HOME/mold/web` (or `~/.local/share/mold/web`)
//! 3. `~/.mold/web`
//! 4. `<binary dir>/web` (for nix / portable installs)
//! 5. `./web/dist` (for `cargo run` in the repo)
//!
//! When nothing is found, we return a small inline HTML page pointing users
//! to `/api/docs`, so the server is still useful without a built UI.
//!
//! The returned router is composed via `Router::fallback_service(...)` so
//! `/api/*` handlers take priority — only unmatched requests reach the SPA.

use axum::{
    http::{header, HeaderValue, StatusCode},
    response::{Html, IntoResponse},
    routing::any_service,
    Router,
};
use std::path::PathBuf;
use tower_http::services::{ServeDir, ServeFile};

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
  <p>The web gallery UI isn't installed on this server yet.</p>
  <p>Build the SPA with <code>cd web && bun install && bun run build</code>, then either set <code>MOLD_WEB_DIR</code> or copy <code>web/dist/</code> to <code>~/.mold/web</code>.</p>
  <p>In the meantime: <a href="/api/docs">API docs</a> · <a href="/api/status">/api/status</a> · <a href="/api/gallery">/api/gallery</a></p>
</div>
</body>
</html>
"##;

/// Build a router whose fallback serves the bundled web UI (or a small
/// placeholder page when no UI has been built).
pub fn router() -> Router {
    match resolve_web_dir() {
        Some(dir) => {
            tracing::info!(path = %dir.display(), "serving web UI");
            let index = dir.join("index.html");
            // `ServeDir::fallback` makes missing paths fall through to the
            // SPA's index.html so client-side routes resolve cleanly.
            let service = ServeDir::new(&dir)
                .append_index_html_on_directories(true)
                .fallback(ServeFile::new(index));
            Router::new().fallback_service(any_service(service))
        }
        None => {
            tracing::info!("no web UI bundle found; serving inline placeholder");
            Router::new().fallback(fallback_placeholder)
        }
    }
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
