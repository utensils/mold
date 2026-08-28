// TUI shell — some types are reserved for planned features.
#![allow(dead_code)]

mod action;
mod animation;
mod app;
mod backend;
mod event;
mod gallery_scan;
mod gallery_trash;
mod h3_references;
mod history;
mod hosts;
mod identity;
mod model_info;
mod motion;
mod palette;
mod prefs;
mod session;
#[cfg(test)]
pub(crate) mod test_env;
mod thumbnails;
mod ui;

use std::io;
use std::panic;

use anyhow::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;

use app::App;

#[derive(Default)]
struct TerminalRestoreGuard {
    active: bool,
}

impl TerminalRestoreGuard {
    fn arm(&mut self) {
        self.active = true;
    }

    fn restore(&mut self) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }
        disable_raw_mode()?;
        execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
        self.active = false;
        Ok(())
    }
}

impl Drop for TerminalRestoreGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

/// Initial workspace for callers that launch the TUI as a focused command.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TuiInitialWorkspace {
    #[default]
    Create,
    Library,
}

/// Launch policy for embedding the TUI behind another CLI command.
#[derive(Debug, Clone, Default)]
pub struct TuiLaunchOptions {
    pub host: Option<String>,
    pub local: bool,
    /// Ephemeral credential for the selected launch host. It is never saved
    /// to the TUI database.
    pub api_key: Option<String>,
    pub initial_workspace: TuiInitialWorkspace,
    /// Refuse an unreachable `host` / `MOLD_HOST` instead of silently
    /// changing the data authority to local files.
    pub strict_host: bool,
}

/// Launch the mold TUI.
///
/// `host` overrides the default MOLD_HOST for remote generation.
/// `local` forces local-only inference (no server connection).
pub async fn run_tui(host: Option<String>, local: bool) -> Result<()> {
    run_tui_with_options(TuiLaunchOptions {
        host,
        local,
        ..Default::default()
    })
    .await
}

/// Launch the TUI with an explicit workspace and host-fallback policy.
pub async fn run_tui_with_options(options: TuiLaunchOptions) -> Result<()> {
    // Probe the terminal image protocol *before* entering raw mode / alternate screen,
    // because the query writes to stdout and reads the terminal's reply.
    let picker = ratatui_image::picker::Picker::from_query_stdio()
        .unwrap_or_else(|_| ratatui_image::picker::Picker::from_fontsize((8, 16)));

    // Resolve the requested data authority before raw mode. In particular, a
    // strict unreachable host must return a normal shell error without leaving
    // the caller's terminal in the alternate screen.
    let version = mold_core::build_info::version_string();
    tracing::info!(%version, "starting mold tui");
    let api_key = options
        .api_key
        .or_else(|| std::env::var("MOLD_API_KEY").ok());
    let mut app = App::new_with_launch_policy(
        options.host,
        options.local,
        picker,
        api_key,
        options.strict_host,
    )?;
    if options.initial_workspace == TuiInitialWorkspace::Library {
        app.open_library();
    }

    // Set up the terminal
    let mut restore_guard = TerminalRestoreGuard::default();
    enable_raw_mode()?;
    restore_guard.arm();
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Install a panic hook that restores the terminal before printing the panic
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(move |panic_info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
        original_hook(panic_info);
    }));

    let result = run_event_loop(&mut terminal, &mut app).await;

    // Clean up background server process
    app.shutdown();

    // Restore the terminal
    restore_guard.restore()?;
    terminal.show_cursor()?;

    result
}

async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    let mut last_resource_refresh = std::time::Instant::now();
    // Initial resource info refresh
    if app.should_poll_remote() {
        app.spawn_server_status_fetch();
    } else {
        app.resource_info.refresh_local();
    }

    loop {
        terminal.draw(|frame| ui::render(frame, app))?;

        // Poll for crossterm events with a short timeout (~60fps)
        if crossterm::event::poll(std::time::Duration::from_millis(16))? {
            let event = crossterm::event::read()?;
            app.handle_crossterm_event(event);
        }

        // Process any background task results
        app.process_background_events();

        // Advance any animated previews so the next draw shows the next frame
        // when its delay has elapsed.
        app.tick_animations();

        // Refresh resource info every 2 seconds
        if last_resource_refresh.elapsed() >= std::time::Duration::from_secs(2) {
            if app.should_poll_remote() {
                app.spawn_server_status_fetch();
            } else {
                app.resource_info.refresh_local();
            }
            // Multi-host telemetry: all registered hosts while Machines
            // is active, only the generation-target host otherwise.
            app.tick_host_polling();
            last_resource_refresh = std::time::Instant::now();
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod launch_tests {
    use super::*;
    use serial_test::serial;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn strict_library_launch_refuses_unreachable_explicit_host() {
        let picker = ratatui_image::picker::Picker::from_fontsize((8, 16));
        let error = match App::new_with_launch_policy(
            Some("http://127.0.0.1:1".to_string()),
            false,
            picker,
            None,
            true,
        ) {
            Ok(_) => panic!("strict launch must not fall back to local files"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("did not fall back to local files"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[serial(mold_env)]
    async fn strict_library_launch_authenticates_the_gallery_before_opening() {
        crate::test_env::disable_db_for_non_isolated_tests();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/gallery"))
            .and(header("x-api-key", "library-secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/models"))
            .and(header("x-api-key", "library-secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
            .mount(&server)
            .await;

        let picker = ratatui_image::picker::Picker::from_fontsize((8, 16));
        let app = App::new_with_launch_policy(
            Some(server.uri()),
            false,
            picker,
            Some("library-secret".to_string()),
            true,
        )
        .expect("the strict launch should accept the authenticated gallery");

        let requests = server.received_requests().await.expect("request history");
        assert!(requests.iter().any(|request| {
            request.url.path() == "/api/gallery"
                && request
                    .headers
                    .get("x-api-key")
                    .is_some_and(|value| value == "library-secret")
        }));
        drop(app);
    }
}
