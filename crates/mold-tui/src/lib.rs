// Phase 1 skeleton — many types are defined for future phases.
#![allow(dead_code)]

mod action;
mod app;
mod backend;
mod event;
mod gallery_scan;
mod history;
mod model_info;
mod session;
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

/// Launch the mold TUI.
///
/// `host` overrides the default MOLD_HOST for remote generation.
/// `local` forces local-only inference (no server connection).
pub async fn run_tui(host: Option<String>, local: bool) -> Result<()> {
    // Probe the terminal image protocol *before* entering raw mode / alternate screen,
    // because the query writes to stdout and reads the terminal's reply.
    let picker = ratatui_image::picker::Picker::from_query_stdio()
        .unwrap_or_else(|_| ratatui_image::picker::Picker::from_fontsize((8, 16)));

    // Set up the terminal
    enable_raw_mode()?;
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

    // Create the app and run
    let version = mold_core::build_info::version_string();
    tracing::info!(%version, "starting mold tui");
    let mut app = App::new(host, local, picker)?;
    let result = run_event_loop(&mut terminal, &mut app).await;

    // Clean up background server process
    app.shutdown();

    // Restore the terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
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

        // Refresh resource info every 2 seconds
        if last_resource_refresh.elapsed() >= std::time::Duration::from_secs(2) {
            if app.should_poll_remote() {
                app.spawn_server_status_fetch();
            } else {
                app.resource_info.refresh_local();
            }
            last_resource_refresh = std::time::Instant::now();
        }

        if app.should_quit {
            return Ok(());
        }
    }
}
