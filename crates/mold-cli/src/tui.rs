use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};
use std::io::stdout;

struct App {
    model: String,
    prompt: String,
    status: String,
    should_quit: bool,
}

impl App {
    fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            prompt: String::new(),
            status: "Ready — type a prompt and press Enter to generate".to_string(),
            should_quit: false,
        }
    }
}

pub async fn run(model: &str) -> Result<()> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;

    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    let mut app = App::new(model);

    while !app.should_quit {
        terminal.draw(|frame| render(&app, frame))?;

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') if app.prompt.is_empty() => {
                            app.should_quit = true;
                        }
                        KeyCode::Esc => {
                            app.should_quit = true;
                        }
                        KeyCode::Char(c) => {
                            app.prompt.push(c);
                        }
                        KeyCode::Backspace => {
                            app.prompt.pop();
                        }
                        KeyCode::Enter => {
                            if !app.prompt.is_empty() {
                                app.status = format!(
                                    "Generating: \"{}\" with {} (stub)",
                                    app.prompt, app.model
                                );
                                app.prompt.clear();
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}

fn render(app: &App, frame: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(frame.area());

    // Header
    let header = Paragraph::new(format!(" mold — model: {}", app.model))
        .style(Style::default().fg(Color::Cyan).bold())
        .block(Block::default().borders(Borders::BOTTOM));
    frame.render_widget(header, chunks[0]);

    // Status area
    let status = Paragraph::new(app.status.as_str()).block(
        Block::default()
            .title(" Status ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(status, chunks[1]);

    // Prompt input
    let input = Paragraph::new(format!("❯ {}", app.prompt)).block(
        Block::default()
            .title(" Prompt ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green)),
    );
    frame.render_widget(input, chunks[2]);
}
