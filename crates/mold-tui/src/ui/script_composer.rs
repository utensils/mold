use mold_core::{
    ChainScript, ChainScriptChain, ChainStage, OutputFormat, TransitionMode,
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};

use crate::ui::theme::Theme;

pub struct ScriptComposerState {
    pub script: ChainScript,
    pub selected: usize,
    pub unsaved: bool,
}

impl ScriptComposerState {
    pub fn new(script: ChainScript) -> Self {
        Self {
            script,
            selected: 0,
            unsaved: false,
        }
    }

    pub fn selected_stage(&self) -> Option<&ChainStage> {
        self.script.stages.get(self.selected)
    }

    pub fn selected_stage_mut(&mut self) -> Option<&mut ChainStage> {
        self.script.stages.get_mut(self.selected)
    }

    pub fn move_down(&mut self) {
        if self.selected + 1 < self.script.stages.len() {
            self.selected += 1;
        }
    }

    pub fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    pub fn reorder_down(&mut self) {
        let n = self.script.stages.len();
        if self.selected + 1 < n {
            self.script.stages.swap(self.selected, self.selected + 1);
            self.selected += 1;
            self.unsaved = true;
        }
    }

    pub fn reorder_up(&mut self) {
        if self.selected > 0 {
            self.script.stages.swap(self.selected - 1, self.selected);
            self.selected -= 1;
            self.unsaved = true;
        }
    }
}

impl Default for ScriptComposerState {
    fn default() -> Self {
        Self::new(ChainScript {
            schema: "mold.chain.v1".into(),
            chain: ChainScriptChain {
                model: "ltx-2-19b-distilled:fp8".into(),
                width: 1216,
                height: 704,
                fps: 24,
                seed: None,
                steps: 8,
                guidance: 3.0,
                strength: 1.0,
                motion_tail_frames: 25,
                output_format: OutputFormat::Mp4,
            },
            stages: vec![ChainStage {
                prompt: String::new(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            }],
        })
    }
}

fn truncate(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().count() <= max {
        return s.to_string();
    }
    if max == 1 {
        return "...".chars().next().unwrap().to_string();
    }
    let cut = max - 3;
    let head: String = s.chars().take(cut).collect();
    format!("{head}...")
}

fn transition_label(t: TransitionMode) -> &'static str {
    match t {
        TransitionMode::Smooth => "smooth",
        TransitionMode::Cut => "cut",
        TransitionMode::Fade => "fade",
    }
}

pub fn render(
    frame: &mut Frame,
    state: &ScriptComposerState,
    area: Rect,
    theme: &Theme,
) {
    let [stage_area, editor_area, footer_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(8),
            Constraint::Length(3),
        ])
        .areas(area);

    render_stage_list(frame, state, stage_area, theme);
    render_editor(frame, state, editor_area, theme);
    render_footer(frame, state, footer_area, theme);
}

fn render_stage_list(
    frame: &mut Frame,
    state: &ScriptComposerState,
    area: Rect,
    theme: &Theme,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border_focused())
        .title(" Stages ")
        .title_style(theme.title_focused())
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let prompt_width = inner.width.saturating_sub(22) as usize;

    let rows: Vec<Row> = state
        .script
        .stages
        .iter()
        .enumerate()
        .map(|(i, stage)| {
            let marker = if i == state.selected { ">" } else { " " };
            let prompt_preview = if stage.prompt.is_empty() {
                "(empty)".to_string()
            } else {
                truncate(&stage.prompt, prompt_width)
            };
            let transition = transition_label(stage.transition);
            let style = if i == state.selected {
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme.text)
            };
            Row::new(vec![
                marker.to_string(),
                format!("{}", i + 1),
                transition.to_string(),
                format!("{}f", stage.frames),
                prompt_preview,
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Length(1),
        Constraint::Length(3),
        Constraint::Length(6),
        Constraint::Length(5),
        Constraint::Min(10),
    ];

    let table = Table::new(rows, widths).column_spacing(1);
    frame.render_widget(table, inner);
}

fn render_editor(
    frame: &mut Frame,
    state: &ScriptComposerState,
    area: Rect,
    theme: &Theme,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(" Details ")
        .title_style(theme.title())
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let lines = if let Some(stage) = state.selected_stage() {
        let prompt_display = if stage.prompt.is_empty() {
            "(empty)".to_string()
        } else {
            truncate(&stage.prompt, inner.width.saturating_sub(12) as usize)
        };
        let img_status = if stage.source_image.is_some() {
            "[img]"
        } else {
            "none"
        };
        let transition = transition_label(stage.transition);
        let fade_info = if stage.transition == TransitionMode::Fade {
            stage
                .fade_frames
                .map(|f| format!(" ({f}f)"))
                .unwrap_or_else(|| " (default)".to_string())
        } else {
            String::new()
        };

        vec![
            Line::from(vec![
                Span::styled("  Prompt:     ", Style::default().fg(theme.text_dim)),
                Span::styled(prompt_display, Style::default().fg(theme.text)),
            ]),
            Line::from(vec![
                Span::styled("  Transition: ", Style::default().fg(theme.text_dim)),
                Span::styled(
                    format!("{transition}{fade_info}"),
                    Style::default().fg(theme.text),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Frames:     ", Style::default().fg(theme.text_dim)),
                Span::styled(
                    format!("{}", stage.frames),
                    Style::default().fg(theme.text),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Source:     ", Style::default().fg(theme.text_dim)),
                Span::styled(img_status, Style::default().fg(theme.text)),
            ]),
            Line::from(vec![
                Span::styled("  Neg prompt: ", Style::default().fg(theme.text_dim)),
                Span::styled(
                    stage
                        .negative_prompt
                        .as_deref()
                        .unwrap_or("none")
                        .to_string(),
                    Style::default().fg(theme.text_dim),
                ),
            ]),
        ]
    } else {
        vec![Line::from(Span::styled(
            "  No stages",
            Style::default().fg(theme.text_dim),
        ))]
    };

    let para = Paragraph::new(lines);
    frame.render_widget(para, inner);
}

fn render_footer(
    frame: &mut Frame,
    state: &ScriptComposerState,
    area: Rect,
    theme: &Theme,
) {
    let chain = &state.script.chain;
    let stage_count = state.script.stages.len();
    let total_frames: u32 = state.script.stages.iter().map(|s| s.frames).sum();
    let duration_secs = if chain.fps > 0 {
        total_frames as f64 / chain.fps as f64
    } else {
        0.0
    };

    let unsaved_mark = if state.unsaved { " *" } else { "" };

    let summary = format!(
        " {} stage{}, {} frames, {:.1}s | {} {}x{} {}fps{}",
        stage_count,
        if stage_count == 1 { "" } else { "s" },
        total_frames,
        duration_secs,
        chain.model,
        chain.width,
        chain.height,
        chain.fps,
        unsaved_mark,
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border())
        .title(" Summary ")
        .title_style(theme.title())
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let para = Paragraph::new(Line::from(Span::styled(
        summary,
        Style::default().fg(theme.text),
    )));
    frame.render_widget(para, inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_has_one_stage() {
        let s = ScriptComposerState::default();
        assert_eq!(s.script.stages.len(), 1);
        assert_eq!(s.selected, 0);
        assert!(!s.unsaved);
    }

    #[test]
    fn selected_stage_returns_correct_stage() {
        let mut s = ScriptComposerState::default();
        s.script.stages.push(ChainStage {
            prompt: "second".into(),
            frames: 49,
            source_image: None,
            negative_prompt: None,
            seed_offset: None,
            transition: TransitionMode::Cut,
            fade_frames: None,
            model: None,
            loras: vec![],
            references: vec![],
        });
        s.selected = 1;
        assert_eq!(s.selected_stage().unwrap().prompt, "second");
    }

    fn make_state(n: usize) -> ScriptComposerState {
        let mut s = ScriptComposerState::default();
        // default already has 1 stage, add n-1 more
        for i in 1..n {
            s.script.stages.push(ChainStage {
                prompt: format!("stage {i}"),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            });
        }
        s
    }

    #[test]
    fn move_down_advances_selection() {
        let mut s = make_state(3);
        s.move_down();
        assert_eq!(s.selected, 1);
    }

    #[test]
    fn move_down_clamps_at_bottom() {
        let mut s = make_state(3);
        s.selected = 2;
        s.move_down();
        assert_eq!(s.selected, 2);
    }

    #[test]
    fn move_up_retreats_selection() {
        let mut s = make_state(3);
        s.selected = 2;
        s.move_up();
        assert_eq!(s.selected, 1);
    }

    #[test]
    fn move_up_clamps_at_top() {
        let mut s = make_state(3);
        s.move_up();
        assert_eq!(s.selected, 0);
    }

    #[test]
    fn reorder_down_swaps_and_follows() {
        let mut s = make_state(3);
        // default stage prompt is empty, stage 1 prompt is "stage 1"
        s.reorder_down();
        assert_eq!(s.selected, 1);
        assert_eq!(s.script.stages[1].prompt, ""); // original stage 0 moved to 1
        assert_eq!(s.script.stages[0].prompt, "stage 1"); // original stage 1 moved to 0
        assert!(s.unsaved);
    }

    #[test]
    fn reorder_down_clamps_at_bottom() {
        let mut s = make_state(2);
        s.selected = 1;
        s.reorder_down();
        assert_eq!(s.selected, 1);
    }

    #[test]
    fn reorder_up_swaps_and_follows() {
        let mut s = make_state(3);
        s.selected = 1;
        let p1 = s.script.stages[1].prompt.clone();
        s.reorder_up();
        assert_eq!(s.selected, 0);
        assert_eq!(s.script.stages[0].prompt, p1);
        assert!(s.unsaved);
    }

    #[test]
    fn reorder_up_clamps_at_top() {
        let mut s = make_state(2);
        s.reorder_up();
        assert_eq!(s.selected, 0);
        assert!(!s.unsaved);
    }
}
