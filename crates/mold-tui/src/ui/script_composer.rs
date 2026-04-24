use mold_core::{ChainScript, ChainScriptChain, ChainStage, OutputFormat, TransitionMode};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table};

use crate::ui::theme::Theme;

#[derive(Default)]
pub enum ScriptModal {
    #[default]
    Closed,
    PromptEdit {
        buffer: String,
    },
    FramesEdit {
        buffer: String,
        error: Option<String>,
    },
    SavePath {
        buffer: String,
        error: Option<String>,
    },
    LoadPath {
        buffer: String,
        error: Option<String>,
    },
}

impl ScriptModal {
    pub fn is_open(&self) -> bool {
        !matches!(self, ScriptModal::Closed)
    }
}

pub struct ScriptComposerState {
    pub script: ChainScript,
    pub selected: usize,
    pub unsaved: bool,
    pub modal: ScriptModal,
}

impl ScriptComposerState {
    pub fn new(script: ChainScript) -> Self {
        Self {
            script,
            selected: 0,
            unsaved: false,
            modal: ScriptModal::Closed,
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

    pub fn add_stage_after(&mut self) {
        let insert_at = self.selected + 1;
        self.script.stages.insert(
            insert_at,
            ChainStage {
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
            },
        );
        self.selected = insert_at;
        self.unsaved = true;
    }

    pub fn add_stage_before(&mut self) {
        self.script.stages.insert(
            self.selected,
            ChainStage {
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
            },
        );
        // selection stays on the new stage (at the current index)
        self.unsaved = true;
    }

    pub fn delete_stage(&mut self) {
        if self.script.stages.len() <= 1 {
            return;
        }
        self.script.stages.remove(self.selected);
        if self.selected >= self.script.stages.len() {
            self.selected = self.script.stages.len() - 1;
        }
        self.unsaved = true;
    }

    pub fn cycle_transition(&mut self) {
        if self.selected == 0 {
            return;
        }
        if let Some(stage) = self.script.stages.get_mut(self.selected) {
            stage.transition = match stage.transition {
                TransitionMode::Smooth => TransitionMode::Cut,
                TransitionMode::Cut => TransitionMode::Fade,
                TransitionMode::Fade => TransitionMode::Smooth,
            };
            self.unsaved = true;
        }
    }

    pub fn open_prompt_editor(&mut self) {
        let buf = self
            .selected_stage()
            .map(|s| s.prompt.clone())
            .unwrap_or_default();
        self.modal = ScriptModal::PromptEdit { buffer: buf };
    }

    pub fn commit_prompt(&mut self) {
        if let ScriptModal::PromptEdit { buffer } = &self.modal {
            let value = buffer.clone();
            if let Some(stage) = self.selected_stage_mut() {
                stage.prompt = value;
                self.unsaved = true;
            }
        }
        self.modal = ScriptModal::Closed;
    }

    pub fn open_frames_editor(&mut self) {
        let buf = self
            .selected_stage()
            .map(|s| s.frames.to_string())
            .unwrap_or_default();
        self.modal = ScriptModal::FramesEdit {
            buffer: buf,
            error: None,
        };
    }

    pub fn commit_frames(&mut self) {
        if let ScriptModal::FramesEdit { buffer, .. } = &self.modal {
            let buf = buffer.clone();
            match buf.parse::<u32>() {
                Ok(n) if n >= 9 && n % 8 == 1 => {
                    if let Some(stage) = self.selected_stage_mut() {
                        stage.frames = n;
                        self.unsaved = true;
                    }
                    self.modal = ScriptModal::Closed;
                }
                Ok(n) => {
                    self.modal = ScriptModal::FramesEdit {
                        buffer: buf,
                        error: Some(format!("{n} is not 8k+1 (valid: 9, 17, 25, \u{2026}, 97)")),
                    };
                }
                Err(_) => {
                    self.modal = ScriptModal::FramesEdit {
                        buffer: buf,
                        error: Some("not a number".to_string()),
                    };
                }
            }
        }
    }

    pub fn open_save_dialog(&mut self) {
        self.modal = ScriptModal::SavePath {
            buffer: String::new(),
            error: None,
        };
    }

    pub fn open_load_dialog(&mut self) {
        self.modal = ScriptModal::LoadPath {
            buffer: String::new(),
            error: None,
        };
    }

    pub fn save_to_path(&mut self) {
        if let ScriptModal::SavePath { buffer, .. } = &self.modal {
            let path = std::path::PathBuf::from(buffer.trim());
            match mold_core::chain_toml::write_script(&self.script) {
                Ok(toml) => match std::fs::write(&path, &toml) {
                    Ok(()) => {
                        self.unsaved = false;
                        self.modal = ScriptModal::Closed;
                    }
                    Err(e) => {
                        self.modal = ScriptModal::SavePath {
                            buffer: buffer.clone(),
                            error: Some(format!("write failed: {e}")),
                        };
                    }
                },
                Err(e) => {
                    self.modal = ScriptModal::SavePath {
                        buffer: buffer.clone(),
                        error: Some(format!("serialise failed: {e}")),
                    };
                }
            }
        }
    }

    pub fn load_from_path(&mut self) {
        if let ScriptModal::LoadPath { buffer, .. } = &self.modal {
            let path = std::path::PathBuf::from(buffer.trim());
            match std::fs::read_to_string(&path) {
                Ok(contents) => match mold_core::chain_toml::read_script(&contents) {
                    Ok(script) => {
                        self.script = script;
                        self.selected = 0;
                        self.unsaved = false;
                        self.modal = ScriptModal::Closed;
                    }
                    Err(e) => {
                        self.modal = ScriptModal::LoadPath {
                            buffer: buffer.clone(),
                            error: Some(format!("parse failed: {e}")),
                        };
                    }
                },
                Err(e) => {
                    self.modal = ScriptModal::LoadPath {
                        buffer: buffer.clone(),
                        error: Some(format!("read failed: {e}")),
                    };
                }
            }
        }
    }

    pub fn cancel_modal(&mut self) {
        self.modal = ScriptModal::Closed;
    }

    /// Build a [`ChainRequest`] from the current script state, ready for
    /// submission to the server's `/api/generate/chain/stream` endpoint.
    pub fn build_chain_request(&self) -> mold_core::ChainRequest {
        mold_core::ChainRequest {
            model: self.script.chain.model.clone(),
            stages: self.script.stages.clone(),
            motion_tail_frames: self.script.chain.motion_tail_frames,
            width: self.script.chain.width,
            height: self.script.chain.height,
            fps: self.script.chain.fps,
            seed: self.script.chain.seed,
            steps: self.script.chain.steps,
            guidance: self.script.chain.guidance,
            strength: self.script.chain.strength,
            output_format: self.script.chain.output_format,
            placement: None,
            prompt: None,
            total_frames: None,
            clip_frames: None,
            source_image: None,
        }
    }
}

impl Default for ScriptComposerState {
    fn default() -> Self {
        Self {
            script: ChainScript {
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
            },
            selected: 0,
            unsaved: false,
            modal: ScriptModal::Closed,
        }
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

pub fn render(frame: &mut Frame, state: &ScriptComposerState, area: Rect, theme: &Theme) {
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
    render_modal(frame, state, area, theme);
}

fn render_stage_list(frame: &mut Frame, state: &ScriptComposerState, area: Rect, theme: &Theme) {
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

fn render_editor(frame: &mut Frame, state: &ScriptComposerState, area: Rect, theme: &Theme) {
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
                Span::styled(format!("{}", stage.frames), Style::default().fg(theme.text)),
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

fn render_footer(frame: &mut Frame, state: &ScriptComposerState, area: Rect, theme: &Theme) {
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

fn render_modal(frame: &mut Frame, state: &ScriptComposerState, area: Rect, theme: &Theme) {
    let (title, body_lines, hint) = match &state.modal {
        ScriptModal::Closed => return,
        ScriptModal::PromptEdit { buffer } => {
            let display = if buffer.is_empty() {
                "(empty)".to_string()
            } else {
                buffer.clone()
            };
            let lines: Vec<Line> = display.lines().map(|l| Line::from(l.to_string())).collect();
            (
                " Edit Prompt ",
                lines,
                "[Ctrl-S] Save  [Esc] Cancel  [Enter] Newline",
            )
        }
        ScriptModal::FramesEdit { buffer, error } => {
            let mut lines = vec![Line::from(buffer.as_str().to_string())];
            if let Some(err) = error {
                lines.push(Line::from(Span::styled(
                    err.clone(),
                    Style::default().fg(Color::Red),
                )));
            }
            (" Edit Frames ", lines, "[Enter] Save  [Esc] Cancel")
        }
        ScriptModal::SavePath { buffer, error } => {
            let mut lines = vec![Line::from(buffer.as_str().to_string())];
            if let Some(err) = error {
                lines.push(Line::from(Span::styled(
                    err.clone(),
                    Style::default().fg(Color::Red),
                )));
            }
            (" Save Script ", lines, "[Enter] Confirm  [Esc] Cancel")
        }
        ScriptModal::LoadPath { buffer, error } => {
            let mut lines = vec![Line::from(buffer.as_str().to_string())];
            if let Some(err) = error {
                lines.push(Line::from(Span::styled(
                    err.clone(),
                    Style::default().fg(Color::Red),
                )));
            }
            (" Load Script ", lines, "[Enter] Confirm  [Esc] Cancel")
        }
    };

    let modal_height = match &state.modal {
        ScriptModal::PromptEdit { .. } => 8u16,
        ScriptModal::FramesEdit { .. }
        | ScriptModal::SavePath { .. }
        | ScriptModal::LoadPath { .. } => 6u16,
        ScriptModal::Closed => return,
    };

    let w = (area.width * 60 / 100).clamp(30, 80).min(area.width);
    let h = modal_height.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    let popup_area = Rect::new(x, y, w, h);

    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border_focused())
        .title(title)
        .title_style(theme.title_focused())
        .style(Style::default().bg(theme.bg));

    let inner = block.inner(popup_area);
    frame.render_widget(block, popup_area);

    if inner.width == 0 || inner.height == 0 {
        return;
    }

    // Reserve last line for the hint
    let body_height = inner.height.saturating_sub(1);
    let body_area = Rect::new(inner.x, inner.y, inner.width, body_height);
    let hint_area = Rect::new(inner.x, inner.y + body_height, inner.width, 1);

    let body = Paragraph::new(body_lines).style(Style::default().fg(theme.text));
    frame.render_widget(body, body_area);

    let hint_line = Paragraph::new(Line::from(Span::styled(
        hint,
        Style::default().fg(theme.text_dim),
    )));
    frame.render_widget(hint_line, hint_area);
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

    #[test]
    fn add_stage_after_inserts_and_selects() {
        let mut s = make_state(2);
        s.selected = 0;
        s.add_stage_after();
        assert_eq!(s.script.stages.len(), 3);
        assert_eq!(s.selected, 1);
        assert!(s.script.stages[1].prompt.is_empty());
        assert!(s.unsaved);
    }

    #[test]
    fn add_stage_before_inserts_at_current() {
        let mut s = make_state(2);
        s.selected = 1;
        s.add_stage_before();
        assert_eq!(s.script.stages.len(), 3);
        assert_eq!(s.selected, 1);
        assert!(s.script.stages[1].prompt.is_empty());
        assert!(s.unsaved);
    }

    #[test]
    fn delete_stage_removes_current() {
        let mut s = make_state(3);
        s.selected = 1;
        s.delete_stage();
        assert_eq!(s.script.stages.len(), 2);
        assert!(s.unsaved);
    }

    #[test]
    fn delete_stage_adjusts_selection_at_end() {
        let mut s = make_state(3);
        s.selected = 2;
        s.delete_stage();
        assert_eq!(s.script.stages.len(), 2);
        assert_eq!(s.selected, 1);
    }

    #[test]
    fn cannot_delete_last_stage() {
        let mut s = make_state(1);
        s.delete_stage();
        assert_eq!(s.script.stages.len(), 1);
        assert!(!s.unsaved);
    }

    #[test]
    fn cycle_transition_smooth_cut_fade() {
        let mut s = make_state(2);
        s.selected = 1;
        s.script.stages[1].transition = TransitionMode::Smooth;
        s.cycle_transition();
        assert_eq!(s.script.stages[1].transition, TransitionMode::Cut);
        s.cycle_transition();
        assert_eq!(s.script.stages[1].transition, TransitionMode::Fade);
        s.cycle_transition();
        assert_eq!(s.script.stages[1].transition, TransitionMode::Smooth);
        assert!(s.unsaved);
    }

    #[test]
    fn cycle_transition_noop_on_stage_0() {
        let mut s = make_state(2);
        s.selected = 0;
        s.cycle_transition();
        assert_eq!(s.script.stages[0].transition, TransitionMode::Smooth);
        assert!(!s.unsaved);
    }

    #[test]
    fn commit_frames_rejects_non_8k1() {
        let mut s = make_state(2);
        s.modal = ScriptModal::FramesEdit {
            buffer: "100".into(),
            error: None,
        };
        s.commit_frames();
        assert!(matches!(
            s.modal,
            ScriptModal::FramesEdit { error: Some(_), .. }
        ));
    }

    #[test]
    fn commit_frames_accepts_8k1() {
        let mut s = make_state(2);
        s.modal = ScriptModal::FramesEdit {
            buffer: "49".into(),
            error: None,
        };
        s.commit_frames();
        assert_eq!(s.selected_stage().unwrap().frames, 49);
        assert!(matches!(s.modal, ScriptModal::Closed));
        assert!(s.unsaved);
    }

    #[test]
    fn commit_frames_rejects_below_9() {
        let mut s = make_state(2);
        s.modal = ScriptModal::FramesEdit {
            buffer: "1".into(),
            error: None,
        };
        s.commit_frames();
        assert!(matches!(
            s.modal,
            ScriptModal::FramesEdit { error: Some(_), .. }
        ));
    }

    #[test]
    fn commit_frames_rejects_non_number() {
        let mut s = make_state(2);
        s.modal = ScriptModal::FramesEdit {
            buffer: "abc".into(),
            error: None,
        };
        s.commit_frames();
        assert!(matches!(
            s.modal,
            ScriptModal::FramesEdit {
                error: Some(ref e),
                ..
            } if e == "not a number"
        ));
    }

    #[test]
    fn commit_prompt_saves_buffer() {
        let mut s = make_state(2);
        s.modal = ScriptModal::PromptEdit {
            buffer: "hello world".into(),
        };
        s.commit_prompt();
        assert_eq!(s.selected_stage().unwrap().prompt, "hello world");
        assert!(matches!(s.modal, ScriptModal::Closed));
        assert!(s.unsaved);
    }

    #[test]
    fn cancel_modal_closes_without_saving() {
        let mut s = make_state(2);
        let original = s.selected_stage().unwrap().frames;
        s.modal = ScriptModal::FramesEdit {
            buffer: "49".into(),
            error: None,
        };
        s.cancel_modal();
        assert!(matches!(s.modal, ScriptModal::Closed));
        assert_eq!(s.selected_stage().unwrap().frames, original);
        assert!(!s.unsaved);
    }

    #[test]
    fn open_prompt_editor_loads_current_prompt() {
        let mut s = make_state(2);
        s.script.stages[0].prompt = "existing prompt".into();
        s.open_prompt_editor();
        match &s.modal {
            ScriptModal::PromptEdit { buffer } => {
                assert_eq!(buffer, "existing prompt");
            }
            _ => panic!("expected PromptEdit modal"),
        }
    }

    #[test]
    fn open_frames_editor_loads_current_frames() {
        let mut s = make_state(2);
        s.open_frames_editor();
        match &s.modal {
            ScriptModal::FramesEdit { buffer, error } => {
                assert_eq!(buffer, "97");
                assert!(error.is_none());
            }
            _ => panic!("expected FramesEdit modal"),
        }
    }

    #[test]
    fn modal_is_open_reports_correctly() {
        let closed = ScriptModal::Closed;
        assert!(!closed.is_open());

        let prompt = ScriptModal::PromptEdit {
            buffer: String::new(),
        };
        assert!(prompt.is_open());

        let frames = ScriptModal::FramesEdit {
            buffer: String::new(),
            error: None,
        };
        assert!(frames.is_open());

        let save = ScriptModal::SavePath {
            buffer: String::new(),
            error: None,
        };
        assert!(save.is_open());

        let load = ScriptModal::LoadPath {
            buffer: String::new(),
            error: None,
        };
        assert!(load.is_open());
    }

    #[test]
    fn save_then_load_round_trips() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut s = make_state(3);
        s.script.stages[1].transition = TransitionMode::Cut;
        s.script.stages[2].prompt = "final scene".into();

        // Save
        s.modal = ScriptModal::SavePath {
            buffer: tmp.path().to_str().unwrap().to_string(),
            error: None,
        };
        s.save_to_path();
        assert!(matches!(s.modal, ScriptModal::Closed));
        assert!(!s.unsaved);

        // Load into a fresh state
        let mut s2 = ScriptComposerState {
            modal: ScriptModal::LoadPath {
                buffer: tmp.path().to_str().unwrap().to_string(),
                error: None,
            },
            ..ScriptComposerState::default()
        };
        s2.load_from_path();
        assert!(matches!(s2.modal, ScriptModal::Closed));
        assert_eq!(s2.script.stages.len(), 3);
        assert_eq!(s2.script.stages[1].transition, TransitionMode::Cut);
        assert_eq!(s2.script.stages[2].prompt, "final scene");
    }

    #[test]
    fn load_nonexistent_file_shows_error() {
        let mut s = ScriptComposerState {
            modal: ScriptModal::LoadPath {
                buffer: "/nonexistent/path.toml".to_string(),
                error: None,
            },
            ..ScriptComposerState::default()
        };
        s.load_from_path();
        assert!(matches!(
            s.modal,
            ScriptModal::LoadPath { error: Some(_), .. }
        ));
    }

    #[test]
    fn load_invalid_toml_shows_error() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "not valid toml [[[").unwrap();
        let mut s = ScriptComposerState {
            modal: ScriptModal::LoadPath {
                buffer: tmp.path().to_str().unwrap().to_string(),
                error: None,
            },
            ..ScriptComposerState::default()
        };
        s.load_from_path();
        assert!(matches!(
            s.modal,
            ScriptModal::LoadPath { error: Some(_), .. }
        ));
    }
}
