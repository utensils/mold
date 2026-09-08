//! Theme cards — the Appearance panel's bordered mini-card grid.
//!
//! Each preset renders as a `CARD_W × CARD_H` bordered card: three swatch
//! dots (the preset's bg / accent / info — the mockup's sw1/sw2/sw3) plus
//! a dim descriptor on the first row, the preset label on the second. The
//! selected card carries the focus-coloured border (focus-by-border-colour,
//! no glow, per the design system).
//!
//! Geometry is pure math ([`card_grid`], [`first_visible_row`],
//! [`appearance_panel_height`]) so the layout contract stays unit-testable:
//! the panel is sized to fit every card row and clips to whole rows with
//! vertical scrolling when the terminal is too short.

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};

use super::theme::{Theme, ThemePreset};
use super::widgets::truncate_with_ellipsis;

/// Card width in terminal cells (borders included).
pub const CARD_W: u16 = 18;
/// Card height in terminal rows (borders included): 2 borders + dots row
/// + label row.
pub const CARD_H: u16 = 4;

/// Grid geometry for `count` cards in a pane `width` cells wide.
///
/// Returns `(cols, rows)`. Columns are never zero — a pane narrower than
/// one card still gets a single (clipped) column — and never exceed
/// `count`, so a short list doesn't reserve phantom columns.
pub fn card_grid(count: usize, width: u16) -> (usize, usize) {
    let fit = (width / CARD_W) as usize;
    let cols = fit.clamp(1, count.max(1));
    let rows = count.div_ceil(cols);
    (cols, rows)
}

/// First card-row to draw so the selected row stays on screen.
///
/// With `visible_rows == 0` (degenerate pane) this returns 0. Scrolling is
/// clamped so the last page is always full when `total_rows` allows.
pub fn first_visible_row(selected_row: usize, total_rows: usize, visible_rows: usize) -> usize {
    if visible_rows == 0 || total_rows <= visible_rows {
        return 0;
    }
    let max_first = total_rows - visible_rows;
    selected_row
        .saturating_sub(visible_rows.saturating_sub(1))
        .min(max_first)
}

/// Height of the Appearance panel (borders included) for a pane whose
/// *inner* width is `inner_width`, clamped to `available` total rows.
///
/// Layout contract: unclipped, the inner area (height − 2 borders) is
/// exactly `rows × CARD_H`, so every card row fits. Clipped, the height
/// still holds at least one full card row (scrolling covers the rest)
/// unless `available` itself can't.
pub fn appearance_panel_height(inner_width: u16, available: u16) -> u16 {
    let (_, rows) = card_grid(ThemePreset::FAMILIES.len(), inner_width);
    let desired = (rows as u16) * CARD_H + TONE_ROW_H + 2;
    desired.min(available.max(CARD_H + TONE_ROW_H + 2))
}

/// Rows the Light / Dark control occupies beneath the cards.
pub const TONE_ROW_H: u16 = 1;

/// Render the tone control — the other half of the picker. The cards name the
/// THEME and this names the TONE; neither moves the other. There is no System
/// position: a terminal has no OS appearance to follow, the emulator owns the
/// background, and offering a choice that cannot resolve would be a lie.
pub fn render_tone_row(
    frame: &mut Frame,
    theme: &Theme,
    area: Rect,
    current: ThemePreset,
    focused: bool,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let pick = |label: &'static str, on: bool| {
        Span::styled(
            format!(" {label} "),
            if on && focused {
                Style::default().fg(theme.frame).bg(theme.accent)
            } else if on {
                Style::default().fg(theme.text).bg(theme.surface2)
            } else {
                Style::default().fg(theme.text_dim)
            },
        )
    };
    let label_style = if focused {
        Style::default().fg(theme.accent)
    } else {
        Style::default().fg(theme.text_dim)
    };
    let line = Line::from(vec![
        Span::styled("Light or dark  ", label_style),
        pick("Light", current.is_light()),
        Span::raw(" "),
        pick("Dark", !current.is_light()),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

/// Render the theme card grid into `area` (the Appearance panel's inner
/// rect). `current` is both the selection and the live-applied preset.
pub fn render_theme_cards(
    frame: &mut Frame,
    theme: &Theme,
    area: Rect,
    current: ThemePreset,
    focused: bool,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    // Five cards, one per THEME, each painted in the tone in force. The tone
    // itself is the Appearance panel's own Light / Dark row — a card that also
    // named a tone would be a second, contradicting picker.
    let presets: Vec<ThemePreset> = ThemePreset::FAMILIES
        .iter()
        .map(|p| p.with_tone(current.is_light()))
        .collect();
    let (cols, rows) = card_grid(presets.len(), area.width);
    let visible_rows = (area.height / CARD_H) as usize;
    let selected_idx = current.family_index();
    let scroll = first_visible_row(selected_idx / cols, rows, visible_rows);

    for (i, preset) in presets.iter().enumerate() {
        let (row, col) = (i / cols, i % cols);
        if row < scroll || row >= scroll + visible_rows.max(1) {
            continue;
        }
        let x = area.x + (col as u16) * CARD_W;
        let y = area.y + ((row - scroll) as u16) * CARD_H;
        if x >= area.right() || y >= area.bottom() {
            continue;
        }
        let card = Rect {
            x,
            y,
            width: CARD_W.min(area.right() - x),
            height: CARD_H.min(area.bottom() - y),
        };
        render_card(frame, theme, card, *preset, *preset == current, focused);
    }
}

/// Render a single theme card. Selection is signalled by border colour
/// only — the app-wide focus signal.
fn render_card(
    frame: &mut Frame,
    theme: &Theme,
    area: Rect,
    preset: ThemePreset,
    selected: bool,
    focused: bool,
) {
    let border_style = if selected && focused {
        theme.border_focused()
    } else if selected {
        // Selected but the pane isn't focused — text-coloured border keeps
        // the selection findable without claiming the focus colour.
        Style::default().fg(theme.text)
    } else {
        theme.border()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    // Row 1 — a plane and the theme's two accents, plus the dim descriptor,
    // truncated to the card width. The first dot is `surface2` rather than
    // `bg`: a light theme's ground is within a percent or two of the card it
    // sits on, so a bg dot renders invisible and the trio reads as two.
    let palette = preset.build();
    let desc_budget = (inner.width as usize).saturating_sub(4);
    let desc = truncate_with_ellipsis(preset.description(), desc_budget);
    let dots = Line::from(vec![
        Span::styled("●", Style::default().fg(palette.surface2)),
        Span::styled("●", Style::default().fg(palette.accent)),
        Span::styled("●", Style::default().fg(palette.info)),
        Span::raw(" "),
        Span::styled(desc, theme.faint()),
    ]);

    // Row 2 — the preset label.
    let label_style = if selected {
        Style::default().fg(theme.text).add_modifier(Modifier::BOLD)
    } else {
        theme.dim()
    };
    let label = Line::from(Span::styled(
        truncate_with_ellipsis(preset.label(), inner.width as usize),
        label_style,
    ));

    frame.render_widget(Paragraph::new(vec![dots, label]), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    const PRESET_COUNT: usize = ThemePreset::FAMILIES.len();

    #[test]
    fn card_grid_never_zero_cols() {
        // Panes narrower than one card, zero-width panes, and empty
        // lists must all keep cols >= 1 — a zero would divide-by-zero
        // the row math and index math downstream.
        assert_eq!(card_grid(PRESET_COUNT, 0).0, 1);
        assert_eq!(card_grid(PRESET_COUNT, CARD_W - 1).0, 1);
        assert_eq!(card_grid(0, 0).0, 1);
        assert_eq!(card_grid(0, 200).0, 1);
        for width in 0..300u16 {
            assert!(card_grid(PRESET_COUNT, width).0 >= 1, "width {width}");
        }
    }

    #[test]
    fn card_grid_fits_every_theme_at_80_cols() {
        // An 80-column terminal gives the Appearance panel a 78-cell
        // inner width → 4 columns of 18-wide cards → 2 rows for the five
        // themes. Tone is not a card, so the grid holds five, not ten.
        let (cols, rows) = card_grid(PRESET_COUNT, 78);
        assert_eq!(PRESET_COUNT, 5);
        assert_eq!((cols, rows), (4, 2));
        assert!(cols * rows >= PRESET_COUNT, "grid must hold every theme");
    }

    #[test]
    fn card_grid_cols_capped_by_count() {
        // A very wide pane must not reserve phantom columns past the
        // card count.
        assert_eq!(card_grid(2, 500), (2, 1));
        assert_eq!(card_grid(PRESET_COUNT, 500), (PRESET_COUNT, 1));
    }

    #[test]
    fn card_grid_rows_cover_all_cards() {
        for width in 1..300u16 {
            let (cols, rows) = card_grid(PRESET_COUNT, width);
            assert!(
                cols * rows >= PRESET_COUNT,
                "width {width}: {cols}x{rows} drops cards"
            );
            // And never a fully-empty trailing row.
            assert!(
                cols * (rows - 1) < PRESET_COUNT,
                "width {width}: {cols}x{rows} has an empty row"
            );
        }
    }

    #[test]
    fn first_visible_row_keeps_selection_on_screen() {
        // 3 total rows, 2 visible: selecting rows 0/1 keeps the top,
        // selecting row 2 scrolls by one.
        assert_eq!(first_visible_row(0, 3, 2), 0);
        assert_eq!(first_visible_row(1, 3, 2), 0);
        assert_eq!(first_visible_row(2, 3, 2), 1);
        // Everything visible → never scroll.
        assert_eq!(first_visible_row(2, 3, 3), 0);
        // Degenerate pane → no scroll math.
        assert_eq!(first_visible_row(2, 3, 0), 0);
        // Selection is always within [first, first + visible).
        for total in 1..6usize {
            for visible in 1..6usize {
                for sel in 0..total {
                    let first = first_visible_row(sel, total, visible);
                    assert!(
                        (first..first + visible).contains(&sel),
                        "sel {sel} total {total} visible {visible} first {first}"
                    );
                    assert!(first + visible.min(total) <= total.max(visible));
                }
            }
        }
    }

    #[test]
    fn appearance_panel_height_matches_row_count() {
        // Layout contract (repo rule): with enough vertical room the
        // panel's inner area (height − 2 borders) must fit every card
        // row exactly — otherwise the last presets render clipped.
        for width in 1..300u16 {
            let (_, rows) = card_grid(PRESET_COUNT, width);
            let h = appearance_panel_height(width, 200);
            assert_eq!(
                (h - 2 - TONE_ROW_H) as usize,
                rows * CARD_H as usize,
                "width {width}: inner height must equal rows × CARD_H + the tone row"
            );
        }
    }

    #[test]
    fn appearance_panel_height_clips_to_whole_rows_when_short() {
        // At 78 inner width the full grid wants 2×4 + 1 tone row + 2
        // borders = 11; with only 8 available the panel clips but still
        // shows a whole card row (scrolling covers the rest).
        let h = appearance_panel_height(78, 8);
        assert_eq!(h, 8);
        let visible_rows = (h - 2 - TONE_ROW_H) / CARD_H;
        assert!(visible_rows >= 1, "at least one full card row visible");
        // Never taller than desired even when space is plentiful.
        assert_eq!(appearance_panel_height(78, 100), 11);
    }

    #[test]
    fn the_swatch_trio_stays_visible_on_a_light_theme() {
        // A light theme's bg is within a couple of percent of the card ground,
        // so a bg dot renders invisible and the trio reads as two. The first
        // dot is a raised plane instead — legible in both tones.
        for preset in ThemePreset::ALL {
            let t = preset.build();
            assert_ne!(t.surface2, t.bg, "{preset:?} plane must differ from ground");
        }
    }

    #[test]
    fn render_theme_cards_is_safe_at_any_size() {
        let theme = Theme::default();
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        for (w, h) in [(0, 0), (1, 1), (10, 3), (78, 12), (80, 4)] {
            terminal
                .draw(|frame| {
                    let area = Rect::new(0, 0, w, h);
                    render_theme_cards(frame, &theme, area, ThemePreset::MochaDark, true);
                })
                .unwrap();
        }
    }

    #[test]
    fn render_shows_selected_label_and_scrolls_to_selection() {
        let theme = Theme::default();

        let render_with = |current: ThemePreset, w: u16, h: u16| -> String {
            let backend = TestBackend::new(w, h);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal
                .draw(|frame| {
                    render_theme_cards(frame, &theme, Rect::new(0, 0, w, h), current, true);
                })
                .unwrap();
            let buf = terminal.backend().buffer().clone();
            let mut out = String::new();
            for y in 0..buf.area.height {
                for x in 0..buf.area.width {
                    out.push_str(buf[(x, y)].symbol());
                }
                out.push('\n');
            }
            out
        };

        // Full-height grid shows both the first and last themes.
        let full = render_with(ThemePreset::MochaDark, 78, 12);
        assert!(full.contains("Mocha"), "{full}");
        assert!(full.contains("Nebula"), "{full}");
        // Names only: no card says which tone it is.
        assert!(!full.contains("Dark"), "{full}");

        // Clipped to one card row, the selection scrolls into view:
        // selecting the last theme must render it, and hide row 0.
        let clipped = render_with(ThemePreset::NebulaLight, 78, 4);
        assert!(clipped.contains("Nebula"), "{clipped}");
        assert!(!clipped.contains("Mocha"), "{clipped}");
    }
}
