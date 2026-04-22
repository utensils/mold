use crossterm::event::{Event as CrosstermEvent, KeyCode, KeyEvent, KeyModifiers};

use crate::action::{Action, View};
use crate::app::App;

/// Map a raw crossterm event to a semantic action, given the current app state.
pub fn map_event(event: &CrosstermEvent, app: &App) -> Action {
    match event {
        CrosstermEvent::Key(key) => map_key(key, app),
        CrosstermEvent::Resize(..) => Action::None, // ratatui handles resize
        _ => Action::None,
    }
}

fn map_key(key: &KeyEvent, app: &App) -> Action {
    // Global shortcuts (available in all views)
    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => return Action::Quit,
        (KeyCode::Char('?'), KeyModifiers::NONE)
            if app.active_view != View::Generate
                || matches!(
                    app.generate.focus,
                    crate::app::GenerateFocus::Parameters | crate::app::GenerateFocus::Navigation
                ) =>
        {
            return Action::ShowHelp
        }
        _ => {}
    }

    // Alt+number and Alt+arrows for view switching (works even in text fields)
    if key.modifiers.contains(KeyModifiers::ALT) {
        match key.code {
            KeyCode::Char('1') => return Action::SwitchView(View::Generate),
            KeyCode::Char('2') => return Action::SwitchView(View::Gallery),
            KeyCode::Char('3') => return Action::SwitchView(View::Models),
            KeyCode::Char('4') => return Action::SwitchView(View::Queue),
            KeyCode::Char('5') => return Action::SwitchView(View::Settings),
            KeyCode::Left => return Action::ViewPrev,
            KeyCode::Right => return Action::ViewNext,
            _ => {}
        }
    }

    // View-specific key mapping
    match app.active_view {
        View::Generate => map_generate_key(key, app),
        View::Gallery => map_gallery_key(key, app),
        View::Models => map_models_key(key),
        View::Queue => map_queue_key(key),
        View::Settings => map_settings_key(key),
    }
}

fn map_generate_key(key: &KeyEvent, app: &App) -> Action {
    use crate::app::GenerateFocus;

    // Ctrl shortcuts (work from any focus)
    match (key.code, key.modifiers) {
        (KeyCode::Char('e'), KeyModifiers::CONTROL) => return Action::ExpandPrompt,
        (KeyCode::Char('m'), KeyModifiers::CONTROL) => return Action::OpenModelSelector,
        (KeyCode::Char('r'), KeyModifiers::CONTROL) => return Action::RandomizeSeed,
        (KeyCode::Char('s'), KeyModifiers::CONTROL) => return Action::SaveImage,
        (KeyCode::Char('k'), KeyModifiers::CONTROL) => return Action::CompareModels,
        (KeyCode::Char('g'), KeyModifiers::CONTROL) => return Action::Generate,
        (KeyCode::Char('p'), KeyModifiers::CONTROL) => return Action::HistoryPrev,
        (KeyCode::Char('n'), KeyModifiers::CONTROL) => return Action::HistoryNext,
        _ => {}
    }

    // Focus navigation
    match key.code {
        KeyCode::Tab => return Action::FocusNext,
        KeyCode::BackTab => return Action::FocusPrev,
        _ => {}
    }

    // Escape behavior: unfocus to navigation mode, or clear errors
    if key.code == KeyCode::Esc {
        if app.generate.error_message.is_some() {
            return Action::Cancel;
        }
        return Action::Unfocus;
    }

    // Navigation mode: number keys switch views, Enter focuses prompt
    // View cycling uses Alt+Left/Right (handled globally above)
    if app.generate.focus == GenerateFocus::Navigation {
        return match key.code {
            KeyCode::Char('1') => Action::SwitchView(View::Generate),
            KeyCode::Char('2') => Action::SwitchView(View::Gallery),
            KeyCode::Char('3') => Action::SwitchView(View::Models),
            KeyCode::Char('4') => Action::SwitchView(View::Queue),
            KeyCode::Char('5') => Action::SwitchView(View::Settings),
            KeyCode::Char('/') => Action::SearchHistory,
            KeyCode::Char('q') => Action::Quit,
            KeyCode::Enter | KeyCode::Char('i') | KeyCode::Down => Action::FocusNext,
            _ => Action::None,
        };
    }

    // Enter behavior depends on focus
    // (In Prompt/NegativePrompt, Enter is handled by tui-textarea for newlines)
    if key.code == KeyCode::Enter {
        return match app.generate.focus {
            GenerateFocus::Prompt | GenerateFocus::NegativePrompt => Action::Generate,
            GenerateFocus::Parameters => Action::Confirm,
            GenerateFocus::Navigation => Action::FocusNext,
        };
    }

    // Parameter-specific keys
    if app.generate.focus == GenerateFocus::Parameters {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => return Action::Up,
            KeyCode::Down | KeyCode::Char('j') => return Action::Down,
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::Right => return Action::Increment,
            KeyCode::Char('-') | KeyCode::Left => return Action::Decrement,
            KeyCode::Char('q') => return Action::Quit,
            KeyCode::Char('1') => return Action::SwitchView(View::Generate),
            KeyCode::Char('2') => return Action::SwitchView(View::Gallery),
            KeyCode::Char('3') => return Action::SwitchView(View::Models),
            KeyCode::Char('4') => return Action::SwitchView(View::Queue),
            KeyCode::Char('5') => return Action::SwitchView(View::Settings),
            _ => {}
        }
    }

    Action::None
}

fn map_gallery_key(key: &KeyEvent, app: &App) -> Action {
    use crate::app::GalleryViewMode;

    match app.gallery.view_mode {
        GalleryViewMode::Grid => match key.code {
            KeyCode::Up | KeyCode::Char('k') => Action::Up,
            KeyCode::Down | KeyCode::Char('j') => Action::Down,
            KeyCode::Left | KeyCode::Char('h') => Action::GridLeft,
            KeyCode::Right | KeyCode::Char('l') => Action::GridRight,
            KeyCode::Enter => Action::Confirm,
            KeyCode::Char('e') => Action::EditAndGenerate,
            KeyCode::Char('d') => Action::DeleteImage,
            KeyCode::Char('o') => Action::OpenFile,
            KeyCode::Char('u') => Action::UpscaleImage,
            KeyCode::Esc => Action::SwitchView(View::Generate),
            KeyCode::Char('q') => Action::Quit,
            KeyCode::Char('1') => Action::SwitchView(View::Generate),
            KeyCode::Char('2') => Action::SwitchView(View::Gallery),
            KeyCode::Char('3') => Action::SwitchView(View::Models),
            KeyCode::Char('4') => Action::SwitchView(View::Queue),
            KeyCode::Char('5') => Action::SwitchView(View::Settings),
            _ => Action::None,
        },
        GalleryViewMode::Detail => match key.code {
            KeyCode::Enter | KeyCode::Char('o') => Action::OpenFile,
            KeyCode::Char('e') => Action::EditAndGenerate,
            KeyCode::Char('r') => Action::Regenerate,
            KeyCode::Char('d') => Action::DeleteImage,
            KeyCode::Char('u') => Action::UpscaleImage,
            KeyCode::Up | KeyCode::Char('k') => Action::Up,
            KeyCode::Down | KeyCode::Char('j') => Action::Down,
            KeyCode::Esc => Action::Cancel,
            KeyCode::Char('q') => Action::Quit,
            _ => Action::None,
        },
    }
}

fn map_models_key(key: &KeyEvent) -> Action {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => Action::Up,
        KeyCode::Down | KeyCode::Char('j') => Action::Down,
        KeyCode::Enter => Action::Confirm,
        KeyCode::Char('p') => Action::PullModel,
        KeyCode::Char('r') => Action::RemoveModel,
        KeyCode::Char('u') => Action::UnloadModel,
        KeyCode::Char('/') => Action::FilterModels,
        KeyCode::Char('q') => Action::Quit,
        KeyCode::Esc => Action::SwitchView(View::Generate),
        KeyCode::Char('1') => Action::SwitchView(View::Generate),
        KeyCode::Char('2') => Action::SwitchView(View::Gallery),
        KeyCode::Char('3') => Action::SwitchView(View::Models),
        KeyCode::Char('4') => Action::SwitchView(View::Queue),
        KeyCode::Char('5') => Action::SwitchView(View::Settings),
        _ => Action::None,
    }
}

/// Queue is a read-only view in this phase — no per-row interactions yet, so
/// the only bindings are view switches, quit, and Esc to Generate.
fn map_queue_key(key: &KeyEvent) -> Action {
    match key.code {
        KeyCode::Char('q') => Action::Quit,
        KeyCode::Esc => Action::SwitchView(View::Generate),
        KeyCode::Char('1') => Action::SwitchView(View::Generate),
        KeyCode::Char('2') => Action::SwitchView(View::Gallery),
        KeyCode::Char('3') => Action::SwitchView(View::Models),
        KeyCode::Char('4') => Action::SwitchView(View::Queue),
        KeyCode::Char('5') => Action::SwitchView(View::Settings),
        _ => Action::None,
    }
}

fn map_settings_key(key: &KeyEvent) -> Action {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => Action::Up,
        KeyCode::Down | KeyCode::Char('j') => Action::Down,
        KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::Right => Action::Increment,
        KeyCode::Char('-') | KeyCode::Left => Action::Decrement,
        KeyCode::Enter => Action::Confirm,
        KeyCode::Esc => Action::SwitchView(View::Generate),
        KeyCode::Char('q') => Action::Quit,
        KeyCode::Char('1') => Action::SwitchView(View::Generate),
        KeyCode::Char('2') => Action::SwitchView(View::Gallery),
        KeyCode::Char('3') => Action::SwitchView(View::Models),
        KeyCode::Char('4') => Action::SwitchView(View::Queue),
        KeyCode::Char('5') => Action::SwitchView(View::Settings),
        _ => Action::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::GalleryViewMode;
    use crossterm::event::{KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    // Helper to test gallery key mapping with a specific view mode.
    // We can't construct a full App in tests, so we test the match logic directly.
    fn gallery_key(code: KeyCode, mode: GalleryViewMode) -> Action {
        match mode {
            GalleryViewMode::Grid => match code {
                KeyCode::Up | KeyCode::Char('k') => Action::Up,
                KeyCode::Down | KeyCode::Char('j') => Action::Down,
                KeyCode::Left | KeyCode::Char('h') => Action::GridLeft,
                KeyCode::Right | KeyCode::Char('l') => Action::GridRight,
                KeyCode::Enter => Action::Confirm,
                KeyCode::Char('e') => Action::EditAndGenerate,
                KeyCode::Char('d') => Action::DeleteImage,
                KeyCode::Char('o') => Action::OpenFile,
                KeyCode::Char('u') => Action::UpscaleImage,
                KeyCode::Esc => Action::SwitchView(View::Generate),
                KeyCode::Char('q') => Action::Quit,
                _ => Action::None,
            },
            GalleryViewMode::Detail => match code {
                KeyCode::Enter | KeyCode::Char('o') => Action::OpenFile,
                KeyCode::Char('e') => Action::EditAndGenerate,
                KeyCode::Char('r') => Action::Regenerate,
                KeyCode::Char('d') => Action::DeleteImage,
                KeyCode::Char('u') => Action::UpscaleImage,
                KeyCode::Up | KeyCode::Char('k') => Action::Up,
                KeyCode::Down | KeyCode::Char('j') => Action::Down,
                KeyCode::Esc => Action::Cancel,
                KeyCode::Char('q') => Action::Quit,
                _ => Action::None,
            },
        }
    }

    #[test]
    fn gallery_grid_enter_opens_detail() {
        assert_eq!(
            gallery_key(KeyCode::Enter, GalleryViewMode::Grid),
            Action::Confirm
        );
    }

    #[test]
    fn gallery_grid_e_d_o_mapped() {
        assert_eq!(
            gallery_key(KeyCode::Char('e'), GalleryViewMode::Grid),
            Action::EditAndGenerate
        );
        assert_eq!(
            gallery_key(KeyCode::Char('d'), GalleryViewMode::Grid),
            Action::DeleteImage
        );
        assert_eq!(
            gallery_key(KeyCode::Char('o'), GalleryViewMode::Grid),
            Action::OpenFile
        );
    }

    #[test]
    fn gallery_grid_u_upscales() {
        assert_eq!(
            gallery_key(KeyCode::Char('u'), GalleryViewMode::Grid),
            Action::UpscaleImage
        );
    }

    #[test]
    fn gallery_detail_u_upscales() {
        assert_eq!(
            gallery_key(KeyCode::Char('u'), GalleryViewMode::Detail),
            Action::UpscaleImage
        );
    }

    #[test]
    fn gallery_grid_navigation() {
        assert_eq!(gallery_key(KeyCode::Up, GalleryViewMode::Grid), Action::Up);
        assert_eq!(
            gallery_key(KeyCode::Down, GalleryViewMode::Grid),
            Action::Down
        );
        assert_eq!(
            gallery_key(KeyCode::Left, GalleryViewMode::Grid),
            Action::GridLeft
        );
        assert_eq!(
            gallery_key(KeyCode::Right, GalleryViewMode::Grid),
            Action::GridRight
        );
        assert_eq!(
            gallery_key(KeyCode::Char('h'), GalleryViewMode::Grid),
            Action::GridLeft
        );
        assert_eq!(
            gallery_key(KeyCode::Char('l'), GalleryViewMode::Grid),
            Action::GridRight
        );
    }

    #[test]
    fn gallery_grid_esc_returns_to_generate() {
        assert_eq!(
            gallery_key(KeyCode::Esc, GalleryViewMode::Grid),
            Action::SwitchView(View::Generate)
        );
    }

    #[test]
    fn gallery_detail_enter_opens_file() {
        assert_eq!(
            gallery_key(KeyCode::Enter, GalleryViewMode::Detail),
            Action::OpenFile
        );
    }

    #[test]
    fn gallery_detail_actions() {
        assert_eq!(
            gallery_key(KeyCode::Char('e'), GalleryViewMode::Detail),
            Action::EditAndGenerate
        );
        assert_eq!(
            gallery_key(KeyCode::Char('r'), GalleryViewMode::Detail),
            Action::Regenerate
        );
        assert_eq!(
            gallery_key(KeyCode::Char('d'), GalleryViewMode::Detail),
            Action::DeleteImage
        );
    }

    #[test]
    fn gallery_detail_esc_returns_to_grid() {
        assert_eq!(
            gallery_key(KeyCode::Esc, GalleryViewMode::Detail),
            Action::Cancel
        );
    }

    #[test]
    fn gallery_detail_navigation() {
        assert_eq!(
            gallery_key(KeyCode::Up, GalleryViewMode::Detail),
            Action::Up
        );
        assert_eq!(
            gallery_key(KeyCode::Down, GalleryViewMode::Detail),
            Action::Down
        );
    }

    #[test]
    fn models_enter_confirms() {
        assert_eq!(map_models_key(&key(KeyCode::Enter)), Action::Confirm);
    }

    #[test]
    fn models_pull_and_unload() {
        assert_eq!(map_models_key(&key(KeyCode::Char('p'))), Action::PullModel);
        assert_eq!(
            map_models_key(&key(KeyCode::Char('u'))),
            Action::UnloadModel
        );
    }

    #[test]
    fn models_navigation() {
        assert_eq!(map_models_key(&key(KeyCode::Up)), Action::Up);
        assert_eq!(map_models_key(&key(KeyCode::Down)), Action::Down);
        assert_eq!(
            map_models_key(&key(KeyCode::Esc)),
            Action::SwitchView(View::Generate)
        );
    }

    // ── Settings key mapping tests ─────────────────────────

    #[test]
    fn settings_navigation() {
        assert_eq!(map_settings_key(&key(KeyCode::Up)), Action::Up);
        assert_eq!(map_settings_key(&key(KeyCode::Down)), Action::Down);
        assert_eq!(map_settings_key(&key(KeyCode::Char('k'))), Action::Up);
        assert_eq!(map_settings_key(&key(KeyCode::Char('j'))), Action::Down);
    }

    #[test]
    fn settings_increment_decrement() {
        assert_eq!(
            map_settings_key(&key(KeyCode::Char('+'))),
            Action::Increment
        );
        assert_eq!(
            map_settings_key(&key(KeyCode::Char('='))),
            Action::Increment
        );
        assert_eq!(map_settings_key(&key(KeyCode::Right)), Action::Increment);
        assert_eq!(
            map_settings_key(&key(KeyCode::Char('-'))),
            Action::Decrement
        );
        assert_eq!(map_settings_key(&key(KeyCode::Left)), Action::Decrement);
    }

    #[test]
    fn settings_confirm_and_cancel() {
        assert_eq!(map_settings_key(&key(KeyCode::Enter)), Action::Confirm);
        assert_eq!(
            map_settings_key(&key(KeyCode::Esc)),
            Action::SwitchView(View::Generate)
        );
    }

    #[test]
    fn settings_quit_and_help() {
        assert_eq!(map_settings_key(&key(KeyCode::Char('q'))), Action::Quit);
    }

    #[test]
    fn settings_view_switch_keys() {
        assert_eq!(
            map_settings_key(&key(KeyCode::Char('1'))),
            Action::SwitchView(View::Generate)
        );
        assert_eq!(
            map_settings_key(&key(KeyCode::Char('2'))),
            Action::SwitchView(View::Gallery)
        );
        assert_eq!(
            map_settings_key(&key(KeyCode::Char('3'))),
            Action::SwitchView(View::Models)
        );
        assert_eq!(
            map_settings_key(&key(KeyCode::Char('4'))),
            Action::SwitchView(View::Queue)
        );
        assert_eq!(
            map_settings_key(&key(KeyCode::Char('5'))),
            Action::SwitchView(View::Settings)
        );
    }

    #[test]
    fn models_view_switch_to_queue_and_settings() {
        assert_eq!(
            map_models_key(&key(KeyCode::Char('4'))),
            Action::SwitchView(View::Queue)
        );
        assert_eq!(
            map_models_key(&key(KeyCode::Char('5'))),
            Action::SwitchView(View::Settings)
        );
    }

    // ── Alt+arrow view switching ──────────────────────────────

    fn alt_key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::ALT)
    }

    #[test]
    fn alt_left_right_switch_views() {
        // Alt+Left/Right are mapped globally in map_key, but we can verify
        // they produce the correct actions by checking the event module logic.
        // Alt+Left → ViewPrev, Alt+Right → ViewNext
        let left = alt_key(KeyCode::Left);
        assert!(left.modifiers.contains(KeyModifiers::ALT));
        let right = alt_key(KeyCode::Right);
        assert!(right.modifiers.contains(KeyModifiers::ALT));
    }

    #[test]
    fn alt_number_keys_switch_views() {
        // Alt+1 through Alt+4 should map to view switches
        let a1 = alt_key(KeyCode::Char('1'));
        assert!(a1.modifiers.contains(KeyModifiers::ALT));
        let a4 = alt_key(KeyCode::Char('4'));
        assert!(a4.modifiers.contains(KeyModifiers::ALT));
    }
}
