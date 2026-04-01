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
                || app.generate.focus == crate::app::GenerateFocus::Parameters =>
        {
            return Action::ShowHelp
        }
        _ => {}
    }

    // Alt+number for view switching (works even in text fields)
    if key.modifiers.contains(KeyModifiers::ALT) {
        match key.code {
            KeyCode::Char('1') => return Action::SwitchView(View::Generate),
            KeyCode::Char('2') => return Action::SwitchView(View::Gallery),
            KeyCode::Char('3') => return Action::SwitchView(View::Models),
            _ => {}
        }
    }

    // View-specific key mapping
    match app.active_view {
        View::Generate => map_generate_key(key, app),
        View::Gallery => map_gallery_key(key),
        View::Models => map_models_key(key),
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
        _ => {}
    }

    // Focus navigation
    match key.code {
        KeyCode::Tab => return Action::FocusNext,
        KeyCode::BackTab => return Action::FocusPrev,
        _ => {}
    }

    // Escape: clear errors, or if nothing to clear, deselect/go to prompt
    if key.code == KeyCode::Esc {
        if app.generate.error_message.is_some() {
            return Action::Cancel;
        }
        // If in params, go back to prompt
        if app.generate.focus != GenerateFocus::Prompt {
            return Action::FocusPrev;
        }
        return Action::None;
    }

    // Enter behavior depends on focus
    if key.code == KeyCode::Enter {
        return match app.generate.focus {
            GenerateFocus::Prompt | GenerateFocus::NegativePrompt => {
                // In text fields: Ctrl+Enter or just Enter generates
                // (multiline input uses Shift+Enter in textarea)
                Action::Generate
            }
            GenerateFocus::Parameters => {
                // In params: Enter activates the field (model selector, toggle, etc.)
                Action::Confirm
            }
        };
    }

    // Parameter-specific keys
    if app.generate.focus == GenerateFocus::Parameters {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => return Action::Up,
            KeyCode::Down | KeyCode::Char('j') => return Action::Down,
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::Right => return Action::Increment,
            KeyCode::Char('-') | KeyCode::Left => return Action::Decrement,
            _ => {}
        }
    }

    // 'q' to quit only when not in a text field
    if key.code == KeyCode::Char('q') && app.generate.focus == GenerateFocus::Parameters {
        return Action::Quit;
    }

    // Number keys for view switching (only when not in text input)
    if app.generate.focus == GenerateFocus::Parameters {
        match key.code {
            KeyCode::Char('1') => return Action::SwitchView(View::Generate),
            KeyCode::Char('2') => return Action::SwitchView(View::Gallery),
            KeyCode::Char('3') => return Action::SwitchView(View::Models),
            _ => {}
        }
    }

    Action::None
}

fn map_gallery_key(key: &KeyEvent) -> Action {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => Action::Up,
        KeyCode::Down | KeyCode::Char('j') => Action::Down,
        KeyCode::Enter => Action::Regenerate,
        KeyCode::Char('e') => Action::EditAndGenerate,
        KeyCode::Char('d') => Action::DeleteImage,
        KeyCode::Char('o') => Action::OpenFile,
        KeyCode::Char('h') | KeyCode::Left => Action::PanLeft,
        KeyCode::Char('l') | KeyCode::Right => Action::PanRight,
        KeyCode::Char('+') | KeyCode::Char('=') => Action::ZoomIn,
        KeyCode::Char('-') => Action::ZoomOut,
        KeyCode::Esc => Action::ResetView,
        KeyCode::Char('q') => Action::Quit,
        KeyCode::Char('1') => Action::SwitchView(View::Generate),
        KeyCode::Char('2') => Action::SwitchView(View::Gallery),
        KeyCode::Char('3') => Action::SwitchView(View::Models),
        _ => Action::None,
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
        KeyCode::Esc => Action::Cancel,
        KeyCode::Char('1') => Action::SwitchView(View::Generate),
        KeyCode::Char('2') => Action::SwitchView(View::Gallery),
        KeyCode::Char('3') => Action::SwitchView(View::Models),
        _ => Action::None,
    }
}
