/// Semantic actions the TUI can perform, decoupled from raw key events.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Quit the application.
    Quit,
    /// Switch to a specific view tab.
    SwitchView(View),
    /// Switch to the next view tab (right arrow).
    ViewNext,
    /// Switch to the previous view tab (left arrow).
    ViewPrev,
    /// Cycle focus to the next panel/field.
    FocusNext,
    /// Cycle focus to the previous panel/field.
    FocusPrev,
    /// Navigate up in a list or parameter field.
    Up,
    /// Navigate down in a list or parameter field.
    Down,
    /// Increment the focused numeric parameter.
    Increment,
    /// Decrement the focused numeric parameter.
    Decrement,
    /// Confirm / submit (Enter).
    Confirm,
    /// Cancel / close popup (Escape).
    Cancel,
    /// Unfocus current panel — go to navigation mode.
    Unfocus,
    /// Start image generation.
    Generate,
    /// Open the model selector popup.
    OpenModelSelector,
    /// Randomize the seed.
    RandomizeSeed,
    /// Expand the current prompt via LLM.
    ExpandPrompt,
    /// Save the current preview image to a file.
    SaveImage,
    /// Toggle between local and remote inference mode.
    ToggleMode,
    /// Open model comparison mode.
    CompareModels,
    /// Navigate prompt history backward.
    HistoryPrev,
    /// Navigate prompt history forward.
    HistoryNext,
    /// Open fuzzy search over prompt history.
    SearchHistory,
    /// Gallery: re-generate with same parameters.
    Regenerate,
    /// Gallery: load parameters into Create view for editing.
    EditAndGenerate,
    /// Gallery: delete the selected image (shows confirmation).
    DeleteImage,
    /// Gallery: open the image file in system viewer.
    OpenFile,
    /// Gallery: upscale the selected image.
    UpscaleImage,
    /// Models: pull the selected model.
    PullModel,
    /// Models: remove the selected model.
    RemoveModel,
    /// Models: unload the loaded model from GPU.
    UnloadModel,
    /// Models: start filtering by name.
    FilterModels,
    /// Library: start editing the `/` filter.
    FilterLibrary,
    /// Library: clear a confirmed filter (Esc with a filter applied).
    FilterLibraryClear,
    /// Show the help overlay.
    ShowHelp,
    /// Toggle the collapsed state of the Negative prompt panel on the
    /// Create view.
    ToggleNegativePrompt,
    /// Toggle the Advanced disclosure on the Create view.
    ToggleAdvanced,
    /// Open the ^K command palette.
    OpenPalette,
    /// Apply a theme preset (palette theme commands).
    SetTheme(crate::ui::theme::ThemePreset),
    /// Enter the chain composer sub-mode of the Create view.
    ChainEnter,
    /// Leave the chain composer back to the Create compose mode.
    ChainExit,
    /// Image crop/pan: move viewport.
    PanLeft,
    PanRight,
    PanUp,
    PanDown,
    /// Image zoom.
    ZoomIn,
    ZoomOut,
    /// Reset image viewport to fit.
    ResetView,
    /// Gallery grid: move left one cell.
    GridLeft,
    /// Gallery grid: move right one cell.
    GridRight,
    /// Script: move selection down one stage.
    ScriptMoveDown,
    /// Script: move selection up one stage.
    ScriptMoveUp,
    /// Script: move selected stage down (reorder).
    ScriptReorderDown,
    /// Script: move selected stage up (reorder).
    ScriptReorderUp,
    /// Script: add a new stage after the current selection.
    ScriptAddAfter,
    /// Script: add a new stage before the current selection.
    ScriptAddBefore,
    /// Script: delete the current stage (shows confirmation if >1 stage).
    ScriptDelete,
    ScriptCycleTransition,
    /// Script: save the current script to a TOML file.
    ScriptSave,
    /// Script: load a script from a TOML file.
    ScriptLoad,
    /// Script: open the prompt editor modal.
    ScriptOpenPromptEditor,
    /// Script: open the frames editor modal.
    ScriptOpenFramesEditor,
    /// Script modal: insert a character.
    ScriptModalChar(char),
    /// Script modal: delete the last character.
    ScriptModalBackspace,
    /// Script modal: insert a newline (prompt editor).
    ScriptModalNewline,
    /// Script modal: submit the current value.
    ScriptModalSubmit,
    /// Script modal: cancel and close without saving.
    ScriptModalCancel,
    /// Script: submit the current chain script for generation.
    ScriptSubmit,
    /// Machines: open the stepped connect-a-machine flow.
    MachinesConnect,
    /// Machines: set the selected row as the sticky generation target.
    MachinesSetTarget,
    /// Machines: forget the selected host (confirmed; deletes its API key).
    MachinesForget,
    /// Machines: refresh telemetry and queue for all hosts now.
    MachinesRefresh,
    /// Machines: cancel the selected queued job (confirmed).
    MachinesCancelJob,
    /// No action (key not mapped or consumed by text input).
    None,
}

/// The five top-level workspaces — the shared Mold Studio IA
/// (Create · Library · Models · Machines · Settings, keys `1`–`5`).
///
/// The chain composer is not a tab: it is a sub-mode of Create
/// (`App::create_mode`), mirroring the graphical surfaces, where Sequence
/// is an Output setting of Create.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Create,
    Library,
    Models,
    Machines,
    Settings,
}

impl View {
    pub fn label(&self) -> &'static str {
        match self {
            View::Create => "Create",
            View::Library => "Library",
            View::Models => "Models",
            View::Machines => "Machines",
            View::Settings => "Settings",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            View::Create => 0,
            View::Library => 1,
            View::Models => 2,
            View::Machines => 3,
            View::Settings => 4,
        }
    }

    pub const ALL: [View; 5] = [
        View::Create,
        View::Library,
        View::Models,
        View::Machines,
        View::Settings,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_all_has_five_entries_in_ia_order() {
        // The 1–5 keys, the tab strip, and `scripts/tui-uat.sh view` all
        // depend on this exact order — the shared Mold Studio IA.
        assert_eq!(
            View::ALL.map(|v| v.label()),
            ["Create", "Library", "Models", "Machines", "Settings"]
        );
    }

    #[test]
    fn view_index_matches_all_position() {
        for (i, view) in View::ALL.iter().enumerate() {
            assert_eq!(view.index(), i, "{view:?} index drifted from ALL order");
        }
    }
}
