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
    /// Gallery: load parameters into Generate view for editing.
    EditAndGenerate,
    /// Gallery: delete the selected image.
    DeleteImage,
    /// Gallery: open the image file in system viewer.
    OpenFile,
    /// Models: pull the selected model.
    PullModel,
    /// Models: remove the selected model.
    RemoveModel,
    /// Models: unload the loaded model from GPU.
    UnloadModel,
    /// Models: start filtering by name.
    FilterModels,
    /// Show the help overlay.
    ShowHelp,
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
    /// No action (key not mapped or consumed by text input).
    None,
}

/// The three top-level views.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Generate,
    Gallery,
    Models,
}

impl View {
    pub fn label(&self) -> &'static str {
        match self {
            View::Generate => "Generate",
            View::Gallery => "Gallery",
            View::Models => "Models",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            View::Generate => 0,
            View::Gallery => 1,
            View::Models => 2,
        }
    }

    pub const ALL: [View; 3] = [View::Generate, View::Gallery, View::Models];
}
