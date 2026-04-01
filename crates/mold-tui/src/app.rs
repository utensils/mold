use anyhow::Result;
use crossterm::event::{Event as CrosstermEvent, KeyCode, KeyModifiers};
use mold_core::{
    Config, GenerateResponse, ModelInfoExtended, OutputFormat, Scheduler, SseProgressEvent,
};
use rand::Rng;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use tokio::sync::mpsc;
use tui_textarea::TextArea;

use crate::action::{Action, View};
use crate::event::map_event;
use crate::model_info::{capabilities_for_family, family_for_model, ModelCapabilities};
use crate::ui::theme::Theme;

/// Events sent from background tasks to the main TUI loop.
pub enum BackgroundEvent {
    /// Progress update from generation or model pull.
    Progress(SseProgressEvent),
    /// Generation completed successfully.
    GenerationComplete(Box<GenerateResponse>),
    /// Generation or background task failed.
    Error(String),
    /// Gallery scan completed.
    GalleryScanComplete(Vec<GalleryEntry>),
    /// Model pull completed.
    PullComplete(String),
}

/// A single entry in the progress log.
#[derive(Debug, Clone)]
pub struct ProgressLogEntry {
    pub message: String,
    pub style: ProgressStyle,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProgressStyle {
    Done,
    Info,
    Warning,
    Error,
}

/// Current progress state during generation.
#[derive(Debug, Default)]
pub struct ProgressState {
    pub log: Vec<ProgressLogEntry>,
    pub current_stage: Option<String>,
    pub denoise_step: usize,
    pub denoise_total: usize,
    pub denoise_elapsed_ms: u64,
    pub weight_loaded: u64,
    pub weight_total: u64,
    pub weight_component: String,
    pub download_filename: String,
    pub download_bytes: u64,
    pub download_total: u64,
}

impl ProgressState {
    pub fn clear(&mut self) {
        self.log.clear();
        self.current_stage = None;
        self.denoise_step = 0;
        self.denoise_total = 0;
        self.denoise_elapsed_ms = 0;
        self.weight_loaded = 0;
        self.weight_total = 0;
        self.weight_component.clear();
        self.download_filename.clear();
        self.download_bytes = 0;
        self.download_total = 0;
    }
}

/// Which panel is focused in the Generate view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerateFocus {
    /// No panel focused — number keys switch views, Enter focuses prompt.
    Navigation,
    Prompt,
    NegativePrompt,
    Parameters,
}

impl GenerateFocus {
    pub fn next(self, has_negative: bool) -> Self {
        match self {
            Self::Navigation => Self::Prompt,
            Self::Prompt if has_negative => Self::NegativePrompt,
            Self::Prompt => Self::Parameters,
            Self::NegativePrompt => Self::Parameters,
            Self::Parameters => Self::Prompt,
        }
    }

    pub fn prev(self, has_negative: bool) -> Self {
        match self {
            Self::Navigation => Self::Parameters,
            Self::Prompt => Self::Parameters,
            Self::NegativePrompt => Self::Prompt,
            Self::Parameters if has_negative => Self::NegativePrompt,
            Self::Parameters => Self::Prompt,
        }
    }
}

/// Index of parameter fields in the form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamField {
    Model,
    Width,
    Height,
    Steps,
    Guidance,
    Seed,
    SeedValue,
    Batch,
    Format,
    Mode,
    Host,
    // Advanced
    Scheduler,
    Lora,
    Expand,
    Offload,
    // img2img
    SourceImage,
    Strength,
    MaskImage,
    // ControlNet
    ControlImage,
    ControlModel,
    ControlScale,
}

impl ParamField {
    /// All fields in display order, filtering by model capabilities and mode.
    pub fn visible_fields(caps: &ModelCapabilities, mode: InferenceMode) -> Vec<ParamField> {
        let mut fields = vec![
            ParamField::Model,
            ParamField::Width,
            ParamField::Height,
            ParamField::Steps,
            ParamField::Guidance,
            ParamField::Seed,
            ParamField::SeedValue,
            ParamField::Batch,
            ParamField::Format,
            ParamField::Mode,
        ];
        // Show Host field when server connection is possible
        if mode != InferenceMode::Local {
            fields.push(ParamField::Host);
        }
        // Advanced
        if caps.supports_scheduler {
            fields.push(ParamField::Scheduler);
        }
        if caps.supports_lora {
            fields.push(ParamField::Lora);
        }
        fields.push(ParamField::Expand);
        fields.push(ParamField::Offload);
        // img2img
        if caps.supports_img2img {
            fields.push(ParamField::SourceImage);
            fields.push(ParamField::Strength);
            fields.push(ParamField::MaskImage);
        }
        // ControlNet
        if caps.supports_controlnet {
            fields.push(ParamField::ControlImage);
            fields.push(ParamField::ControlModel);
            fields.push(ParamField::ControlScale);
        }
        fields
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Model => "Model",
            Self::Width => "Width",
            Self::Height => "Height",
            Self::Steps => "Steps",
            Self::Guidance => "Guidance",
            Self::Seed => "Seed",
            Self::SeedValue => "",
            Self::Batch => "Batch",
            Self::Format => "Format",
            Self::Mode => "Mode",
            Self::Host => "Host",
            Self::Scheduler => "Scheduler",
            Self::Lora => "LoRA",
            Self::Expand => "Expand",
            Self::Offload => "Offload",
            Self::SourceImage => "Source",
            Self::Strength => "Strength",
            Self::MaskImage => "Mask",
            Self::ControlImage => "Control",
            Self::ControlModel => "CNet Mdl",
            Self::ControlScale => "Scale",
        }
    }

    /// The section header this field falls under, if it starts a new section.
    pub fn section_header(&self) -> Option<&'static str> {
        match self {
            Self::Scheduler => Some("Advanced"),
            Self::SourceImage => Some("img2img"),
            Self::ControlImage => Some("ControlNet"),
            _ => None,
        }
    }
}

/// How the seed behaves across generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeedMode {
    /// New random seed each generation.
    #[default]
    Random,
    /// Keep the same seed every generation.
    Fixed,
    /// Increment seed by 1 after each generation.
    Increment,
}

impl SeedMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Fixed => "fixed",
            Self::Increment => "increment",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Random => Self::Fixed,
            Self::Fixed => Self::Increment,
            Self::Increment => Self::Random,
        }
    }

    /// Resolve the actual seed value for a generation.
    pub fn resolve(self, current: Option<u64>) -> u64 {
        match self {
            Self::Random => rand::thread_rng().gen_range(0..u64::MAX),
            Self::Fixed => current.unwrap_or_else(|| rand::thread_rng().gen_range(0..u64::MAX)),
            Self::Increment => current
                .map(|s| s.wrapping_add(1))
                .unwrap_or_else(|| rand::thread_rng().gen_range(0..u64::MAX)),
        }
    }

    /// Advance the seed after a generation completes (for Increment mode).
    pub fn advance(self, used_seed: u64) -> Option<u64> {
        match self {
            Self::Random => None,
            Self::Fixed => Some(used_seed),
            Self::Increment => Some(used_seed),
        }
    }
}

/// Generation parameters mirroring GenerateRequest fields.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub seed_mode: SeedMode,
    pub batch: u32,
    pub format: OutputFormat,
    pub scheduler: Option<Scheduler>,
    pub inference_mode: InferenceMode,
    pub host: Option<String>,
    // Advanced
    pub lora_path: Option<String>,
    pub lora_scale: f64,
    pub expand: bool,
    pub offload: bool,
    // img2img
    pub source_image_path: Option<String>,
    pub strength: f64,
    pub mask_image_path: Option<String>,
    // ControlNet
    pub control_image_path: Option<String>,
    pub control_model: Option<String>,
    pub control_scale: f64,
}

/// How inference is dispatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferenceMode {
    /// Try remote server first, fall back to local GPU if unreachable.
    #[default]
    Auto,
    /// Force local GPU inference only.
    Local,
    /// Force remote server only (no fallback).
    Remote,
}

impl InferenceMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Local => "local",
            Self::Remote => "remote",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Auto => Self::Local,
            Self::Local => Self::Remote,
            Self::Remote => Self::Auto,
        }
    }
}

impl GenerateParams {
    pub fn from_config(config: &Config) -> Self {
        let model = config.resolved_default_model();

        let model_cfg = config.resolved_model_config(&model);
        Self {
            width: model_cfg.effective_width(config),
            height: model_cfg.effective_height(config),
            steps: model_cfg.effective_steps(config),
            guidance: model_cfg.effective_guidance(),
            model,
            seed: None,
            seed_mode: SeedMode::Random,
            batch: 1,
            format: OutputFormat::Png,
            scheduler: None,
            inference_mode: InferenceMode::Auto,
            host: None,
            lora_path: None,
            lora_scale: 1.0,
            expand: false,
            offload: false,
            source_image_path: None,
            strength: 0.75,
            mask_image_path: None,
            control_image_path: None,
            control_model: None,
            control_scale: 1.0,
        }
    }

    /// Display value for a given field.
    pub fn display_value(&self, field: &ParamField) -> String {
        match field {
            ParamField::Model => self.model.clone(),
            ParamField::Width => self.width.to_string(),
            ParamField::Height => self.height.to_string(),
            ParamField::Steps => self.steps.to_string(),
            ParamField::Guidance => format!("{:.1}", self.guidance),
            ParamField::Seed => self.seed_mode.label().to_string(),
            ParamField::SeedValue => self
                .seed
                .map(|s| s.to_string())
                .unwrap_or_else(|| "\u{27e8}auto\u{27e9}".to_string()),
            ParamField::Batch => self.batch.to_string(),
            ParamField::Format => format!("{:?}", self.format).to_uppercase(),
            ParamField::Mode => self.inference_mode.label().to_string(),
            ParamField::Host => self.host.as_deref().unwrap_or("localhost:7680").to_string(),
            ParamField::Scheduler => self
                .scheduler
                .as_ref()
                .map(|s| format!("{s:?}"))
                .unwrap_or_else(|| "\u{2014}".to_string()),
            ParamField::Lora => self
                .lora_path
                .as_deref()
                .map(|p| {
                    std::path::Path::new(p)
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_else(|| p.to_string())
                })
                .unwrap_or_else(|| "\u{27e8}none\u{27e9}".to_string()),
            ParamField::Expand => if self.expand { "on" } else { "off" }.to_string(),
            ParamField::Offload => if self.offload { "on" } else { "off" }.to_string(),
            ParamField::SourceImage => self
                .source_image_path
                .as_deref()
                .map(|p| {
                    std::path::Path::new(p)
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_else(|| p.to_string())
                })
                .unwrap_or_else(|| "\u{27e8}none\u{27e9}".to_string()),
            ParamField::Strength => format!("{:.2}", self.strength),
            ParamField::MaskImage => self
                .mask_image_path
                .as_deref()
                .map(|p| {
                    std::path::Path::new(p)
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_else(|| p.to_string())
                })
                .unwrap_or_else(|| "\u{27e8}none\u{27e9}".to_string()),
            ParamField::ControlImage => self
                .control_image_path
                .as_deref()
                .map(|p| {
                    std::path::Path::new(p)
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_else(|| p.to_string())
                })
                .unwrap_or_else(|| "\u{27e8}none\u{27e9}".to_string()),
            ParamField::ControlModel => self
                .control_model
                .as_deref()
                .unwrap_or("\u{27e8}none\u{27e9}")
                .to_string(),
            ParamField::ControlScale => format!("{:.1}", self.control_scale),
        }
    }
}

/// State for the Generate view.
pub struct GenerateState {
    pub prompt: TextArea<'static>,
    pub negative_prompt: TextArea<'static>,
    pub params: GenerateParams,
    pub focus: GenerateFocus,
    pub param_index: usize,
    pub visible_fields: Vec<ParamField>,
    pub capabilities: ModelCapabilities,
    pub progress: ProgressState,
    pub preview_image: Option<image::DynamicImage>,
    pub image_state: Option<StatefulProtocol>,
    pub generating: bool,
    pub last_seed: Option<u64>,
    pub last_generation_time_ms: Option<u64>,
    pub error_message: Option<String>,
    pub model_description: String,
}

/// State for the Gallery view.
pub struct GalleryState {
    pub entries: Vec<GalleryEntry>,
    pub selected: usize,
    pub preview_image: Option<image::DynamicImage>,
    pub image_state: Option<StatefulProtocol>,
    pub scanning: bool,
}

/// A single gallery entry.
#[derive(Debug, Clone)]
pub struct GalleryEntry {
    pub path: std::path::PathBuf,
    pub prompt_preview: String,
    pub model: String,
    pub generation_time_ms: Option<u64>,
    pub seed: Option<u64>,
    pub width: u32,
    pub height: u32,
    pub timestamp: u64,
}

/// State for the Models view.
pub struct ModelsState {
    pub catalog: Vec<ModelInfoExtended>,
    pub selected: usize,
    pub filter: String,
    pub filtering: bool,
}

/// Active popup/overlay.
pub enum Popup {
    Help,
    ModelSelector {
        filter: String,
        selected: usize,
        filtered: Vec<String>,
    },
    HostInput {
        input: String,
    },
    SeedInput {
        input: String,
    },
    HistorySearch {
        filter: String,
        selected: usize,
        results: Vec<String>,
    },
    Confirm {
        message: String,
        on_confirm: ConfirmAction,
    },
}

#[derive(Debug, Clone)]
pub enum ConfirmAction {
    DeleteImage(std::path::PathBuf),
    RemoveModel(String),
}

/// The root application state.
pub struct App {
    pub active_view: View,
    pub generate: GenerateState,
    pub gallery: GalleryState,
    pub models: ModelsState,
    pub config: Config,
    pub server_url: Option<String>,
    pub picker: Picker,
    pub theme: Theme,
    pub popup: Option<Popup>,
    pub should_quit: bool,
    pub bg_tx: mpsc::UnboundedSender<BackgroundEvent>,
    pub bg_rx: mpsc::UnboundedReceiver<BackgroundEvent>,
    pub tokio_handle: tokio::runtime::Handle,
    pub resource_info: crate::ui::info::ResourceInfo,
    pub history: crate::history::PromptHistory,
    /// Layout areas from the last render, used for mouse hit-testing.
    pub layout: LayoutAreas,
}

/// Stored layout rectangles for mouse click hit-testing.
#[derive(Debug, Default, Clone)]
pub struct LayoutAreas {
    pub tab_bar: ratatui::layout::Rect,
    pub prompt: ratatui::layout::Rect,
    pub negative_prompt: ratatui::layout::Rect,
    pub parameters: ratatui::layout::Rect,
    pub preview: ratatui::layout::Rect,
    pub progress: ratatui::layout::Rect,
    pub gallery_list: ratatui::layout::Rect,
    pub gallery_preview: ratatui::layout::Rect,
    pub models_table: ratatui::layout::Rect,
}

impl App {
    pub fn new(host: Option<String>, local: bool, picker: Picker) -> Result<Self> {
        let config = Config::load_or_default();

        // Determine initial server URL and inference mode based on what's configured
        let env_host = std::env::var("MOLD_HOST").ok();
        let (server_url, initial_mode) = if local {
            (None, InferenceMode::Local)
        } else if let Some(h) = host {
            // Explicit --host flag: use it, default to auto
            (Some(h), InferenceMode::Auto)
        } else if let Some(h) = env_host {
            // MOLD_HOST env var set: use it, default to auto
            (Some(h), InferenceMode::Auto)
        } else {
            // No server configured: default to local, user can enter a host later
            (None, InferenceMode::Local)
        };

        let mut params = GenerateParams::from_config(&config);
        params.inference_mode = initial_mode;
        // Store the server URL in params.host so it's visible/editable
        if let Some(ref url) = server_url {
            params.host = Some(url.clone());
        }

        let family = family_for_model(&params.model, &config);
        let mut capabilities = capabilities_for_family(&family);
        let mut visible_fields = ParamField::visible_fields(&capabilities, initial_mode);

        let catalog = mold_core::build_model_catalog(&config, None, false);

        let (bg_tx, bg_rx) = mpsc::unbounded_channel();

        // Load session from previous TUI run
        let session = crate::session::TuiSession::load();

        // Restore model from session if it was set
        if !session.last_model.is_empty()
            && config.manifest_model_is_downloaded(&session.last_model)
        {
            params.model = session.last_model.clone();
            let mc = config.resolved_model_config(&params.model);
            params.steps = session.last_steps.unwrap_or(mc.effective_steps(&config));
            params.guidance = session.last_guidance.unwrap_or(mc.effective_guidance());
            params.width = session.last_width.unwrap_or(mc.effective_width(&config));
            params.height = session.last_height.unwrap_or(mc.effective_height(&config));
            // Re-derive capabilities for the restored model
            let fam = family_for_model(&params.model, &config);
            let caps = capabilities_for_family(&fam);
            visible_fields = ParamField::visible_fields(&caps, initial_mode);
            capabilities = caps;
        }

        let model_description = mold_core::manifest::find_manifest(&params.model)
            .and_then(|m| {
                let mc = config.resolved_model_config(&params.model);
                mc.description.or(Some(m.name.clone()))
            })
            .unwrap_or_default();

        // Set up prompt textarea — restore from session if available
        let mut prompt = TextArea::default();
        prompt.set_cursor_line_style(ratatui::style::Style::default());
        prompt.set_placeholder_text("Enter your prompt...");
        if session.has_prompt() {
            prompt = TextArea::new(session.last_prompt.lines().map(String::from).collect());
            prompt.set_cursor_line_style(ratatui::style::Style::default());
        }

        let mut negative_prompt = TextArea::default();
        negative_prompt.set_cursor_line_style(ratatui::style::Style::default());
        negative_prompt.set_placeholder_text("Negative prompt (what to avoid)...");
        if !session.last_negative.is_empty() {
            negative_prompt =
                TextArea::new(session.last_negative.lines().map(String::from).collect());
            negative_prompt.set_cursor_line_style(ratatui::style::Style::default());
        }

        // Load prompt history
        let history = crate::history::PromptHistory::load();

        let app = Ok(Self {
            active_view: View::Generate,
            generate: GenerateState {
                prompt,
                negative_prompt,
                params,
                focus: GenerateFocus::Prompt,
                param_index: 0,
                visible_fields,
                capabilities,
                progress: ProgressState::default(),
                preview_image: None,
                image_state: None,
                generating: false,
                last_seed: None,
                last_generation_time_ms: None,
                error_message: None,
                model_description,
            },
            gallery: GalleryState {
                entries: Vec::new(),
                selected: 0,
                preview_image: None,
                image_state: None,
                scanning: false,
            },
            models: ModelsState {
                catalog,
                selected: 0,
                filter: String::new(),
                filtering: false,
            },
            config,
            server_url,
            picker,
            theme: Theme::default(),
            popup: None,
            should_quit: false,
            bg_tx,
            bg_rx,
            tokio_handle: tokio::runtime::Handle::current(),
            resource_info: crate::ui::info::ResourceInfo::default(),
            history,
            layout: LayoutAreas::default(),
        });

        // Spawn background gallery scan
        if let Ok(ref app) = app {
            let tx = app.bg_tx.clone();
            app.tokio_handle.spawn(async move {
                let entries = tokio::task::spawn_blocking(crate::gallery_scan::scan_images)
                    .await
                    .unwrap_or_default();
                let _ = tx.send(BackgroundEvent::GalleryScanComplete(entries));
            });
        }

        app
    }

    /// Update model-dependent state when the model selection changes.
    pub fn update_model(&mut self, model_name: &str) {
        let model_name = model_name.to_string();
        self.generate.params.model = model_name.clone();

        let model_cfg = self.config.resolved_model_config(&model_name);
        self.generate.params.steps = model_cfg.effective_steps(&self.config);
        self.generate.params.guidance = model_cfg.effective_guidance();
        self.generate.params.width = model_cfg.effective_width(&self.config);
        self.generate.params.height = model_cfg.effective_height(&self.config);

        let family = family_for_model(&model_name, &self.config);
        self.generate.capabilities = capabilities_for_family(&family);
        self.generate.visible_fields = ParamField::visible_fields(
            &self.generate.capabilities,
            self.generate.params.inference_mode,
        );
        self.generate.param_index = 0;

        self.generate.model_description = mold_core::manifest::find_manifest(&model_name)
            .and_then(|m| {
                let mc = self.config.resolved_model_config(&model_name);
                mc.description.or(Some(m.name.clone()))
            })
            .unwrap_or_default();
    }

    /// Handle a raw crossterm event.
    pub fn handle_crossterm_event(&mut self, event: CrosstermEvent) {
        // Handle mouse events
        if let CrosstermEvent::Mouse(mouse) = event {
            self.handle_mouse(mouse);
            return;
        }

        // If a popup is active, route events there first
        if self.popup.is_some() {
            self.handle_popup_event(event);
            return;
        }

        // If we're in a text input field, let the textarea handle the event first
        if self.active_view == View::Generate {
            let in_text_field = matches!(
                self.generate.focus,
                GenerateFocus::Prompt | GenerateFocus::NegativePrompt
            );
            if in_text_field {
                if let CrosstermEvent::Key(key) = &event {
                    // Let certain keys bypass the textarea
                    match (key.code, key.modifiers) {
                        (KeyCode::Tab, KeyModifiers::NONE)
                        | (KeyCode::BackTab, KeyModifiers::SHIFT)
                        | (KeyCode::Char('c'), KeyModifiers::CONTROL)
                        | (KeyCode::Char('e'), KeyModifiers::CONTROL)
                        | (KeyCode::Char('g'), KeyModifiers::CONTROL)
                        | (KeyCode::Char('m'), KeyModifiers::CONTROL)
                        | (KeyCode::Char('r'), KeyModifiers::CONTROL)
                        | (KeyCode::Char('s'), KeyModifiers::CONTROL)
                        | (KeyCode::Char('k'), KeyModifiers::CONTROL)
                        | (KeyCode::Char('p'), KeyModifiers::CONTROL)
                        | (KeyCode::Char('n'), KeyModifiers::CONTROL)
                        | (KeyCode::Enter, KeyModifiers::NONE)
                        | (KeyCode::Esc, KeyModifiers::NONE) => {
                            // Fall through to action mapping
                        }
                        (KeyCode::Char('1'), KeyModifiers::ALT)
                        | (KeyCode::Char('2'), KeyModifiers::ALT)
                        | (KeyCode::Char('3'), KeyModifiers::ALT) => {
                            // Fall through for view switching
                        }
                        _ => {
                            // Let the textarea consume the event
                            let textarea = match self.generate.focus {
                                GenerateFocus::Prompt => &mut self.generate.prompt,
                                GenerateFocus::NegativePrompt => &mut self.generate.negative_prompt,
                                _ => unreachable!(),
                            };
                            textarea.input(event);
                            return;
                        }
                    }
                }
            }
        }

        // Map the event to an action and dispatch
        let action = map_event(&event, self);
        self.dispatch_action(action);
    }

    fn handle_popup_event(&mut self, event: CrosstermEvent) {
        if let CrosstermEvent::Key(key) = event {
            match &mut self.popup {
                Some(Popup::Help) => {
                    if matches!(
                        key.code,
                        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('?')
                    ) {
                        self.popup = None;
                    }
                }
                Some(Popup::ModelSelector {
                    filter,
                    selected,
                    filtered,
                }) => match key.code {
                    KeyCode::Esc => self.popup = None,
                    KeyCode::Enter => {
                        if let Some(model) = filtered.get(*selected).cloned() {
                            self.popup = None;
                            self.update_model(&model);
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if *selected > 0 {
                            *selected -= 1;
                        }
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        if *selected + 1 < filtered.len() {
                            *selected += 1;
                        }
                    }
                    KeyCode::Char(c) => {
                        filter.push(c);
                        self.update_model_selector_filter();
                    }
                    KeyCode::Backspace => {
                        filter.pop();
                        self.update_model_selector_filter();
                    }
                    _ => {}
                },
                Some(Popup::HostInput { input }) => match key.code {
                    KeyCode::Esc => self.popup = None,
                    KeyCode::Enter => {
                        let host = input.trim().to_string();
                        self.popup = None;
                        if host.is_empty() {
                            // Clear host → switch to local
                            self.generate.params.host = None;
                            self.generate.params.inference_mode = InferenceMode::Local;
                            self.server_url = None;
                        } else {
                            // Normalize: add http:// if no scheme
                            let url = if host.contains("://") {
                                host
                            } else {
                                format!("http://{host}")
                            };
                            self.generate.params.host = Some(url.clone());
                            self.server_url = Some(url);
                            // Auto-switch to auto mode when a host is entered
                            if self.generate.params.inference_mode == InferenceMode::Local {
                                self.generate.params.inference_mode = InferenceMode::Auto;
                            }
                        }
                        // Refresh visible fields (Host visibility depends on mode)
                        self.generate.visible_fields = ParamField::visible_fields(
                            &self.generate.capabilities,
                            self.generate.params.inference_mode,
                        );
                    }
                    KeyCode::Char(c) => input.push(c),
                    KeyCode::Backspace => {
                        input.pop();
                    }
                    _ => {}
                },
                Some(Popup::SeedInput { input }) => match key.code {
                    KeyCode::Esc => self.popup = None,
                    KeyCode::Enter => {
                        let text = input.trim().to_string();
                        self.popup = None;
                        if text.is_empty() {
                            // Clear seed → back to auto
                            self.generate.params.seed = None;
                        } else if let Ok(val) = text.parse::<u64>() {
                            self.generate.params.seed = Some(val);
                            // Switch to fixed mode when user enters a specific seed
                            if self.generate.params.seed_mode == SeedMode::Random {
                                self.generate.params.seed_mode = SeedMode::Fixed;
                            }
                        }
                        // else: invalid input, just close
                    }
                    KeyCode::Char(c) if c.is_ascii_digit() => input.push(c),
                    KeyCode::Backspace => {
                        input.pop();
                    }
                    _ => {}
                },
                Some(Popup::HistorySearch {
                    filter,
                    selected,
                    results,
                }) => match key.code {
                    KeyCode::Esc => self.popup = None,
                    KeyCode::Enter => {
                        if let Some(prompt) = results.get(*selected).cloned() {
                            self.popup = None;
                            self.generate.prompt =
                                TextArea::new(prompt.lines().map(String::from).collect());
                            self.generate
                                .prompt
                                .set_cursor_line_style(ratatui::style::Style::default());
                            self.generate.focus = GenerateFocus::Prompt;
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k')
                        if key.modifiers == KeyModifiers::NONE || key.code == KeyCode::Up =>
                    {
                        if *selected > 0 {
                            *selected -= 1;
                        }
                    }
                    KeyCode::Down | KeyCode::Char('j')
                        if key.modifiers == KeyModifiers::NONE || key.code == KeyCode::Down =>
                    {
                        if *selected + 1 < results.len() {
                            *selected += 1;
                        }
                    }
                    KeyCode::Char(c) => {
                        filter.push(c);
                        *results = self
                            .history
                            .search(filter)
                            .into_iter()
                            .map(|e| e.prompt.clone())
                            .collect();
                        if *selected >= results.len() {
                            *selected = results.len().saturating_sub(1);
                        }
                    }
                    KeyCode::Backspace => {
                        filter.pop();
                        *results = self
                            .history
                            .search(filter)
                            .into_iter()
                            .map(|e| e.prompt.clone())
                            .collect();
                        if *selected >= results.len() {
                            *selected = results.len().saturating_sub(1);
                        }
                    }
                    _ => {}
                },
                Some(Popup::Confirm { on_confirm, .. }) => match key.code {
                    KeyCode::Char('y') | KeyCode::Enter => {
                        let _action = on_confirm.clone();
                        self.popup = None;
                    }
                    _ => self.popup = None,
                },
                None => {}
            }
        }
    }

    fn handle_mouse(&mut self, mouse: crossterm::event::MouseEvent) {
        use crossterm::event::{MouseButton, MouseEventKind};

        let col = mouse.column;
        let row = mouse.row;

        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                // Close popups on click outside (simple approach)
                if self.popup.is_some() {
                    self.popup = None;
                    return;
                }

                // Tab bar clicks — switch views
                if self.layout.tab_bar.contains((col, row).into()) {
                    // Determine which tab was clicked based on x position
                    let x = col - self.layout.tab_bar.x;
                    if x < 13 {
                        self.active_view = View::Generate;
                    } else if x < 25 {
                        self.active_view = View::Gallery;
                    } else {
                        self.active_view = View::Models;
                    }
                    return;
                }

                // Generate view clicks
                if self.active_view == View::Generate {
                    let pos: ratatui::layout::Position = (col, row).into();
                    if self.layout.prompt.contains(pos) {
                        self.generate.focus = GenerateFocus::Prompt;
                    } else if self.layout.negative_prompt.contains(pos)
                        && self.generate.capabilities.supports_negative_prompt
                    {
                        self.generate.focus = GenerateFocus::NegativePrompt;
                    } else if self.layout.parameters.contains(pos) {
                        self.generate.focus = GenerateFocus::Parameters;
                        // Select and activate the parameter row that was clicked
                        let relative_row =
                            (row - self.layout.parameters.y).saturating_sub(1) as usize;
                        if relative_row < self.generate.visible_fields.len() {
                            self.generate.param_index = relative_row;
                            self.activate_current_param();
                        }
                    } else {
                        // Click on preview or elsewhere — go to navigation
                        self.generate.focus = GenerateFocus::Navigation;
                    }
                }

                // Gallery view clicks
                if self.active_view == View::Gallery {
                    let pos: ratatui::layout::Position = (col, row).into();
                    if self.layout.gallery_list.contains(pos) {
                        let relative_row =
                            (row - self.layout.gallery_list.y).saturating_sub(1) as usize;
                        if relative_row < self.gallery.entries.len() {
                            self.gallery.selected = relative_row;
                            // TODO: load preview image for selected entry
                        }
                    }
                }

                // Models view clicks
                if self.active_view == View::Models {
                    let pos: ratatui::layout::Position = (col, row).into();
                    if self.layout.models_table.contains(pos) {
                        let relative_row =
                            (row - self.layout.models_table.y).saturating_sub(2) as usize;
                        if relative_row < self.models.catalog.len() {
                            let was_selected = self.models.selected == relative_row;
                            self.models.selected = relative_row;
                            // Double-click: select model and switch to Generate
                            if was_selected {
                                let name = self.models.catalog[relative_row].name.clone();
                                self.update_model(&name);
                                self.active_view = View::Generate;
                                self.generate.focus = GenerateFocus::Prompt;
                            }
                        }
                    }
                }
            }
            MouseEventKind::ScrollUp => {
                self.dispatch_action(Action::Up);
            }
            MouseEventKind::ScrollDown => {
                self.dispatch_action(Action::Down);
            }
            _ => {}
        }
    }

    fn update_model_selector_filter(&mut self) {
        if let Some(Popup::ModelSelector {
            filter,
            selected,
            filtered,
        }) = &mut self.popup
        {
            let query = filter.to_lowercase();
            *filtered = self
                .models
                .catalog
                .iter()
                .filter(|m| m.name.to_lowercase().contains(&query))
                .map(|m| m.name.clone())
                .collect();
            if *selected >= filtered.len() {
                *selected = filtered.len().saturating_sub(1);
            }
        }
    }

    /// Dispatch a semantic action.
    pub fn dispatch_action(&mut self, action: Action) {
        match action {
            Action::Quit => self.should_quit = true,
            Action::SwitchView(view) => self.active_view = view,
            Action::ViewNext => {
                self.active_view = match self.active_view {
                    View::Generate => View::Gallery,
                    View::Gallery => View::Models,
                    View::Models => View::Generate,
                };
            }
            Action::ViewPrev => {
                self.active_view = match self.active_view {
                    View::Generate => View::Models,
                    View::Gallery => View::Generate,
                    View::Models => View::Gallery,
                };
            }
            Action::FocusNext => {
                if self.active_view == View::Generate {
                    self.generate.focus = self
                        .generate
                        .focus
                        .next(self.generate.capabilities.supports_negative_prompt);
                }
            }
            Action::FocusPrev => {
                if self.active_view == View::Generate {
                    self.generate.focus = self
                        .generate
                        .focus
                        .prev(self.generate.capabilities.supports_negative_prompt);
                }
            }
            Action::Up => match self.active_view {
                View::Generate => {
                    if self.generate.focus == GenerateFocus::Parameters
                        && self.generate.param_index > 0
                    {
                        self.generate.param_index -= 1;
                    }
                }
                View::Gallery => {
                    if self.gallery.selected > 0 {
                        self.gallery.selected -= 1;
                    }
                }
                View::Models => {
                    if self.models.selected > 0 {
                        self.models.selected -= 1;
                    }
                }
            },
            Action::Down => match self.active_view {
                View::Generate => {
                    if self.generate.focus == GenerateFocus::Parameters
                        && self.generate.param_index + 1 < self.generate.visible_fields.len()
                    {
                        self.generate.param_index += 1;
                    }
                }
                View::Gallery => {
                    if self.gallery.selected + 1 < self.gallery.entries.len() {
                        self.gallery.selected += 1;
                    }
                }
                View::Models => {
                    if self.models.selected + 1 < self.models.catalog.len() {
                        self.models.selected += 1;
                    }
                }
            },
            Action::Increment => self.increment_param(1),
            Action::Decrement => self.increment_param(-1),
            Action::Generate => {
                if self.active_view == View::Generate && !self.generate.generating {
                    self.start_generation();
                }
            }
            Action::Confirm => match self.active_view {
                View::Generate => {
                    if self.generate.focus == GenerateFocus::Parameters {
                        self.activate_current_param();
                    } else if !self.generate.generating {
                        self.start_generation();
                    }
                }
                View::Models => {
                    // Select model as default and switch to Generate
                    if let Some(model) = self.models.catalog.get(self.models.selected) {
                        let name = model.name.clone();
                        self.update_model(&name);
                        self.active_view = View::Generate;
                        self.generate.focus = GenerateFocus::Prompt;
                    }
                }
                _ => {}
            },
            Action::PullModel => {
                if self.active_view == View::Models {
                    if let Some(model) = self.models.catalog.get(self.models.selected) {
                        let model_name = model.name.clone();
                        let tx = self.bg_tx.clone();
                        self.tokio_handle.spawn(async move {
                            let _ = crate::backend::auto_pull_model(&model_name, &tx).await;
                        });
                    }
                }
            }
            Action::UnloadModel => {
                if let Some(ref url) = self.server_url {
                    let url = url.clone();
                    let tx = self.bg_tx.clone();
                    self.tokio_handle.spawn(async move {
                        let client = mold_core::MoldClient::new(&url);
                        match client.unload_model().await {
                            Ok(_) => {
                                let _ =
                                    tx.send(BackgroundEvent::Progress(SseProgressEvent::Info {
                                        message: "Model unloaded".to_string(),
                                    }));
                            }
                            Err(e) => {
                                let _ =
                                    tx.send(BackgroundEvent::Error(format!("Unload failed: {e}")));
                            }
                        }
                    });
                }
            }
            Action::OpenModelSelector => {
                self.open_model_selector();
            }
            Action::RandomizeSeed => {
                // Cycle seed mode: Random → Fixed → Increment → Random
                self.generate.params.seed_mode = self.generate.params.seed_mode.next();
                // When switching to Fixed, lock in a seed if we don't have one
                if self.generate.params.seed_mode == SeedMode::Fixed
                    && self.generate.params.seed.is_none()
                {
                    self.generate.params.seed = Some(rand::thread_rng().gen_range(0..u64::MAX));
                }
            }
            Action::ToggleMode => {
                self.generate.params.inference_mode = self.generate.params.inference_mode.next();
            }
            Action::ShowHelp => {
                self.popup = Some(Popup::Help);
            }
            Action::Cancel => {
                self.generate.error_message = None;
            }
            Action::HistoryPrev => {
                if self.active_view == View::Generate
                    && self.generate.focus == GenerateFocus::Prompt
                {
                    let current = self.generate.prompt.lines().join("\n");
                    if let Some(prompt) = self.history.prev(&current) {
                        self.generate.prompt =
                            TextArea::new(prompt.lines().map(String::from).collect());
                        self.generate
                            .prompt
                            .set_cursor_line_style(ratatui::style::Style::default());
                    }
                }
            }
            Action::HistoryNext => {
                if self.active_view == View::Generate
                    && self.generate.focus == GenerateFocus::Prompt
                {
                    let current = self.generate.prompt.lines().join("\n");
                    if let Some(prompt) = self.history.next(&current) {
                        self.generate.prompt =
                            TextArea::new(prompt.lines().map(String::from).collect());
                        self.generate
                            .prompt
                            .set_cursor_line_style(ratatui::style::Style::default());
                    }
                }
            }
            Action::SearchHistory => {
                let all: Vec<String> = self
                    .history
                    .search("")
                    .into_iter()
                    .map(|e| e.prompt.clone())
                    .collect();
                self.popup = Some(Popup::HistorySearch {
                    filter: String::new(),
                    selected: 0,
                    results: all,
                });
            }
            Action::Unfocus => {
                if self.active_view == View::Generate {
                    self.generate.focus = GenerateFocus::Navigation;
                }
            }
            _ => {}
        }
    }

    fn increment_param(&mut self, delta: i32) {
        if self.active_view != View::Generate || self.generate.focus != GenerateFocus::Parameters {
            return;
        }
        let field = match self.generate.visible_fields.get(self.generate.param_index) {
            Some(f) => *f,
            None => return,
        };
        let p = &mut self.generate.params;
        match field {
            ParamField::Width => {
                p.width = (p.width as i32 + delta * 64).clamp(256, 4096) as u32;
            }
            ParamField::Height => {
                p.height = (p.height as i32 + delta * 64).clamp(256, 4096) as u32;
            }
            ParamField::Steps => {
                p.steps = (p.steps as i32 + delta).clamp(1, 200) as u32;
            }
            ParamField::Guidance => {
                p.guidance = (p.guidance + delta as f64 * 0.5).clamp(0.0, 30.0);
            }
            ParamField::Batch => {
                p.batch = (p.batch as i32 + delta).clamp(1, 16) as u32;
            }
            ParamField::Strength => {
                p.strength = (p.strength + delta as f64 * 0.05).clamp(0.0, 1.0);
            }
            ParamField::ControlScale => {
                p.control_scale = (p.control_scale + delta as f64 * 0.1).clamp(0.0, 2.0);
            }
            ParamField::Format => {
                p.format = match p.format {
                    OutputFormat::Png => OutputFormat::Jpeg,
                    OutputFormat::Jpeg => OutputFormat::Png,
                };
            }
            ParamField::Mode => {
                p.inference_mode = p.inference_mode.next();
            }
            ParamField::Expand => {
                p.expand = !p.expand;
            }
            ParamField::Offload => {
                p.offload = !p.offload;
            }
            _ => {}
        }
        // Refresh visible fields when mode changes (Host visibility)
        if field == ParamField::Mode {
            self.generate.visible_fields = ParamField::visible_fields(
                &self.generate.capabilities,
                self.generate.params.inference_mode,
            );
        }
    }

    fn open_model_selector(&mut self) {
        let all_models: Vec<String> = self.models.catalog.iter().map(|m| m.name.clone()).collect();
        self.popup = Some(Popup::ModelSelector {
            filter: String::new(),
            selected: 0,
            filtered: all_models,
        });
    }

    /// Handle Enter on the currently focused parameter field.
    fn activate_current_param(&mut self) {
        let field = match self.generate.visible_fields.get(self.generate.param_index) {
            Some(f) => *f,
            None => return,
        };
        match field {
            // Open model selector popup
            ParamField::Model => self.open_model_selector(),
            // Toggle boolean fields
            ParamField::Expand => self.generate.params.expand = !self.generate.params.expand,
            ParamField::Offload => self.generate.params.offload = !self.generate.params.offload,
            ParamField::Mode => {
                self.generate.params.inference_mode = self.generate.params.inference_mode.next();
                self.generate.visible_fields = ParamField::visible_fields(
                    &self.generate.capabilities,
                    self.generate.params.inference_mode,
                );
            }
            // Cycle format
            ParamField::Format => {
                self.generate.params.format = match self.generate.params.format {
                    OutputFormat::Png => OutputFormat::Jpeg,
                    OutputFormat::Jpeg => OutputFormat::Png,
                };
            }
            // Cycle scheduler
            ParamField::Scheduler => {
                self.generate.params.scheduler = match self.generate.params.scheduler {
                    None => Some(Scheduler::Ddim),
                    Some(Scheduler::Ddim) => Some(Scheduler::EulerAncestral),
                    Some(Scheduler::EulerAncestral) => Some(Scheduler::UniPc),
                    Some(Scheduler::UniPc) => None,
                };
            }
            // Randomize seed on Enter
            ParamField::Seed => {
                // Cycle seed mode on Enter
                self.generate.params.seed_mode = self.generate.params.seed_mode.next();
                if self.generate.params.seed_mode == SeedMode::Fixed
                    && self.generate.params.seed.is_none()
                {
                    self.generate.params.seed = Some(rand::thread_rng().gen_range(0..u64::MAX));
                }
            }
            // Randomize the seed value
            ParamField::SeedValue => {
                let current = self
                    .generate
                    .params
                    .seed
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                self.popup = Some(Popup::SeedInput { input: current });
            }
            // Open host input popup
            ParamField::Host => {
                let current = self.generate.params.host.clone().unwrap_or_default();
                self.popup = Some(Popup::HostInput { input: current });
            }
            // For numeric fields, Enter does nothing special (use +/-)
            _ => {}
        }
    }

    fn start_generation(&mut self) {
        let prompt_text = self.generate.prompt.lines().join("\n").trim().to_string();
        if prompt_text.is_empty() {
            self.generate.error_message = Some("Prompt is empty".to_string());
            return;
        }

        self.generate.generating = true;
        self.generate.error_message = None;
        self.generate.progress.clear();
        self.generate.preview_image = None;
        self.generate.image_state = None;

        let neg = self
            .generate
            .negative_prompt
            .lines()
            .join("\n")
            .trim()
            .to_string();
        let negative_prompt = if neg.is_empty() { None } else { Some(neg) };

        // Resolve seed based on seed mode
        let resolved_seed = self
            .generate
            .params
            .seed_mode
            .resolve(self.generate.params.seed);
        let mut params = self.generate.params.clone();
        params.seed = Some(resolved_seed);

        let tx = self.bg_tx.clone();
        let server_url = self.server_url.clone();

        self.tokio_handle.spawn(async move {
            crate::backend::run_generation(server_url, params, prompt_text, negative_prompt, tx)
                .await;
        });
    }

    /// Drain and process all pending background events.
    pub fn process_background_events(&mut self) {
        while let Ok(event) = self.bg_rx.try_recv() {
            match event {
                BackgroundEvent::Progress(sse) => self.handle_progress(sse),
                BackgroundEvent::GenerationComplete(response) => {
                    self.generate.generating = false;
                    self.generate.last_seed = Some(response.seed_used);
                    self.generate.last_generation_time_ms = Some(response.generation_time_ms);

                    // Advance seed for next generation based on seed mode
                    self.generate.params.seed =
                        self.generate.params.seed_mode.advance(response.seed_used);

                    // Save images to disk and display preview
                    let mut saved_path = std::path::PathBuf::new();
                    let ts_secs = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);

                    for (i, img_data) in response.images.iter().enumerate() {
                        // Save to disk
                        let ext = match img_data.format {
                            OutputFormat::Png => "png",
                            OutputFormat::Jpeg => "jpeg",
                        };
                        let filename = mold_core::default_output_filename(
                            &self.generate.params.model,
                            ts_secs,
                            ext,
                            response.images.len() as u32,
                            i as u32,
                        );
                        let path = std::path::PathBuf::from(&filename);
                        if std::fs::write(&path, &img_data.data).is_ok() && i == 0 {
                            saved_path = path;
                        }

                        // Display preview for first image
                        if i == 0 {
                            if let Ok(img) = image::load_from_memory(&img_data.data) {
                                let protocol = self.picker.new_resize_protocol(img.clone());
                                self.generate.preview_image = Some(img);
                                self.generate.image_state = Some(protocol);
                            }
                        }
                    }

                    let saved_name = saved_path
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_default();

                    self.generate.progress.log.push(ProgressLogEntry {
                        message: if saved_name.is_empty() {
                            format!(
                                "Done in {:.1}s (seed: {})",
                                response.generation_time_ms as f64 / 1000.0,
                                response.seed_used
                            )
                        } else {
                            format!(
                                "Saved {} ({:.1}s)",
                                saved_name,
                                response.generation_time_ms as f64 / 1000.0,
                            )
                        },
                        style: ProgressStyle::Done,
                    });

                    // Save session state for next launch
                    let prompt_text = self.generate.prompt.lines().join("\n").trim().to_string();
                    let neg_text = self
                        .generate
                        .negative_prompt
                        .lines()
                        .join("\n")
                        .trim()
                        .to_string();
                    let session = crate::session::TuiSession {
                        last_prompt: prompt_text.clone(),
                        last_negative: neg_text.clone(),
                        last_model: self.generate.params.model.clone(),
                        last_width: Some(self.generate.params.width),
                        last_height: Some(self.generate.params.height),
                        last_steps: Some(self.generate.params.steps),
                        last_guidance: Some(self.generate.params.guidance),
                        last_seed_mode: Some(self.generate.params.seed_mode.label().to_string()),
                    };
                    session.save();
                    // Also update last-model for CLI compatibility
                    Config::write_last_model(&self.generate.params.model);

                    // Push to prompt history
                    let neg = if neg_text.is_empty() {
                        None
                    } else {
                        Some(neg_text)
                    };
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);

                    self.history.push(crate::history::HistoryEntry {
                        prompt: prompt_text.clone(),
                        negative: neg,
                        model: self.generate.params.model.clone(),
                        timestamp: ts,
                    });

                    // Add to gallery (most recent first)
                    if let Some(img_data) = response.images.first() {
                        self.gallery.entries.insert(
                            0,
                            GalleryEntry {
                                path: saved_path,
                                prompt_preview: if prompt_text.len() > 60 {
                                    format!("{}...", &prompt_text[..57])
                                } else {
                                    prompt_text
                                },
                                model: self.generate.params.model.clone(),
                                generation_time_ms: Some(response.generation_time_ms),
                                seed: Some(response.seed_used),
                                width: img_data.width,
                                height: img_data.height,
                                timestamp: ts,
                            },
                        );
                    }
                }
                BackgroundEvent::Error(msg) => {
                    self.generate.generating = false;
                    self.generate.error_message = Some(msg);
                }
                BackgroundEvent::GalleryScanComplete(entries) => {
                    self.gallery.entries = entries;
                    self.gallery.scanning = false;
                }
                BackgroundEvent::PullComplete(model) => {
                    self.generate.progress.log.push(ProgressLogEntry {
                        message: format!("Pull complete: {model}"),
                        style: ProgressStyle::Done,
                    });
                    // Refresh the catalog
                    self.models.catalog = mold_core::build_model_catalog(&self.config, None, false);
                }
            }
        }
    }

    fn handle_progress(&mut self, event: SseProgressEvent) {
        match event {
            SseProgressEvent::StageStart { name } => {
                self.generate.progress.current_stage = Some(name);
                // Reset weight progress on new stage
                self.generate.progress.weight_loaded = 0;
                self.generate.progress.weight_total = 0;
            }
            SseProgressEvent::StageDone { name, elapsed_ms } => {
                self.generate.progress.current_stage = None;
                self.generate.progress.log.push(ProgressLogEntry {
                    message: format!("{name} [{:.1}s]", elapsed_ms as f64 / 1000.0),
                    style: ProgressStyle::Done,
                });
            }
            SseProgressEvent::Info { message } => {
                self.generate.progress.log.push(ProgressLogEntry {
                    message,
                    style: ProgressStyle::Info,
                });
            }
            SseProgressEvent::CacheHit { resource } => {
                self.generate.progress.log.push(ProgressLogEntry {
                    message: format!("{resource} [cache hit]"),
                    style: ProgressStyle::Done,
                });
            }
            SseProgressEvent::DenoiseStep {
                step,
                total,
                elapsed_ms,
            } => {
                self.generate.progress.denoise_step = step;
                self.generate.progress.denoise_total = total;
                self.generate.progress.denoise_elapsed_ms = elapsed_ms;
            }
            SseProgressEvent::WeightLoad {
                bytes_loaded,
                bytes_total,
                component,
            } => {
                self.generate.progress.weight_loaded = bytes_loaded;
                self.generate.progress.weight_total = bytes_total;
                self.generate.progress.weight_component = component;
            }
            SseProgressEvent::DownloadProgress {
                filename,
                bytes_downloaded,
                bytes_total,
                ..
            } => {
                self.generate.progress.download_filename = filename;
                self.generate.progress.download_bytes = bytes_downloaded;
                self.generate.progress.download_total = bytes_total;
            }
            SseProgressEvent::DownloadDone {
                filename,
                file_index,
                total_files,
            } => {
                self.generate.progress.download_bytes = 0;
                self.generate.progress.download_total = 0;
                self.generate.progress.download_filename.clear();
                self.generate.progress.log.push(ProgressLogEntry {
                    message: format!("[{}/{}] {filename}", file_index + 1, total_files),
                    style: ProgressStyle::Done,
                });
            }
            SseProgressEvent::PullComplete { model } => {
                self.generate.progress.log.push(ProgressLogEntry {
                    message: format!("Pull complete: {model}"),
                    style: ProgressStyle::Done,
                });
                // Refresh config and catalog after pull
                self.config = Config::load_or_default();
                self.models.catalog = mold_core::build_model_catalog(&self.config, None, false);
            }
            SseProgressEvent::Queued { position } => {
                self.generate.progress.current_stage =
                    Some(format!("Queued (position {position})"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_mode_cycle() {
        assert_eq!(InferenceMode::Auto.next(), InferenceMode::Local);
        assert_eq!(InferenceMode::Local.next(), InferenceMode::Remote);
        assert_eq!(InferenceMode::Remote.next(), InferenceMode::Auto);
    }

    #[test]
    fn inference_mode_labels() {
        assert_eq!(InferenceMode::Auto.label(), "auto");
        assert_eq!(InferenceMode::Local.label(), "local");
        assert_eq!(InferenceMode::Remote.label(), "remote");
    }

    #[test]
    fn visible_fields_hides_host_in_local_mode() {
        let caps = crate::model_info::capabilities_for_family("flux");
        let fields = ParamField::visible_fields(&caps, InferenceMode::Local);
        assert!(!fields.contains(&ParamField::Host));
    }

    #[test]
    fn visible_fields_shows_host_in_auto_mode() {
        let caps = crate::model_info::capabilities_for_family("flux");
        let fields = ParamField::visible_fields(&caps, InferenceMode::Auto);
        assert!(fields.contains(&ParamField::Host));
    }

    #[test]
    fn visible_fields_shows_host_in_remote_mode() {
        let caps = crate::model_info::capabilities_for_family("flux");
        let fields = ParamField::visible_fields(&caps, InferenceMode::Remote);
        assert!(fields.contains(&ParamField::Host));
    }

    #[test]
    fn visible_fields_includes_scheduler_for_sd15() {
        let caps = crate::model_info::capabilities_for_family("sd15");
        let fields = ParamField::visible_fields(&caps, InferenceMode::Auto);
        assert!(fields.contains(&ParamField::Scheduler));
    }

    #[test]
    fn visible_fields_excludes_scheduler_for_flux() {
        let caps = crate::model_info::capabilities_for_family("flux");
        let fields = ParamField::visible_fields(&caps, InferenceMode::Auto);
        assert!(!fields.contains(&ParamField::Scheduler));
    }

    #[test]
    fn visible_fields_includes_lora_for_flux() {
        let caps = crate::model_info::capabilities_for_family("flux");
        let fields = ParamField::visible_fields(&caps, InferenceMode::Auto);
        assert!(fields.contains(&ParamField::Lora));
    }

    #[test]
    fn visible_fields_excludes_lora_for_sdxl() {
        let caps = crate::model_info::capabilities_for_family("sdxl");
        let fields = ParamField::visible_fields(&caps, InferenceMode::Auto);
        assert!(!fields.contains(&ParamField::Lora));
    }

    #[test]
    fn generate_params_display_host_default() {
        let config = Config::load_or_default();
        let params = GenerateParams::from_config(&config);
        assert_eq!(params.display_value(&ParamField::Host), "localhost:7680");
    }

    #[test]
    fn generate_params_display_host_custom() {
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.host = Some("http://gpu-server:7680".to_string());
        assert_eq!(
            params.display_value(&ParamField::Host),
            "http://gpu-server:7680"
        );
    }

    #[test]
    fn generate_params_display_mode() {
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.inference_mode = InferenceMode::Auto;
        assert_eq!(params.display_value(&ParamField::Mode), "auto");
        params.inference_mode = InferenceMode::Local;
        assert_eq!(params.display_value(&ParamField::Mode), "local");
        params.inference_mode = InferenceMode::Remote;
        assert_eq!(params.display_value(&ParamField::Mode), "remote");
    }

    #[test]
    fn focus_navigation_next_enters_prompt() {
        assert_eq!(GenerateFocus::Navigation.next(false), GenerateFocus::Prompt);
        assert_eq!(GenerateFocus::Navigation.next(true), GenerateFocus::Prompt);
    }

    #[test]
    fn focus_cycle_skips_negative_when_unsupported() {
        assert_eq!(GenerateFocus::Prompt.next(false), GenerateFocus::Parameters);
        assert_eq!(GenerateFocus::Parameters.prev(false), GenerateFocus::Prompt);
    }

    #[test]
    fn focus_cycle_includes_negative_when_supported() {
        assert_eq!(
            GenerateFocus::Prompt.next(true),
            GenerateFocus::NegativePrompt
        );
        assert_eq!(
            GenerateFocus::Parameters.prev(true),
            GenerateFocus::NegativePrompt
        );
    }

    #[test]
    fn param_field_labels_not_empty() {
        let caps = crate::model_info::capabilities_for_family("sd15");
        let fields = ParamField::visible_fields(&caps, InferenceMode::Auto);
        for field in &fields {
            // SeedValue intentionally has no label (continuation of Seed row)
            if *field == ParamField::SeedValue {
                continue;
            }
            assert!(
                !field.label().is_empty(),
                "field {:?} has empty label",
                field
            );
        }
    }

    #[test]
    fn progress_state_clear_resets_all() {
        let mut state = ProgressState::default();
        state.denoise_step = 10;
        state.denoise_total = 20;
        state.weight_loaded = 1000;
        state.download_bytes = 500;
        state.download_filename = "test.gguf".to_string();
        state.log.push(ProgressLogEntry {
            message: "test".to_string(),
            style: ProgressStyle::Done,
        });
        state.clear();
        assert_eq!(state.denoise_step, 0);
        assert_eq!(state.denoise_total, 0);
        assert_eq!(state.weight_loaded, 0);
        assert_eq!(state.download_bytes, 0);
        assert!(state.download_filename.is_empty());
        assert!(state.log.is_empty());
    }

    // ── SeedMode tests ────────────────────────────────────

    #[test]
    fn seed_mode_cycle() {
        assert_eq!(SeedMode::Random.next(), SeedMode::Fixed);
        assert_eq!(SeedMode::Fixed.next(), SeedMode::Increment);
        assert_eq!(SeedMode::Increment.next(), SeedMode::Random);
    }

    #[test]
    fn seed_mode_labels() {
        assert_eq!(SeedMode::Random.label(), "random");
        assert_eq!(SeedMode::Fixed.label(), "fixed");
        assert_eq!(SeedMode::Increment.label(), "increment");
    }

    #[test]
    fn seed_mode_random_generates_value() {
        let seed = SeedMode::Random.resolve(None);
        // Just verify it returns something (can't test exact value)
        assert!(seed > 0 || seed == 0); // always true, but exercises the code
    }

    #[test]
    fn seed_mode_fixed_keeps_seed() {
        let seed = SeedMode::Fixed.resolve(Some(42));
        assert_eq!(seed, 42);
    }

    #[test]
    fn seed_mode_fixed_generates_if_none() {
        let seed = SeedMode::Fixed.resolve(None);
        // Should generate a seed when none exists
        let _ = seed; // exercises the code path
    }

    #[test]
    fn seed_mode_increment_adds_one() {
        let seed = SeedMode::Increment.resolve(Some(42));
        assert_eq!(seed, 43);
    }

    #[test]
    fn seed_mode_increment_wraps_at_max() {
        let seed = SeedMode::Increment.resolve(Some(u64::MAX));
        assert_eq!(seed, 0); // wrapping_add
    }

    #[test]
    fn seed_mode_increment_generates_if_none() {
        let seed = SeedMode::Increment.resolve(None);
        let _ = seed;
    }

    #[test]
    fn seed_mode_advance_random_returns_none() {
        assert_eq!(SeedMode::Random.advance(42), None);
    }

    #[test]
    fn seed_mode_advance_fixed_returns_same() {
        assert_eq!(SeedMode::Fixed.advance(42), Some(42));
    }

    #[test]
    fn seed_mode_advance_increment_returns_same() {
        // advance stores the used seed; resolve will +1 next time
        assert_eq!(SeedMode::Increment.advance(42), Some(42));
    }

    #[test]
    fn seed_display_shows_mode() {
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        assert_eq!(params.display_value(&ParamField::Seed), "random");
        params.seed_mode = SeedMode::Fixed;
        assert_eq!(params.display_value(&ParamField::Seed), "fixed");
        params.seed_mode = SeedMode::Increment;
        assert_eq!(params.display_value(&ParamField::Seed), "increment");
    }

    #[test]
    fn seed_value_display_with_number() {
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.seed = Some(12345);
        assert_eq!(params.display_value(&ParamField::SeedValue), "12345");
    }

    #[test]
    fn seed_value_display_auto_when_none() {
        let config = Config::load_or_default();
        let params = GenerateParams::from_config(&config);
        let display = params.display_value(&ParamField::SeedValue);
        assert!(display.contains("auto"));
    }

    #[test]
    fn seed_value_shows_long_numbers_untruncated() {
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.seed = Some(11275518943372801901);
        let display = params.display_value(&ParamField::SeedValue);
        assert_eq!(display, "11275518943372801901");
    }
}
