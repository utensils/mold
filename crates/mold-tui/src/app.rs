use anyhow::Result;
use crossterm::event::{Event as CrosstermEvent, KeyCode, KeyModifiers};
use mold_core::{
    Config, GenerateResponse, ModelInfoExtended, OutputFormat, Scheduler, SseProgressEvent,
};
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
    Prompt,
    NegativePrompt,
    Parameters,
}

impl GenerateFocus {
    pub fn next(self, has_negative: bool) -> Self {
        match self {
            Self::Prompt if has_negative => Self::NegativePrompt,
            Self::Prompt => Self::Parameters,
            Self::NegativePrompt => Self::Parameters,
            Self::Parameters => Self::Prompt,
        }
    }

    pub fn prev(self, has_negative: bool) -> Self {
        match self {
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
    Batch,
    Format,
    Mode,
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
    /// All fields in display order, filtering by model capabilities.
    pub fn visible_fields(caps: &ModelCapabilities) -> Vec<ParamField> {
        let mut fields = vec![
            ParamField::Model,
            ParamField::Width,
            ParamField::Height,
            ParamField::Steps,
            ParamField::Guidance,
            ParamField::Seed,
            ParamField::Batch,
            ParamField::Format,
            ParamField::Mode,
        ];
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
            Self::Batch => "Batch",
            Self::Format => "Format",
            Self::Mode => "Mode",
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

/// Generation parameters mirroring GenerateRequest fields.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    pub seed: Option<u64>,
    pub batch: u32,
    pub format: OutputFormat,
    pub scheduler: Option<Scheduler>,
    pub local_mode: bool,
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
            batch: 1,
            format: OutputFormat::Png,
            scheduler: None,
            local_mode: false,
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
            ParamField::Seed => self
                .seed
                .map(|s| s.to_string())
                .unwrap_or_else(|| "\u{27e8}random\u{27e9}".to_string()),
            ParamField::Batch => self.batch.to_string(),
            ParamField::Format => format!("{:?}", self.format).to_uppercase(),
            ParamField::Mode => {
                if self.local_mode {
                    "local".to_string()
                } else {
                    "remote".to_string()
                }
            }
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
}

impl App {
    pub fn new(host: Option<String>, local: bool, picker: Picker) -> Result<Self> {
        let config = Config::load_or_default();

        let server_url = if local {
            None
        } else {
            Some(host.unwrap_or_else(|| {
                std::env::var("MOLD_HOST")
                    .unwrap_or_else(|_| format!("http://localhost:{}", config.server_port))
            }))
        };

        let params = GenerateParams::from_config(&config);
        let family = family_for_model(&params.model, &config);
        let capabilities = capabilities_for_family(&family);
        let visible_fields = ParamField::visible_fields(&capabilities);

        let model_description = mold_core::manifest::find_manifest(&params.model)
            .and_then(|m| {
                let mc = config.resolved_model_config(&params.model);
                mc.description.or(Some(m.name.clone()))
            })
            .unwrap_or_default();

        let catalog = mold_core::build_model_catalog(&config, None, false);

        let (bg_tx, bg_rx) = mpsc::unbounded_channel();

        let mut prompt = TextArea::default();
        prompt.set_cursor_line_style(ratatui::style::Style::default());
        prompt.set_placeholder_text("Enter your prompt...");

        let mut negative_prompt = TextArea::default();
        negative_prompt.set_cursor_line_style(ratatui::style::Style::default());
        negative_prompt.set_placeholder_text("Negative prompt (what to avoid)...");

        Ok(Self {
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
        })
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
        self.generate.visible_fields = ParamField::visible_fields(&self.generate.capabilities);
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
                Some(Popup::Confirm { on_confirm, .. }) => match key.code {
                    KeyCode::Char('y') | KeyCode::Enter => {
                        let _action = on_confirm.clone();
                        self.popup = None;
                        // TODO: handle confirm action
                    }
                    _ => self.popup = None,
                },
                None => {}
            }
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
            Action::Confirm => {
                if self.active_view == View::Generate {
                    if self.generate.focus == GenerateFocus::Parameters {
                        // Enter on a param field: activate it
                        self.activate_current_param();
                    } else if !self.generate.generating {
                        // Enter in prompt/negative: generate
                        self.start_generation();
                    }
                }
            }
            Action::OpenModelSelector => {
                self.open_model_selector();
            }
            Action::RandomizeSeed => {
                self.generate.params.seed = None;
            }
            Action::ToggleMode => {
                self.generate.params.local_mode = !self.generate.params.local_mode;
            }
            Action::ShowHelp => {
                self.popup = Some(Popup::Help);
            }
            Action::Cancel => {
                self.generate.error_message = None;
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
                p.local_mode = !p.local_mode;
            }
            ParamField::Expand => {
                p.expand = !p.expand;
            }
            ParamField::Offload => {
                p.offload = !p.offload;
            }
            _ => {}
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
            ParamField::Mode => self.generate.params.local_mode = !self.generate.params.local_mode,
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
                use rand::Rng;
                self.generate.params.seed = Some(rand::thread_rng().gen_range(0..u64::MAX));
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

        let tx = self.bg_tx.clone();
        let server_url = self.server_url.clone();
        let params = self.generate.params.clone();

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

                    if let Some(img_data) = response.images.first() {
                        if let Ok(img) = image::load_from_memory(&img_data.data) {
                            let protocol = self.picker.new_resize_protocol(img.clone());
                            self.generate.preview_image = Some(img);
                            self.generate.image_state = Some(protocol);
                        }
                    }

                    self.generate.progress.log.push(ProgressLogEntry {
                        message: format!(
                            "Done in {:.1}s (seed: {})",
                            response.generation_time_ms as f64 / 1000.0,
                            response.seed_used
                        ),
                        style: ProgressStyle::Done,
                    });
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
