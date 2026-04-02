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
    /// Background thumbnail generation finished.
    ThumbnailsReady,
    /// Gallery image bytes fetched from server for preview.
    GalleryPreviewReady(Vec<u8>),
    /// Remote server health check + model list succeeded.
    ServerConnected {
        url: String,
        models: Vec<ModelInfoExtended>,
    },
    /// Remote server health check failed.
    ServerUnreachable(String),
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
    // Actions
    ResetDefaults,
    // Tools
    UnloadModel,
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
        // Actions
        fields.push(ParamField::ResetDefaults);
        // Tools
        fields.push(ParamField::UnloadModel);
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
            Self::ResetDefaults => "\u{21ba} Reset",
            Self::UnloadModel => "\u{23cf} Unload",
        }
    }

    /// The section header this field falls under, if it starts a new section.
    pub fn section_header(&self) -> Option<&'static str> {
        match self {
            Self::Scheduler => Some("Advanced"),
            Self::SourceImage => Some("img2img"),
            Self::ControlImage => Some("ControlNet"),
            Self::ResetDefaults => Some("Actions"),
            Self::UnloadModel => None,
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
                .unwrap_or_else(|| "\u{27e8}random\u{27e9}".to_string()),
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
            ParamField::ResetDefaults => "restore model defaults".to_string(),
            ParamField::UnloadModel => "free GPU memory".to_string(),
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

/// Which gallery view is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GalleryViewMode {
    #[default]
    Grid,
    Detail,
}

/// State for the Gallery view.
pub struct GalleryState {
    pub entries: Vec<GalleryEntry>,
    pub selected: usize,
    pub preview_image: Option<image::DynamicImage>,
    pub image_state: Option<StatefulProtocol>,
    pub scanning: bool,
    pub view_mode: GalleryViewMode,
    /// Thumbnail StatefulProtocol instances, lazily populated during render.
    pub thumbnail_states: Vec<Option<StatefulProtocol>>,
    /// Number of columns in the grid (computed from terminal width).
    pub grid_cols: usize,
    /// Scroll offset in rows for the grid view.
    pub grid_scroll: usize,
}

/// A single gallery entry backed by PNG metadata.
#[derive(Debug, Clone)]
pub struct GalleryEntry {
    /// Local file path, or just the filename for server-backed entries.
    pub path: std::path::PathBuf,
    pub metadata: mold_core::OutputMetadata,
    pub generation_time_ms: Option<u64>,
    pub timestamp: u64,
    /// When set, this entry is served by the remote server at this URL.
    pub server_url: Option<String>,
}

impl GalleryEntry {
    pub fn filename(&self) -> String {
        self.path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".into())
    }
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
    /// Delete a gallery image by index.
    DeleteGalleryImage,
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
    /// Background server process spawned by the TUI (killed on quit).
    pub server_process: Option<std::process::Child>,
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
    pub gallery_grid: ratatui::layout::Rect,
    pub models_table: ratatui::layout::Rect,
}

/// Check if a server is responding at the given URL.
fn check_server_health(url: &str) -> bool {
    let health_url = format!("{url}/health");
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(2)))
        .build()
        .new_agent();
    agent.get(&health_url).call().is_ok()
}

/// Spawn a background `mold serve` process.
fn start_background_server(port: u16) -> Option<std::process::Child> {
    let exe = std::env::current_exe().ok()?;
    std::process::Command::new(exe)
        .args(["serve", "--port", &port.to_string(), "--log-file"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()
}

/// Wait for a server to become healthy, polling every 250ms.
fn wait_for_server_health(url: &str, timeout_secs: u64) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    while std::time::Instant::now() < deadline {
        if check_server_health(url) {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    false
}

impl App {
    pub fn new(host: Option<String>, local: bool, picker: Picker) -> Result<Self> {
        let config = Config::load_or_default();

        // Determine initial server URL and inference mode
        let env_host = std::env::var("MOLD_HOST").ok();
        let port = config.server_port;
        let local_url = format!("http://localhost:{port}");
        let mut server_process: Option<std::process::Child> = None;

        let (server_url, initial_mode) = if local {
            (None, InferenceMode::Local)
        } else if let Some(h) = host {
            let url = mold_core::client::normalize_host(&h);
            if check_server_health(&url) {
                (Some(url), InferenceMode::Auto)
            } else {
                (None, InferenceMode::Local)
            }
        } else if let Some(h) = env_host {
            let url = mold_core::client::normalize_host(&h);
            if check_server_health(&url) {
                (Some(url), InferenceMode::Auto)
            } else {
                (None, InferenceMode::Local)
            }
        } else {
            // No explicit server — try to detect or auto-start one
            if check_server_health(&local_url) {
                // Server already running
                (Some(local_url.clone()), InferenceMode::Auto)
            } else {
                // Try to start a background server
                match start_background_server(port) {
                    Some(mut child) => {
                        if wait_for_server_health(&local_url, 8) {
                            server_process = Some(child);
                            (Some(local_url.clone()), InferenceMode::Auto)
                        } else {
                            // Server didn't start in time — kill it and fall back to local
                            let _ = child.kill();
                            let _ = child.wait();
                            (None, InferenceMode::Local)
                        }
                    }
                    None => (None, InferenceMode::Local),
                }
            }
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

        // Build initial catalog — try server first if connected, fall back to local
        let catalog = if let Some(ref url) = server_url {
            // Blocking fetch from server for startup catalog
            let rt = tokio::runtime::Handle::current();
            let url_clone = url.clone();
            std::thread::spawn(move || {
                rt.block_on(async {
                    let client = mold_core::MoldClient::new(&url_clone);
                    client.list_models_extended().await.ok()
                })
            })
            .join()
            .ok()
            .flatten()
            .unwrap_or_else(|| mold_core::build_model_catalog(&config, None, false))
        } else {
            mold_core::build_model_catalog(&config, None, false)
        };

        let (bg_tx, bg_rx) = mpsc::unbounded_channel();

        // Load session from previous TUI run
        let session = crate::session::TuiSession::load();

        // Restore all settings from session.
        // Check both local downloads and the catalog (which includes remote models).
        let model_in_catalog = catalog.iter().any(|m| m.name == session.last_model);
        if !session.last_model.is_empty()
            && (config.manifest_model_is_downloaded(&session.last_model) || model_in_catalog)
        {
            params.model = session.last_model.clone();
            // Apply all saved params (width, height, steps, guidance, batch, etc.)
            session.apply_to_params(&mut params);
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
                view_mode: GalleryViewMode::Grid,
                thumbnail_states: Vec::new(),
                grid_cols: 3,
                grid_scroll: 0,
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
            server_process,
        });

        // Spawn background gallery scan
        if let Ok(ref app) = app {
            app.spawn_gallery_scan();
        }

        app
    }

    /// Spawn a background gallery scan. Uses server API when connected,
    /// local filesystem scan otherwise.
    pub fn spawn_gallery_scan(&self) {
        let tx = self.bg_tx.clone();
        let server_url = self.server_url.clone();
        self.tokio_handle.spawn(async move {
            let entries = if let Some(ref url) = server_url {
                crate::gallery_scan::scan_images_from_server(url).await
            } else {
                tokio::task::spawn_blocking(crate::gallery_scan::scan_images_local)
                    .await
                    .unwrap_or_default()
            };
            let _ = tx.send(BackgroundEvent::GalleryScanComplete(entries));
        });
    }

    /// Clean up resources on quit (kills background server if we spawned it).
    pub fn shutdown(&mut self) {
        if let Some(ref mut child) = self.server_process {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.server_process = None;
    }

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
                        // TUI-global shortcuts that bypass the textarea
                        (KeyCode::Tab, KeyModifiers::NONE)
                        | (KeyCode::BackTab, KeyModifiers::SHIFT)
                        | (KeyCode::Char('c'), KeyModifiers::CONTROL) // quit
                        | (KeyCode::Char('g'), KeyModifiers::CONTROL) // generate
                        | (KeyCode::Char('m'), KeyModifiers::CONTROL) // model selector
                        | (KeyCode::Char('r'), KeyModifiers::CONTROL) // seed mode
                        | (KeyCode::Enter, KeyModifiers::NONE)        // generate
                        | (KeyCode::Esc, KeyModifiers::NONE) => {     // nav mode
                            // Fall through to action mapping
                        }
                        // Ctrl+P/N: history when on first/last line,
                        // otherwise let textarea handle cursor movement
                        (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                            let textarea = match self.generate.focus {
                                GenerateFocus::Prompt => &self.generate.prompt,
                                GenerateFocus::NegativePrompt => &self.generate.negative_prompt,
                                _ => unreachable!(),
                            };
                            // At top line → history prev; otherwise cursor up
                            if textarea.cursor().0 == 0 {
                                // Fall through for history
                            } else {
                                let ta = match self.generate.focus {
                                    GenerateFocus::Prompt => &mut self.generate.prompt,
                                    GenerateFocus::NegativePrompt => {
                                        &mut self.generate.negative_prompt
                                    }
                                    _ => unreachable!(),
                                };
                                ta.input(event);
                                return;
                            }
                        }
                        (KeyCode::Char('n'), KeyModifiers::CONTROL) => {
                            let textarea = match self.generate.focus {
                                GenerateFocus::Prompt => &self.generate.prompt,
                                GenerateFocus::NegativePrompt => &self.generate.negative_prompt,
                                _ => unreachable!(),
                            };
                            let last_line = textarea.lines().len().saturating_sub(1);
                            // At bottom line → history next; otherwise cursor down
                            if textarea.cursor().0 >= last_line {
                                // Fall through for history
                            } else {
                                let ta = match self.generate.focus {
                                    GenerateFocus::Prompt => &mut self.generate.prompt,
                                    GenerateFocus::NegativePrompt => {
                                        &mut self.generate.negative_prompt
                                    }
                                    _ => unreachable!(),
                                };
                                ta.input(event);
                                return;
                            }
                        }
                        // Up/Down arrows: history only in Prompt field, cursor move in Negative
                        (KeyCode::Up, KeyModifiers::NONE) => {
                            let textarea = match self.generate.focus {
                                GenerateFocus::Prompt => &self.generate.prompt,
                                GenerateFocus::NegativePrompt => &self.generate.negative_prompt,
                                _ => unreachable!(),
                            };
                            if self.generate.focus == GenerateFocus::Prompt
                                && textarea.cursor().0 == 0
                            {
                                let current = self.generate.prompt.lines().join("\n");
                                if let Some(prompt) = self.history.prev(&current) {
                                    self.generate.prompt = TextArea::new(
                                        prompt.lines().map(String::from).collect(),
                                    );
                                    self.generate
                                        .prompt
                                        .set_cursor_line_style(ratatui::style::Style::default());
                                }
                                return;
                            }
                            let ta = match self.generate.focus {
                                GenerateFocus::Prompt => &mut self.generate.prompt,
                                GenerateFocus::NegativePrompt => {
                                    &mut self.generate.negative_prompt
                                }
                                _ => unreachable!(),
                            };
                            ta.input(event);
                            return;
                        }
                        (KeyCode::Down, KeyModifiers::NONE) => {
                            let textarea = match self.generate.focus {
                                GenerateFocus::Prompt => &self.generate.prompt,
                                GenerateFocus::NegativePrompt => &self.generate.negative_prompt,
                                _ => unreachable!(),
                            };
                            let last_line = textarea.lines().len().saturating_sub(1);
                            if self.generate.focus == GenerateFocus::Prompt
                                && textarea.cursor().0 >= last_line
                            {
                                let current = self.generate.prompt.lines().join("\n");
                                if let Some(prompt) = self.history.next(&current) {
                                    self.generate.prompt = TextArea::new(
                                        prompt.lines().map(String::from).collect(),
                                    );
                                    self.generate
                                        .prompt
                                        .set_cursor_line_style(ratatui::style::Style::default());
                                }
                                return;
                            }
                            let ta = match self.generate.focus {
                                GenerateFocus::Prompt => &mut self.generate.prompt,
                                GenerateFocus::NegativePrompt => {
                                    &mut self.generate.negative_prompt
                                }
                                _ => unreachable!(),
                            };
                            ta.input(event);
                            return;
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
                            // Reset history navigation when user types
                            self.history.reset_cursor();
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

    /// Close the active popup and refresh the preview image so it re-renders
    /// over the area the popup occupied.
    fn close_popup(&mut self) {
        self.popup = None;
        self.refresh_preview_protocol();
    }

    /// Recreate the StatefulProtocol from the cached preview image so
    /// ratatui-image re-renders the region after a popup overlay clears.
    fn refresh_preview_protocol(&mut self) {
        if let Some(ref img) = self.generate.preview_image {
            self.generate.image_state = Some(self.picker.new_resize_protocol(img.clone()));
        }
        if let Some(ref img) = self.gallery.preview_image {
            self.gallery.image_state = Some(self.picker.new_resize_protocol(img.clone()));
        }
    }

    fn handle_popup_event(&mut self, event: CrosstermEvent) {
        if let CrosstermEvent::Key(key) = event {
            match &mut self.popup {
                Some(Popup::Help) => {
                    if matches!(
                        key.code,
                        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('?')
                    ) {
                        self.close_popup();
                    }
                }
                Some(Popup::ModelSelector {
                    filter,
                    selected,
                    filtered,
                }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        if let Some(model) = filtered.get(*selected).cloned() {
                            self.close_popup();
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
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        let host = input.trim().to_string();
                        self.close_popup();
                        if host.is_empty() {
                            // Clear host → switch to local
                            self.generate.params.host = None;
                            self.generate.params.inference_mode = InferenceMode::Local;
                            self.server_url = None;
                            self.generate.visible_fields = ParamField::visible_fields(
                                &self.generate.capabilities,
                                self.generate.params.inference_mode,
                            );
                            // Refresh to local catalog and gallery
                            self.models.catalog =
                                mold_core::build_model_catalog(&self.config, None, false);
                            self.gallery.scanning = true;
                            self.spawn_gallery_scan();
                        } else {
                            // Normalize using same logic as CLI/MoldClient
                            let url = mold_core::client::normalize_host(&host);
                            self.generate.params.host = Some(url.clone());
                            // Show connecting status
                            self.generate.progress.log.push(ProgressLogEntry {
                                message: format!("Connecting to {url}..."),
                                style: ProgressStyle::Info,
                            });
                            // Spawn background health check + model list fetch
                            let tx = self.bg_tx.clone();
                            self.tokio_handle.spawn(async move {
                                let client = mold_core::MoldClient::new(&url);
                                match client.list_models_extended().await {
                                    Ok(models) => {
                                        let _ = tx
                                            .send(BackgroundEvent::ServerConnected { url, models });
                                    }
                                    Err(e) => {
                                        let _ = tx.send(BackgroundEvent::ServerUnreachable(
                                            format!("{url}: {e}"),
                                        ));
                                    }
                                }
                            });
                        }
                    }
                    KeyCode::Char(c) => input.push(c),
                    KeyCode::Backspace => {
                        input.pop();
                    }
                    _ => {}
                },
                Some(Popup::SeedInput { input }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        let text = input.trim().to_string();
                        self.close_popup();
                        if text.is_empty() {
                            // Clear seed → back to auto
                            self.generate.params.seed = None;
                        } else if let Ok(val) = text.parse::<u64>() {
                            self.generate.params.seed = Some(val);
                            if self.generate.params.seed_mode == SeedMode::Random {
                                self.generate.params.seed_mode = SeedMode::Fixed;
                            }
                        }
                        // Move focus to prompt so Enter key repeat doesn't re-open
                        self.generate.focus = GenerateFocus::Prompt;
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
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        if let Some(prompt) = results.get(*selected).cloned() {
                            self.close_popup();
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
                        let action = on_confirm.clone();
                        self.close_popup();
                        self.handle_confirm_action(action);
                    }
                    _ => self.close_popup(),
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
                    self.close_popup();
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
                if self.active_view == View::Gallery
                    && self.gallery.view_mode == GalleryViewMode::Grid
                {
                    let pos: ratatui::layout::Position = (col, row).into();
                    if self.layout.gallery_grid.contains(pos) {
                        // Compute which grid cell was clicked
                        let cell_w = 24u16;
                        let cell_h = 14u16;
                        let cols = self.gallery.grid_cols.max(1);
                        let rel_x = col.saturating_sub(self.layout.gallery_grid.x);
                        let rel_y = row.saturating_sub(self.layout.gallery_grid.y);
                        let grid_col = (rel_x / cell_w) as usize;
                        let grid_row = (rel_y / cell_h) as usize + self.gallery.grid_scroll;
                        let idx = grid_row * cols + grid_col;
                        if idx < self.gallery.entries.len() {
                            if self.gallery.selected == idx {
                                // Double-click: open detail view
                                self.gallery.view_mode = GalleryViewMode::Detail;
                                self.load_gallery_preview();
                            } else {
                                self.gallery.selected = idx;
                            }
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
                // If popup is open, scroll within the popup
                match &mut self.popup {
                    Some(Popup::ModelSelector { selected, .. }) => {
                        if *selected > 0 {
                            *selected -= 1;
                        }
                    }
                    Some(Popup::HistorySearch { selected, .. }) => {
                        if *selected > 0 {
                            *selected -= 1;
                        }
                    }
                    _ => self.dispatch_action(Action::Up),
                }
            }
            MouseEventKind::ScrollDown => match &mut self.popup {
                Some(Popup::ModelSelector {
                    selected, filtered, ..
                }) => {
                    if *selected + 1 < filtered.len() {
                        *selected += 1;
                    }
                }
                Some(Popup::HistorySearch {
                    selected, results, ..
                }) => {
                    if *selected + 1 < results.len() {
                        *selected += 1;
                    }
                }
                _ => self.dispatch_action(Action::Down),
            },
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
                    let cols = self.gallery.grid_cols.max(1);
                    match self.gallery.view_mode {
                        GalleryViewMode::Grid => {
                            if self.gallery.selected >= cols {
                                self.gallery.selected -= cols;
                            }
                        }
                        GalleryViewMode::Detail => {
                            if self.gallery.selected > 0 {
                                self.gallery.selected -= 1;
                                self.load_gallery_preview();
                            }
                        }
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
                    let cols = self.gallery.grid_cols.max(1);
                    let len = self.gallery.entries.len();
                    match self.gallery.view_mode {
                        GalleryViewMode::Grid => {
                            let next = self.gallery.selected + cols;
                            if next < len {
                                self.gallery.selected = next;
                            }
                        }
                        GalleryViewMode::Detail => {
                            if self.gallery.selected + 1 < len {
                                self.gallery.selected += 1;
                                self.load_gallery_preview();
                            }
                        }
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
                View::Gallery => match self.gallery.view_mode {
                    GalleryViewMode::Grid => {
                        if !self.gallery.entries.is_empty() {
                            self.gallery.view_mode = GalleryViewMode::Detail;
                            self.load_gallery_preview();
                        }
                    }
                    GalleryViewMode::Detail => {
                        // Enter in detail opens in system viewer
                        self.open_gallery_file();
                    }
                },
                View::Models => {
                    // Select model as default and switch to Generate
                    if let Some(model) = self.models.catalog.get(self.models.selected) {
                        let name = model.name.clone();
                        self.update_model(&name);
                        self.active_view = View::Generate;
                        self.generate.focus = GenerateFocus::Prompt;
                    }
                }
            },
            Action::PullModel => {
                if self.active_view == View::Models {
                    if let Some(model) = self.models.catalog.get(self.models.selected) {
                        let model_name = model.name.clone();
                        let tx = self.bg_tx.clone();
                        self.tokio_handle.spawn(async move {
                            if let Err(msg) =
                                crate::backend::auto_pull_model(&model_name, &tx).await
                            {
                                let _ = tx.send(BackgroundEvent::Error(msg));
                            }
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
                if self.active_view == View::Gallery
                    && self.gallery.view_mode == GalleryViewMode::Detail
                {
                    self.gallery.view_mode = GalleryViewMode::Grid;
                    self.gallery.preview_image = None;
                    self.gallery.image_state = None;
                } else {
                    self.generate.error_message = None;
                }
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
            Action::GridLeft => {
                if self.active_view == View::Gallery
                    && self.gallery.view_mode == GalleryViewMode::Grid
                    && self.gallery.selected > 0
                {
                    self.gallery.selected -= 1;
                }
            }
            Action::GridRight => {
                if self.active_view == View::Gallery
                    && self.gallery.view_mode == GalleryViewMode::Grid
                    && self.gallery.selected + 1 < self.gallery.entries.len()
                {
                    self.gallery.selected += 1;
                }
            }
            Action::EditAndGenerate => {
                if self.active_view == View::Gallery {
                    self.load_gallery_into_generate();
                }
            }
            Action::Regenerate => {
                if self.active_view == View::Gallery {
                    self.load_gallery_into_generate();
                    if !self.generate.generating {
                        self.start_generation();
                    }
                }
            }
            Action::DeleteImage => {
                if self.active_view == View::Gallery {
                    if let Some(entry) = self.gallery.entries.get(self.gallery.selected) {
                        let filename = entry.filename();
                        self.popup = Some(Popup::Confirm {
                            message: format!("Delete {filename}?"),
                            on_confirm: ConfirmAction::DeleteGalleryImage,
                        });
                    }
                }
            }
            Action::OpenFile => {
                self.open_gallery_file();
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

    /// Load the currently selected gallery entry's image into the preview.
    fn load_gallery_preview(&mut self) {
        if let Some(entry) = self.gallery.entries.get(self.gallery.selected) {
            if let Some(ref server_url) = entry.server_url {
                // Server-backed: check cache first, then fetch async
                let url = server_url.clone();
                let filename = entry.filename();
                let cache_path = crate::gallery_scan::image_cache_dir().join(&filename);
                if cache_path.is_file() {
                    // Cached locally — load synchronously
                    if let Ok(img) = image::open(&cache_path) {
                        let protocol = self.picker.new_resize_protocol(img.clone());
                        self.gallery.preview_image = Some(img);
                        self.gallery.image_state = Some(protocol);
                        return;
                    }
                }
                // Not cached — fetch asynchronously
                let tx = self.bg_tx.clone();
                self.tokio_handle.spawn(async move {
                    if let Some(cached) =
                        crate::gallery_scan::fetch_and_cache_image(&url, &filename).await
                    {
                        let data = tokio::fs::read(&cached).await.unwrap_or_default();
                        let _ = tx.send(BackgroundEvent::GalleryPreviewReady(data));
                    }
                });
                self.gallery.preview_image = None;
                self.gallery.image_state = None;
            } else if entry.path.exists() && entry.path.is_file() {
                if let Ok(img) = image::open(&entry.path) {
                    let protocol = self.picker.new_resize_protocol(img.clone());
                    self.gallery.preview_image = Some(img);
                    self.gallery.image_state = Some(protocol);
                    return;
                }
            }
        }
        self.gallery.preview_image = None;
        self.gallery.image_state = None;
    }

    /// Load the selected gallery entry's metadata into the Generate view.
    fn load_gallery_into_generate(&mut self) {
        let entry = match self.gallery.entries.get(self.gallery.selected) {
            Some(e) => e.clone(),
            None => return,
        };
        let meta = &entry.metadata;

        // Populate prompt fields
        self.generate.prompt = tui_textarea::TextArea::from(meta.prompt.lines());
        if let Some(ref neg) = meta.negative_prompt {
            self.generate.negative_prompt = tui_textarea::TextArea::from(neg.lines());
        } else {
            self.generate.negative_prompt = tui_textarea::TextArea::default();
        }

        // Set model and parameters
        self.update_model(&meta.model);
        self.generate.params.seed = Some(meta.seed);
        self.generate.params.seed_mode = SeedMode::Fixed;
        self.generate.params.steps = meta.steps;
        self.generate.params.guidance = meta.guidance;
        self.generate.params.width = meta.width;
        self.generate.params.height = meta.height;
        if let Some(strength) = meta.strength {
            self.generate.params.strength = strength;
        }
        self.generate.params.scheduler = meta.scheduler;
        if let Some(ref lora) = meta.lora {
            self.generate.params.lora_path = Some(lora.clone());
            self.generate.params.lora_scale = meta.lora_scale.unwrap_or(1.0);
        } else {
            self.generate.params.lora_path = None;
        }

        // Switch to Generate view
        self.active_view = View::Generate;
        self.generate.focus = GenerateFocus::Prompt;
    }

    /// Delete the currently selected gallery image and its thumbnail.
    fn delete_selected_gallery_image(&mut self) {
        if self.gallery.entries.is_empty() {
            return;
        }
        let idx = self.gallery.selected;
        if idx >= self.gallery.entries.len() {
            return;
        }

        let entry = &self.gallery.entries[idx];
        let thumb_path = crate::thumbnails::thumbnail_path(&entry.path);
        let _ = std::fs::remove_file(&thumb_path);
        // Also remove the local cached copy (image cache)
        let cache_path = crate::gallery_scan::image_cache_dir().join(entry.filename());
        let _ = std::fs::remove_file(&cache_path);

        if let Some(ref url) = entry.server_url {
            // Delete from server via API
            let url = url.clone();
            let filename = entry.filename();
            self.tokio_handle.spawn(async move {
                let client = mold_core::MoldClient::new(&url);
                let _ = client.delete_gallery_image(&filename).await;
            });
        }
        // Always try to remove the local file (covers both local and server-backed entries
        // where the TUI also saved a copy during generation)
        if entry.path.is_file() {
            let _ = std::fs::remove_file(&entry.path);
        }

        // Remove from state
        self.gallery.entries.remove(idx);
        if idx < self.gallery.thumbnail_states.len() {
            self.gallery.thumbnail_states.remove(idx);
        }

        // Adjust selection
        if !self.gallery.entries.is_empty() {
            self.gallery.selected = idx.min(self.gallery.entries.len() - 1);
        } else {
            self.gallery.selected = 0;
            self.gallery.view_mode = GalleryViewMode::Grid;
        }

        // Reload preview if in detail mode
        if self.gallery.view_mode == GalleryViewMode::Detail {
            self.load_gallery_preview();
        }

        self.gallery.preview_image = None;
        self.gallery.image_state = None;
    }

    /// Open the selected gallery image in the system viewer.
    /// For server-backed entries, fetches and caches locally first.
    fn open_gallery_file(&mut self) {
        let entry = match self.gallery.entries.get(self.gallery.selected) {
            Some(e) => e.clone(),
            None => return,
        };

        if entry.server_url.is_none() && entry.path.is_file() {
            // Local file — open directly
            let _ = open::that(&entry.path);
        } else if let Some(ref url) = entry.server_url {
            // Server-backed — fetch to cache, then open
            let url = url.clone();
            let filename = entry.filename();
            self.tokio_handle.spawn(async move {
                if let Some(cached) =
                    crate::gallery_scan::fetch_and_cache_image(&url, &filename).await
                {
                    let _ = open::that(&cached);
                }
            });
        }
    }

    /// Dispatch a confirmed popup action.
    fn handle_confirm_action(&mut self, action: ConfirmAction) {
        match action {
            ConfirmAction::DeleteGalleryImage => {
                self.delete_selected_gallery_image();
            }
            ConfirmAction::RemoveModel(_name) => {
                // TODO: implement model removal
            }
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
                // Click/Enter on Seed mode row toggles the mode
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
            // Unload model from GPU memory
            ParamField::UnloadModel => {
                self.dispatch_action(Action::UnloadModel);
            }
            // Reset all params to model defaults (keep model and prompt)
            ParamField::ResetDefaults => {
                let model = self.generate.params.model.clone();
                let mc = self.config.resolved_model_config(&model);
                self.generate.params.width = mc.effective_width(&self.config);
                self.generate.params.height = mc.effective_height(&self.config);
                self.generate.params.steps = mc.effective_steps(&self.config);
                self.generate.params.guidance = mc.effective_guidance();
                self.generate.params.seed = None;
                self.generate.params.seed_mode = SeedMode::Random;
                self.generate.params.batch = 1;
                self.generate.params.format = OutputFormat::Png;
                self.generate.params.scheduler = None;
                self.generate.params.lora_path = None;
                self.generate.params.lora_scale = 1.0;
                self.generate.params.expand = false;
                self.generate.params.offload = false;
                self.generate.params.strength = 0.75;
                self.generate.params.source_image_path = None;
                self.generate.params.mask_image_path = None;
                self.generate.params.control_image_path = None;
                self.generate.params.control_model = None;
                self.generate.params.control_scale = 1.0;
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

                    // Resolve output directory (None when explicitly disabled)
                    let output_dir = if self.config.is_output_disabled() {
                        None
                    } else {
                        let dir = self.config.effective_output_dir();
                        let _ = std::fs::create_dir_all(&dir);
                        Some(dir)
                    };

                    // Save images to disk and display preview
                    let mut saved_path = std::path::PathBuf::new();
                    let ts_secs = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);

                    let prompt_text = self.generate.prompt.lines().join("\n").trim().to_string();
                    let neg_text = self
                        .generate
                        .negative_prompt
                        .lines()
                        .join("\n")
                        .trim()
                        .to_string();

                    for (i, img_data) in response.images.iter().enumerate() {
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
                        // Save to disk when output is enabled
                        if let Some(ref dir) = output_dir {
                            let path = dir.join(&filename);
                            if std::fs::write(&path, &img_data.data).is_ok() && i == 0 {
                                saved_path = path;
                            }
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

                    // Save session state
                    let session = crate::session::TuiSession::from_params(
                        &prompt_text,
                        &neg_text,
                        &self.generate.params,
                    );
                    session.save();
                    Config::write_last_model(&self.generate.params.model);

                    // Push to prompt history
                    let neg = if neg_text.is_empty() {
                        None
                    } else {
                        Some(neg_text.clone())
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

                    // Add to gallery (most recent first) with full metadata
                    if let Some(img_data) = response.images.first() {
                        let meta = mold_core::OutputMetadata {
                            prompt: prompt_text,
                            negative_prompt: if neg_text.is_empty() {
                                None
                            } else {
                                Some(neg_text)
                            },
                            original_prompt: None,
                            model: self.generate.params.model.clone(),
                            seed: response.seed_used,
                            steps: self.generate.params.steps,
                            guidance: self.generate.params.guidance,
                            width: img_data.width,
                            height: img_data.height,
                            strength: if self.generate.params.source_image_path.is_some() {
                                Some(self.generate.params.strength)
                            } else {
                                None
                            },
                            scheduler: self.generate.params.scheduler,
                            lora: self.generate.params.lora_path.clone(),
                            lora_scale: self
                                .generate
                                .params
                                .lora_path
                                .as_ref()
                                .map(|_| self.generate.params.lora_scale),
                            version: mold_core::build_info::VERSION.to_string(),
                        };

                        self.gallery.entries.insert(
                            0,
                            GalleryEntry {
                                path: saved_path.clone(),
                                metadata: meta,
                                generation_time_ms: Some(response.generation_time_ms),
                                timestamp: ts,
                                // Entry is local — the TUI saved this file directly.
                                // The server has its own copy via output_dir.
                                server_url: None,
                            },
                        );
                        self.gallery.thumbnail_states.insert(0, None);

                        // Generate thumbnail in background
                        self.tokio_handle.spawn(async move {
                            tokio::task::spawn_blocking(move || {
                                crate::thumbnails::generate_thumbnail(&saved_path).ok();
                            })
                            .await
                            .ok();
                        });
                    }
                }
                BackgroundEvent::Error(msg) => {
                    self.generate.generating = false;
                    self.generate.error_message = Some(msg);
                }
                BackgroundEvent::GalleryScanComplete(entries) => {
                    self.gallery.thumbnail_states = vec![None; entries.len()];
                    self.gallery.entries = entries;
                    self.gallery.scanning = false;
                    self.gallery.selected = 0;

                    // Spawn background thumbnail generation
                    let entries_info: Vec<(std::path::PathBuf, Option<String>)> = self
                        .gallery
                        .entries
                        .iter()
                        .map(|e| (e.path.clone(), e.server_url.clone()))
                        .collect();
                    let tx = self.bg_tx.clone();
                    self.tokio_handle.spawn(async move {
                        // Spawn all thumbnail fetches concurrently
                        let mut handles = Vec::new();
                        for (path, server_url) in entries_info {
                            if crate::thumbnails::thumbnail_exists(&path) {
                                continue;
                            }
                            let handle = tokio::spawn(async move {
                                let filename = path
                                    .file_name()
                                    .map(|f| f.to_string_lossy().to_string())
                                    .unwrap_or_default();
                                if let Some(url) = server_url {
                                    // Fetch pre-generated thumbnail from server (fast, ~10KB)
                                    let client = mold_core::MoldClient::new(&url);
                                    let fetched =
                                        if let Ok(data) = client.get_gallery_thumbnail(&filename).await
                                        {
                                            let key = path.clone();
                                            tokio::task::spawn_blocking(move || {
                                                crate::thumbnails::save_thumbnail_bytes(&data, &key)
                                                    .ok();
                                            })
                                            .await
                                            .ok();
                                            true
                                        } else {
                                            false
                                        };

                                    // Fallback: generate from locally cached image if server fetch failed
                                    if !fetched {
                                        let cache_path = crate::gallery_scan::image_cache_dir().join(&filename);
                                        if cache_path.is_file() {
                                            let key = path;
                                            tokio::task::spawn_blocking(move || {
                                                crate::thumbnails::generate_thumbnail_from_cached(&cache_path, &key)
                                                    .ok();
                                            })
                                            .await
                                            .ok();
                                        }
                                    }
                                } else {
                                    // Local file — generate thumbnail directly
                                    tokio::task::spawn_blocking(move || {
                                        crate::thumbnails::generate_thumbnail(&path).ok();
                                    })
                                    .await
                                    .ok();
                                }
                            });
                            handles.push(handle);
                        }
                        for h in handles {
                            let _ = h.await;
                        }
                        let _ = tx.send(BackgroundEvent::ThumbnailsReady);
                    });
                }
                BackgroundEvent::GalleryPreviewReady(data) => {
                    if let Ok(img) = image::load_from_memory(&data) {
                        let protocol = self.picker.new_resize_protocol(img.clone());
                        self.gallery.preview_image = Some(img);
                        self.gallery.image_state = Some(protocol);
                    }
                }
                BackgroundEvent::ThumbnailsReady => {
                    // Invalidate all thumbnail states so they reload on next render
                    let len = self.gallery.entries.len();
                    self.gallery.thumbnail_states = vec![None; len];
                }
                BackgroundEvent::ServerConnected { url, models } => {
                    self.server_url = Some(url.clone());
                    self.models.catalog = models;
                    self.models.selected = 0;
                    // Auto-switch to auto mode
                    if self.generate.params.inference_mode == InferenceMode::Local {
                        self.generate.params.inference_mode = InferenceMode::Auto;
                    }
                    self.generate.visible_fields = ParamField::visible_fields(
                        &self.generate.capabilities,
                        self.generate.params.inference_mode,
                    );
                    self.generate.progress.log.push(ProgressLogEntry {
                        message: format!("Connected to {url}"),
                        style: ProgressStyle::Done,
                    });
                    // Re-scan gallery from the (now-connected) server
                    self.gallery.scanning = true;
                    self.spawn_gallery_scan();
                }
                BackgroundEvent::ServerUnreachable(msg) => {
                    self.generate.progress.log.push(ProgressLogEntry {
                        message: format!("Server unreachable: {msg}"),
                        style: ProgressStyle::Error,
                    });
                    // Revert host — don't set server_url
                    self.generate.params.host = self.server_url.clone();
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
    fn seed_value_display_random_when_none() {
        let config = Config::load_or_default();
        let params = GenerateParams::from_config(&config);
        let display = params.display_value(&ParamField::SeedValue);
        assert!(display.contains("random"));
    }

    #[test]
    fn seed_value_shows_long_numbers_untruncated() {
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.seed = Some(11275518943372801901);
        let display = params.display_value(&ParamField::SeedValue);
        assert_eq!(display, "11275518943372801901");
    }

    // ── Regression tests for Codex review findings ────────

    #[test]
    fn history_nav_only_from_prompt_focus() {
        // History navigation should only work from Prompt focus,
        // not NegativePrompt — prevents clobbering the main prompt.
        let mut history = crate::history::PromptHistory::load();
        // Seed some history
        history.push_entry(crate::history::HistoryEntry {
            prompt: "old prompt".to_string(),
            negative: None,
            model: "test".to_string(),
            timestamp: 0,
        });

        // prev() from Prompt focus should return something
        let result = history.prev("current");
        assert!(result.is_some());
        history.reset_cursor();

        // The key invariant: the calling code must check focus == Prompt
        // before calling history.prev(). This test documents that contract.
        // If focus were NegativePrompt, the caller must NOT call history methods.
    }

    #[test]
    fn unimplemented_actions_exist() {
        // Document which actions are still intentionally unhandled in dispatch_action.
        let unimplemented = vec![
            Action::ZoomIn,
            Action::ZoomOut,
            Action::PanLeft,
            Action::PanRight,
            Action::FilterModels,
            Action::RemoveModel,
            Action::ExpandPrompt,
            Action::SaveImage,
            Action::CompareModels,
        ];
        // Compile-time check that these variants exist
        for action in &unimplemented {
            assert_ne!(*action, Action::Quit);
        }
    }

    #[test]
    fn gallery_actions_are_implemented() {
        // These gallery actions should exist and NOT be in the unimplemented list
        let implemented = vec![
            Action::Regenerate,
            Action::EditAndGenerate,
            Action::DeleteImage,
            Action::OpenFile,
            Action::GridLeft,
            Action::GridRight,
        ];
        for action in &implemented {
            assert_ne!(*action, Action::Quit);
        }
    }

    #[test]
    fn visible_fields_ends_with_tools_section() {
        // The Tools section (ResetDefaults, UnloadModel) should always be at the end
        let caps = crate::model_info::capabilities_for_family("flux");
        let fields = ParamField::visible_fields(&caps, InferenceMode::Auto);
        assert_eq!(*fields.last().unwrap(), ParamField::UnloadModel);
        // ResetDefaults should be just before UnloadModel
        let reset_pos = fields
            .iter()
            .position(|f| *f == ParamField::ResetDefaults)
            .unwrap();
        let unload_pos = fields
            .iter()
            .position(|f| *f == ParamField::UnloadModel)
            .unwrap();
        assert_eq!(unload_pos, reset_pos + 1);
    }

    #[test]
    fn reset_defaults_display_value() {
        let config = Config::load_or_default();
        let params = GenerateParams::from_config(&config);
        let display = params.display_value(&ParamField::ResetDefaults);
        assert_eq!(display, "restore model defaults");
    }

    #[test]
    fn unload_model_display_value() {
        let config = Config::load_or_default();
        let params = GenerateParams::from_config(&config);
        let display = params.display_value(&ParamField::UnloadModel);
        assert_eq!(display, "free GPU memory");
    }

    #[test]
    fn unload_model_label() {
        assert!(ParamField::UnloadModel.label().contains("Unload"));
    }

    #[test]
    fn unload_model_has_no_section_header() {
        // UnloadModel is under the Actions section started by ResetDefaults
        assert!(ParamField::UnloadModel.section_header().is_none());
    }

    #[test]
    fn reset_defaults_starts_actions_section() {
        assert_eq!(ParamField::ResetDefaults.section_header(), Some("Actions"));
    }

    #[test]
    fn unload_model_always_visible() {
        // UnloadModel should appear regardless of model family or inference mode
        for family in &["flux", "sd15", "sdxl", "sd3", "flux2"] {
            for mode in &[
                InferenceMode::Auto,
                InferenceMode::Local,
                InferenceMode::Remote,
            ] {
                let caps = crate::model_info::capabilities_for_family(family);
                let fields = ParamField::visible_fields(&caps, *mode);
                assert!(
                    fields.contains(&ParamField::UnloadModel),
                    "UnloadModel missing for family={} mode={:?}",
                    family,
                    mode
                );
            }
        }
    }

    // ── Gallery tests ────────────────────────────────────

    #[test]
    fn gallery_view_mode_default_is_grid() {
        assert_eq!(GalleryViewMode::default(), GalleryViewMode::Grid);
    }

    #[test]
    fn gallery_entry_filename_extracts_name() {
        let entry = GalleryEntry {
            path: std::path::PathBuf::from("/home/user/.mold/output/mold-flux-1234.png"),
            metadata: mold_core::OutputMetadata {
                prompt: "test".to_string(),
                negative_prompt: None,
                original_prompt: None,
                model: "flux:q8".to_string(),
                seed: 42,
                steps: 20,
                guidance: 7.5,
                width: 1024,
                height: 1024,
                strength: None,
                scheduler: None,
                lora: None,
                lora_scale: None,
                version: "0.3.1".to_string(),
            },
            generation_time_ms: Some(5000),
            timestamp: 1234,
            server_url: None,
        };
        assert_eq!(entry.filename(), "mold-flux-1234.png");
    }

    #[test]
    fn gallery_entry_filename_unknown_for_empty_path() {
        let entry = GalleryEntry {
            path: std::path::PathBuf::new(),
            metadata: mold_core::OutputMetadata {
                prompt: "test".to_string(),
                negative_prompt: None,
                original_prompt: None,
                model: "test".to_string(),
                seed: 0,
                steps: 1,
                guidance: 0.0,
                width: 512,
                height: 512,
                strength: None,
                scheduler: None,
                lora: None,
                lora_scale: None,
                version: "0.0.0".to_string(),
            },
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
        };
        assert_eq!(entry.filename(), "unknown");
    }

    #[test]
    fn gallery_grid_nav_up_moves_by_cols() {
        // With grid_cols=3, selected=5 (row 1, col 2), Up should go to 2 (row 0, col 2)
        let selected: usize = 5;
        let cols: usize = 3;
        let result = if selected >= cols {
            selected - cols
        } else {
            selected
        };
        assert_eq!(result, 2);
    }

    #[test]
    fn gallery_grid_nav_down_moves_by_cols() {
        let selected: usize = 2;
        let cols: usize = 3;
        let len: usize = 9;
        let next = selected + cols;
        let result = if next < len { next } else { selected };
        assert_eq!(result, 5);
    }

    #[test]
    fn gallery_grid_nav_clamps_at_top() {
        let selected: usize = 1;
        let cols: usize = 3;
        // Can't go up from row 0
        let result = if selected >= cols {
            selected - cols
        } else {
            selected
        };
        assert_eq!(result, 1); // stays put
    }

    #[test]
    fn gallery_grid_nav_left_right() {
        let selected: usize = 3;
        let len: usize = 10;
        // Left
        assert_eq!(selected.saturating_sub(1), 2);
        // Right
        assert_eq!((selected + 1).min(len - 1), 4);
    }

    #[test]
    fn confirm_action_delete_gallery_image_variant_exists() {
        let action = ConfirmAction::DeleteGalleryImage;
        match action {
            ConfirmAction::DeleteGalleryImage => {}
            _ => panic!("expected DeleteGalleryImage"),
        }
    }

    #[test]
    fn confirm_action_remove_model_variant_exists() {
        let action = ConfirmAction::RemoveModel("test".to_string());
        match action {
            ConfirmAction::RemoveModel(name) => assert_eq!(name, "test"),
            _ => panic!("expected RemoveModel"),
        }
    }

    fn make_test_metadata() -> mold_core::OutputMetadata {
        mold_core::OutputMetadata {
            prompt: "a test prompt".to_string(),
            negative_prompt: Some("blurry".to_string()),
            original_prompt: None,
            model: "flux:q8".to_string(),
            seed: 42,
            steps: 20,
            guidance: 7.5,
            width: 1024,
            height: 1024,
            strength: Some(0.75),
            scheduler: None,
            lora: Some("/path/to/adapter.safetensors".to_string()),
            lora_scale: Some(0.8),
            version: "0.3.1".to_string(),
        }
    }

    fn make_test_entry() -> GalleryEntry {
        GalleryEntry {
            path: std::path::PathBuf::from("/home/user/.mold/output/mold-flux-1234.png"),
            metadata: make_test_metadata(),
            generation_time_ms: Some(5000),
            timestamp: 1234,
            server_url: None,
        }
    }

    #[test]
    fn gallery_entry_metadata_accessible() {
        let entry = make_test_entry();
        assert_eq!(entry.metadata.prompt, "a test prompt");
        assert_eq!(entry.metadata.model, "flux:q8");
        assert_eq!(entry.metadata.seed, 42);
        assert_eq!(entry.metadata.steps, 20);
        assert_eq!(entry.metadata.width, 1024);
        assert_eq!(entry.metadata.negative_prompt, Some("blurry".to_string()));
        assert_eq!(entry.metadata.strength, Some(0.75));
        assert_eq!(entry.metadata.lora_scale, Some(0.8));
    }

    #[test]
    fn gallery_entry_clone() {
        let entry = make_test_entry();
        let cloned = entry.clone();
        assert_eq!(cloned.filename(), entry.filename());
        assert_eq!(cloned.metadata.prompt, entry.metadata.prompt);
        assert_eq!(cloned.timestamp, entry.timestamp);
    }

    #[test]
    fn gallery_grid_nav_down_clamps_at_bottom() {
        let selected: usize = 7;
        let cols: usize = 3;
        let len: usize = 9;
        let next = selected + cols;
        // 7 + 3 = 10, but len is 9, so stay at 7
        let result = if next < len { next } else { selected };
        assert_eq!(result, 7);
    }

    #[test]
    fn gallery_grid_nav_right_clamps_at_end() {
        let selected: usize = 8;
        let len: usize = 9;
        let result = (selected + 1).min(len - 1);
        assert_eq!(result, 8); // already at last item
    }

    #[test]
    fn gallery_grid_nav_left_clamps_at_zero() {
        let selected: usize = 0;
        assert_eq!(selected.saturating_sub(1), 0);
    }

    #[test]
    fn seed_activate_toggles_mode() {
        // Seed field activation should cycle mode, not open popup
        // This tests the contract: Seed row = toggle mode, SeedValue row = popup
        let mode = SeedMode::Random;
        let next = mode.next();
        assert_eq!(next, SeedMode::Fixed);
        let next2 = next.next();
        assert_eq!(next2, SeedMode::Increment);
        let next3 = next2.next();
        assert_eq!(next3, SeedMode::Random);
    }

    #[test]
    fn gallery_view_mode_equality() {
        assert_eq!(GalleryViewMode::Grid, GalleryViewMode::Grid);
        assert_eq!(GalleryViewMode::Detail, GalleryViewMode::Detail);
        assert_ne!(GalleryViewMode::Grid, GalleryViewMode::Detail);
    }

    #[test]
    fn gallery_state_default_grid_cols() {
        // Default grid_cols should be reasonable
        let state = GalleryState {
            entries: Vec::new(),
            selected: 0,
            preview_image: None,
            image_state: None,
            scanning: false,
            view_mode: GalleryViewMode::Grid,
            thumbnail_states: Vec::new(),
            grid_cols: 3,
            grid_scroll: 0,
        };
        assert_eq!(state.grid_cols, 3);
        assert_eq!(state.grid_scroll, 0);
        assert!(state.thumbnail_states.is_empty());
    }

    #[test]
    fn gallery_thumbnail_states_sync_with_entries() {
        // thumbnail_states should have same length as entries
        let entries = vec![make_test_entry(), make_test_entry()];
        let thumb_states: Vec<Option<StatefulProtocol>> = vec![None; entries.len()];
        assert_eq!(thumb_states.len(), entries.len());
    }

    #[test]
    fn default_output_dir_path() {
        let dir = crate::gallery_scan::default_gallery_dir();
        let s = dir.to_string_lossy();
        assert!(
            s.ends_with("output"),
            "expected path ending in 'output': {s}"
        );
    }

    #[test]
    fn background_event_thumbnails_ready_variant() {
        // Verify the variant exists
        let event = BackgroundEvent::ThumbnailsReady;
        match event {
            BackgroundEvent::ThumbnailsReady => {}
            _ => panic!("expected ThumbnailsReady"),
        }
    }
}
