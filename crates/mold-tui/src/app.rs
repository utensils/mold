use anyhow::Result;
use crossterm::event::{Event as CrosstermEvent, KeyCode, KeyModifiers};
use mold_core::{
    Config, GenerateResponse, ModelInfoExtended, OutputFormat, Scheduler, ServerStatus,
    SseProgressEvent,
};
use rand::Rng;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use std::collections::VecDeque;
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
    /// Model removal completed successfully.
    ModelRemoveComplete(String),
    /// Model removal failed.
    ModelRemoveFailed(String),
    /// Upscale download progress (model pull during upscale).
    UpscaleDownloadProgress(SseProgressEvent),
    /// Upscale tile progress update.
    UpscaleProgress { tile: usize, total: usize },
    /// Upscale completed successfully.
    UpscaleComplete {
        image_data: Vec<u8>,
        source_path: std::path::PathBuf,
        model: String,
        scale_factor: u32,
        original_width: u32,
        original_height: u32,
        upscale_time_ms: u64,
    },
    /// Upscale failed.
    UpscaleFailed(String),
    /// Periodic server status update (remote resource info).
    /// `None` means the server became unreachable — clear stale status.
    ServerStatusUpdate(Option<Box<ServerStatus>>),
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
    pub download_batch_bytes: u64,
    pub download_batch_total: u64,
    pub download_batch_elapsed_ms: u64,
    pub download_rate_bps: Option<f64>,
    pub download_eta_secs: Option<f64>,
    pub download_file_index: usize,
    pub download_total_files: usize,
    pub downloading: bool,
    download_samples: VecDeque<(u64, u64)>,
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
        self.download_batch_bytes = 0;
        self.download_batch_total = 0;
        self.download_batch_elapsed_ms = 0;
        self.download_rate_bps = None;
        self.download_eta_secs = None;
        self.download_file_index = 0;
        self.download_total_files = 0;
        self.downloading = false;
        self.download_samples.clear();
    }

    fn clear_download(&mut self) {
        self.download_filename.clear();
        self.download_bytes = 0;
        self.download_total = 0;
        self.download_batch_bytes = 0;
        self.download_batch_total = 0;
        self.download_batch_elapsed_ms = 0;
        self.download_rate_bps = None;
        self.download_eta_secs = None;
        self.download_file_index = 0;
        self.download_total_files = 0;
        self.downloading = false;
        self.download_samples.clear();
    }

    /// Returns true if a model download or verification is active.
    pub fn is_downloading(&self) -> bool {
        self.downloading
    }

    /// Human-readable status for the bottom bar during pull.
    pub fn download_status_text(&self) -> &str {
        if self.download_batch_total > 0 {
            "Downloading..."
        } else if self
            .current_stage
            .as_deref()
            .is_some_and(|s| s.contains("Verifying"))
        {
            "Verifying..."
        } else if self.downloading {
            "Preparing..."
        } else {
            "Downloading..."
        }
    }

    fn clear_weight(&mut self) {
        self.weight_loaded = 0;
        self.weight_total = 0;
        self.weight_component.clear();
    }

    fn record_download_sample(&mut self, elapsed_ms: u64, position: u64) {
        const MAX_SAMPLES: usize = 8;
        const MIN_SAMPLE_WINDOW_MS: u64 = 1_000;

        if self
            .download_samples
            .back()
            .is_some_and(|(last_elapsed_ms, _)| *last_elapsed_ms == elapsed_ms)
        {
            let _ = self.download_samples.pop_back();
        }
        self.download_samples.push_back((elapsed_ms, position));
        while self.download_samples.len() > MAX_SAMPLES {
            self.download_samples.pop_front();
        }

        if self.download_samples.len() < 2 {
            self.download_rate_bps = None;
            self.download_eta_secs = None;
            return;
        }

        let (t_old_ms, b_old) = self
            .download_samples
            .front()
            .expect("sample window is non-empty");
        let (t_new_ms, b_new) = self
            .download_samples
            .back()
            .expect("sample window is non-empty");
        let dt_ms = t_new_ms.saturating_sub(*t_old_ms);
        if dt_ms < MIN_SAMPLE_WINDOW_MS {
            self.download_rate_bps = None;
            self.download_eta_secs = None;
            return;
        }

        let dt = dt_ms as f64 / 1_000.0;
        let rate = b_new.saturating_sub(*b_old) as f64 / dt;
        if rate < 1.0 {
            self.download_rate_bps = None;
            self.download_eta_secs = None;
            return;
        }

        self.download_rate_bps = Some(rate);
        self.download_eta_secs =
            Some(self.download_batch_total.saturating_sub(position) as f64 / rate);
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
    // Video
    Frames,
    Fps,
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
        // Video
        if caps.supports_video {
            fields.push(ParamField::Frames);
            fields.push(ParamField::Fps);
        }
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
            Self::Frames => "Frames",
            Self::Fps => "FPS",
            Self::ControlScale => "Scale",
            Self::ResetDefaults => "\u{21ba} Reset",
            Self::UnloadModel => "\u{23cf} Unload",
        }
    }

    /// The section header this field falls under, if it starts a new section.
    pub fn section_header(&self) -> Option<&'static str> {
        match self {
            Self::Scheduler => Some("Advanced"),
            Self::Frames => Some("Video"),
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
    // Video
    pub frames: u32,
    pub fps: u32,
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
            frames: 25,
            fps: 24,
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
            ParamField::Frames => self.frames.to_string(),
            ParamField::Fps => self.fps.to_string(),
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
    /// Number of images remaining in the current batch (0 when not batching).
    pub batch_remaining: u32,
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
    /// Actual thumbnail pixel dimensions (width, height), populated when loaded.
    pub thumb_dimensions: Vec<Option<(u32, u32)>>,
    /// Cached fixed-protocol renders for centered grid thumbnails.
    /// Populated lazily on first render, keyed by (thumb_area width, height).
    pub thumb_fixed_cache: Vec<Option<(u16, u16, ratatui_image::protocol::Protocol)>>,
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

// ── Settings view types ─────────────────────────────────────────────

/// Identifies a single config field in the Settings view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsKey {
    // General
    DefaultModel,
    ModelsDir,
    OutputDir,
    ServerPort,
    DefaultWidth,
    DefaultHeight,
    DefaultSteps,
    EmbedMetadata,
    T5Variant,
    Qwen3Variant,
    DefaultNegativePrompt,
    // Expand
    ExpandEnabled,
    ExpandBackend,
    ExpandModel,
    ExpandApiModel,
    ExpandTemperature,
    ExpandTopP,
    ExpandMaxTokens,
    ExpandThinking,
    // Logging
    LogLevel,
    LogFile,
    LogDir,
    LogMaxDays,
    // Model defaults (operate on selected_model)
    ModelSelector,
    ModelSteps,
    ModelGuidance,
    ModelWidth,
    ModelHeight,
    ModelScheduler,
    ModelNegativePrompt,
    ModelLora,
    ModelLoraScale,
    // Model paths (read-only)
    ModelTransformer,
    ModelVae,
}

/// The type of a settings field — determines editing behavior.
#[derive(Debug, Clone)]
pub enum SettingsFieldType {
    /// Opens a text popup on Enter.
    Text,
    /// Inline +/- adjustment with clamping.
    Number { min: f64, max: f64, step: f64 },
    /// Cycles through a fixed set of options.
    Toggle { options: Vec<&'static str> },
    /// On/off toggle.
    Bool,
    /// Opens a path popup on Enter.
    Path,
    /// Display only, no editing.
    ReadOnly,
}

/// A single renderable row in the settings list.
#[derive(Debug, Clone)]
pub enum SettingsRow {
    SectionHeader {
        name: String,
    },
    Field {
        key: SettingsKey,
        label: &'static str,
        field_type: SettingsFieldType,
    },
}

impl SettingsRow {
    pub fn is_field(&self) -> bool {
        matches!(self, SettingsRow::Field { .. })
    }

    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            SettingsRow::Field {
                field_type: SettingsFieldType::ReadOnly,
                ..
            }
        )
    }
}

/// State for the Settings view.
#[derive(Default)]
pub struct SettingsState {
    /// Index into the flat row list (including headers).
    pub row_index: usize,
    /// Scroll offset for the rendered list.
    pub scroll_offset: usize,
    /// Currently selected model name for the "Model Defaults" section.
    pub selected_model: Option<String>,
    /// Brief error message if a save fails.
    pub save_error: Option<String>,
    /// When true, `save_config()` skips writing to disk (used in tests).
    #[cfg(test)]
    pub skip_save: bool,
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
    SettingsInput {
        key: SettingsKey,
        input: String,
        label: String,
    },
    /// Informational message (dismissed with any key).
    Info {
        message: String,
    },
    /// Upscaler model selector (all known upscaler models, auto-pulls on select).
    UpscaleModelSelector {
        filter: String,
        selected: usize,
        filtered: Vec<String>,
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
    pub settings: SettingsState,
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
    /// Whether an upscale job is currently running.
    pub upscale_in_progress: bool,
    /// Handle to the background upscale task (for cancellation).
    pub upscale_task: Option<tokio::task::JoinHandle<()>>,
    /// Current tile progress for in-flight upscale (current, total).
    pub upscale_tile_progress: Option<(usize, usize)>,
    /// Download progress state during upscaler model pull.
    pub upscale_progress: ProgressState,
    /// True while a background server health check / connect is in progress.
    pub connecting: bool,
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
                // Server already running — connect but don't manage its lifecycle
                tracing::info!(%local_url, "connected to existing server");
                (Some(local_url.clone()), InferenceMode::Auto)
            } else {
                // Try to start a background server
                match start_background_server(port) {
                    Some(mut child) => {
                        if wait_for_server_health(&local_url, 8) {
                            tracing::info!(pid = child.id(), "started background server");
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
        // Try the saved model name as-is first (handles config-only custom models
        // like [models."my-flux"]), then try resolving bare manifest names
        // (e.g. "flux2-klein" → "flux2-klein:q8").
        let model_found = if !session.last_model.is_empty() {
            let exact_in_catalog = catalog.iter().any(|m| m.name == session.last_model);
            if config.manifest_model_is_downloaded(&session.last_model) || exact_in_catalog {
                Some(session.last_model.clone())
            } else {
                let resolved = mold_core::manifest::resolve_model_name(&session.last_model);
                let resolved_in_catalog = catalog.iter().any(|m| m.name == resolved);
                if config.manifest_model_is_downloaded(&resolved) || resolved_in_catalog {
                    Some(resolved)
                } else {
                    None
                }
            }
        } else {
            None
        };

        if let Some(model_name) = model_found {
            params.model = model_name;
            // Apply all saved params (width, height, steps, guidance, batch, etc.)
            session.apply_to_params(&mut params);
            // Re-derive capabilities for the restored model
            let fam = family_for_model(&params.model, &config);
            let caps = capabilities_for_family(&fam);
            visible_fields = ParamField::visible_fields(&caps, initial_mode);
            capabilities = caps;
        } else {
            // Model not found — only apply non-model-specific settings.
            // Skip width/height/steps/guidance/scheduler since they belong to
            // the missing model and would be wrong for the current default.
            session.apply_non_model_params(&mut params);
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
                batch_remaining: 0,
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
                thumb_dimensions: Vec::new(),
                thumb_fixed_cache: Vec::new(),
                grid_cols: 3,
                grid_scroll: 0,
            },
            models: ModelsState {
                catalog,
                selected: 0,
                filter: String::new(),
                filtering: false,
            },
            settings: {
                let first_model = config.models.keys().next().cloned();
                SettingsState {
                    selected_model: first_model,
                    row_index: 1, // skip first section header
                    ..Default::default()
                }
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
            upscale_in_progress: false,
            upscale_task: None,
            upscale_tile_progress: None,
            upscale_progress: ProgressState::default(),
            connecting: false,
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

    /// Whether the event loop should poll `/api/status` instead of local sysinfo.
    /// True when connected to a server AND not forced into local mode.
    pub fn should_poll_remote(&self) -> bool {
        self.server_url.is_some() && self.generate.params.inference_mode != InferenceMode::Local
    }

    /// Sync resource info source after mode changes.
    /// Switches between local sysinfo and remote `/api/status` polling.
    fn sync_resource_info_mode(&mut self) {
        if self.generate.params.inference_mode == InferenceMode::Local {
            self.resource_info.clear_server_status();
            self.resource_info.refresh_local();
        } else if self.server_url.is_some() {
            self.spawn_server_status_fetch();
        }
    }

    /// Spawn a background fetch of `/api/status` from the connected server.
    pub fn spawn_server_status_fetch(&self) {
        let Some(ref url) = self.server_url else {
            return;
        };
        let tx = self.bg_tx.clone();
        let url = url.clone();
        self.tokio_handle.spawn(async move {
            let client = mold_core::MoldClient::new(&url);
            match client.server_status().await {
                Ok(status) => {
                    let _ = tx.send(BackgroundEvent::ServerStatusUpdate(Some(Box::new(status))));
                }
                Err(_) => {
                    // Server became unreachable — clear stale status so the UI
                    // stops showing the last-known hostname/memory.
                    let _ = tx.send(BackgroundEvent::ServerStatusUpdate(None));
                }
            }
        });
    }

    /// Apply model defaults from the server's catalog to the current model.
    /// When connected remotely, the server's config is authoritative for steps,
    /// guidance, width, and height.
    fn apply_remote_model_defaults(&mut self, catalog: &[ModelInfoExtended]) {
        let model_name = &self.generate.params.model;
        if let Some(entry) = catalog.iter().find(|m| &m.name == model_name) {
            self.generate.params.steps = entry.defaults.default_steps;
            self.generate.params.guidance = entry.defaults.default_guidance;
            self.generate.params.width = entry.defaults.default_width;
            self.generate.params.height = entry.defaults.default_height;
            if !entry.defaults.description.is_empty() {
                self.generate.model_description = entry.defaults.description.clone();
            }
        }
    }

    /// Spawn a background upscale job for the currently selected gallery image.
    fn spawn_upscale(&mut self, model_name: String) {
        let entry = match self.gallery.entries.get(self.gallery.selected) {
            Some(e) => e.clone(),
            None => return,
        };

        self.upscale_in_progress = true;
        self.upscale_tile_progress = None;
        self.upscale_progress.clear();

        // Switch to grid view to avoid image protocol conflicts with progress overlay
        if self.gallery.view_mode == GalleryViewMode::Detail {
            self.gallery.view_mode = GalleryViewMode::Grid;
            self.gallery.preview_image = None;
            self.gallery.image_state = None;
        }

        let tx = self.bg_tx.clone();
        let server_url = self.server_url.clone();
        let config = self.config.clone();
        let source_path = entry.path.clone();

        let handle = self.tokio_handle.spawn(async move {
            // Read image bytes
            let image_bytes = if let Some(ref url) = entry.server_url {
                let filename = entry.filename();
                match crate::gallery_scan::fetch_and_cache_image(url, &filename).await {
                    Some(cached_path) => match tokio::fs::read(&cached_path).await {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            let _ = tx.send(BackgroundEvent::UpscaleFailed(format!(
                                "Failed to read cached image: {e}"
                            )));
                            return;
                        }
                    },
                    None => {
                        let _ = tx.send(BackgroundEvent::UpscaleFailed(
                            "Failed to fetch image from server".into(),
                        ));
                        return;
                    }
                }
            } else {
                match tokio::fs::read(&entry.path).await {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        let _ = tx.send(BackgroundEvent::UpscaleFailed(format!(
                            "Failed to read image: {e}"
                        )));
                        return;
                    }
                }
            };

            let req = mold_core::UpscaleRequest {
                model: model_name.clone(),
                image: image_bytes,
                output_format: mold_core::OutputFormat::Png,
                tile_size: None,
            };

            // Try server first — use SSE streaming for tile progress
            if let Some(ref url) = server_url {
                let client = mold_core::MoldClient::new(url);

                // Stream progress events from SSE to the TUI
                let (progress_tx, mut progress_rx) =
                    tokio::sync::mpsc::unbounded_channel::<mold_core::SseProgressEvent>();
                let tx_sse = tx.clone();
                tokio::spawn(async move {
                    while let Some(event) = progress_rx.recv().await {
                        match &event {
                            mold_core::SseProgressEvent::DenoiseStep { step, total, .. } => {
                                let _ = tx_sse.send(BackgroundEvent::UpscaleProgress {
                                    tile: *step,
                                    total: *total,
                                });
                            }
                            mold_core::SseProgressEvent::DownloadProgress { .. }
                            | mold_core::SseProgressEvent::DownloadDone { .. }
                            | mold_core::SseProgressEvent::PullComplete { .. }
                            | mold_core::SseProgressEvent::StageStart { .. }
                            | mold_core::SseProgressEvent::Info { .. } => {
                                let _ =
                                    tx_sse.send(BackgroundEvent::UpscaleDownloadProgress(event));
                            }
                            _ => {}
                        }
                    }
                });

                match client.upscale_stream(&req, progress_tx).await {
                    Ok(Some(resp)) => {
                        let _ = tx.send(BackgroundEvent::UpscaleComplete {
                            image_data: resp.image.data,
                            source_path,
                            model: resp.model,
                            scale_factor: resp.scale_factor,
                            original_width: resp.original_width,
                            original_height: resp.original_height,
                            upscale_time_ms: resp.upscale_time_ms,
                        });
                        return;
                    }
                    Ok(None) => {
                        // Server doesn't support streaming upscale, fall back to non-streaming
                        match client.upscale(&req).await {
                            Ok(resp) => {
                                let _ = tx.send(BackgroundEvent::UpscaleComplete {
                                    image_data: resp.image.data,
                                    source_path,
                                    model: resp.model,
                                    scale_factor: resp.scale_factor,
                                    original_width: resp.original_width,
                                    original_height: resp.original_height,
                                    upscale_time_ms: resp.upscale_time_ms,
                                });
                                return;
                            }
                            Err(e) if mold_core::MoldClient::is_connection_error(&e) => {}
                            Err(e) => {
                                let _ = tx.send(BackgroundEvent::UpscaleFailed(format!(
                                    "Server error: {e}"
                                )));
                                return;
                            }
                        }
                    }
                    Err(e) if mold_core::MoldClient::is_connection_error(&e) => {
                        // Fall through to local
                    }
                    Err(e) => {
                        let _ =
                            tx.send(BackgroundEvent::UpscaleFailed(format!("Server error: {e}")));
                        return;
                    }
                }
            }

            // Local fallback — auto-pull if not downloaded, then upscale
            let resolved = mold_core::manifest::resolve_model_name(&model_name);
            let mut config = config;
            if config
                .models
                .get(&resolved)
                .and_then(|c| c.transformer.as_ref())
                .is_none()
            {
                // Wrap the sender so auto_pull_model's Progress events become
                // UpscaleDownloadProgress events (routed to upscale_progress,
                // not generate.progress).
                let (remap_tx, mut remap_rx) = tokio::sync::mpsc::unbounded_channel();
                let tx_remap = tx.clone();
                let remap_task = tokio::spawn(async move {
                    while let Some(event) = remap_rx.recv().await {
                        let remapped = match event {
                            BackgroundEvent::Progress(sse) => {
                                BackgroundEvent::UpscaleDownloadProgress(sse)
                            }
                            other => other,
                        };
                        let _ = tx_remap.send(remapped);
                    }
                });

                match crate::backend::auto_pull_model(&resolved, &remap_tx).await {
                    Ok(updated_config) => {
                        config = updated_config;
                    }
                    Err(msg) => {
                        let _ = tx.send(BackgroundEvent::UpscaleFailed(msg));
                        return;
                    }
                }
                drop(remap_tx);
                let _ = remap_task.await;
            }

            let model_name_local = resolved;
            let tx_progress = tx.clone();
            let result = tokio::task::spawn_blocking(move || {
                let weights_path = config
                    .models
                    .get(&model_name_local)
                    .and_then(|c| c.transformer.as_ref())
                    .map(std::path::PathBuf::from)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Upscaler model '{}' not configured", model_name_local)
                    })?;

                let mut engine = mold_inference::create_upscale_engine(
                    model_name_local.clone(),
                    weights_path,
                    mold_inference::LoadStrategy::Eager,
                )?;

                engine.set_on_progress(Box::new(move |event| {
                    if let mold_inference::ProgressEvent::DenoiseStep { step, total, .. } = event {
                        let _ = tx_progress
                            .send(BackgroundEvent::UpscaleProgress { tile: step, total });
                    }
                }));

                engine.upscale(&req)
            })
            .await;

            match result {
                Ok(Ok(resp)) => {
                    let _ = tx.send(BackgroundEvent::UpscaleComplete {
                        image_data: resp.image.data,
                        source_path,
                        model: resp.model,
                        scale_factor: resp.scale_factor,
                        original_width: resp.original_width,
                        original_height: resp.original_height,
                        upscale_time_ms: resp.upscale_time_ms,
                    });
                }
                Ok(Err(e)) => {
                    let _ = tx.send(BackgroundEvent::UpscaleFailed(format!("{e}")));
                }
                Err(e) => {
                    let _ = tx.send(BackgroundEvent::UpscaleFailed(format!(
                        "Task panicked: {e}"
                    )));
                }
            }
        });

        self.upscale_task = Some(handle);
    }

    /// Clean up resources on quit (kills background server if we spawned it).
    pub fn shutdown(&mut self) {
        // Save current session so settings persist even without generating
        self.save_session();

        if let Some(ref mut child) = self.server_process {
            tracing::info!(pid = child.id(), "stopping background server");
            let _ = child.kill();
            let _ = child.wait();
        }
        self.server_process = None;
    }

    /// Persist current prompt, negative prompt, model, and all params to session file.
    pub fn save_session(&self) {
        let prompt_text = self.generate.prompt.lines().join("\n").trim().to_string();
        let neg_text = self
            .generate
            .negative_prompt
            .lines()
            .join("\n")
            .trim()
            .to_string();
        let session =
            crate::session::TuiSession::from_params(&prompt_text, &neg_text, &self.generate.params);
        session.save();
    }

    pub fn update_model(&mut self, model_name: &str) {
        let model_name = model_name.to_string();
        self.generate.params.model = model_name.clone();

        // Use server catalog defaults when connected to a remote server,
        // local config otherwise.
        let used_remote = if self.server_url.is_some() {
            if let Some(entry) = self.models.catalog.iter().find(|m| m.name == model_name) {
                self.generate.params.steps = entry.defaults.default_steps;
                self.generate.params.guidance = entry.defaults.default_guidance;
                self.generate.params.width = entry.defaults.default_width;
                self.generate.params.height = entry.defaults.default_height;
                if !entry.defaults.description.is_empty() {
                    self.generate.model_description = entry.defaults.description.clone();
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        if !used_remote {
            let model_cfg = self.config.resolved_model_config(&model_name);
            self.generate.params.steps = model_cfg.effective_steps(&self.config);
            self.generate.params.guidance = model_cfg.effective_guidance();
            self.generate.params.width = model_cfg.effective_width(&self.config);
            self.generate.params.height = model_cfg.effective_height(&self.config);

            self.generate.model_description = mold_core::manifest::find_manifest(&model_name)
                .and_then(|m| {
                    let mc = self.config.resolved_model_config(&model_name);
                    mc.description.or(Some(m.name.clone()))
                })
                .unwrap_or_default();
        }

        let family = family_for_model(&model_name, &self.config);
        self.generate.capabilities = capabilities_for_family(&family);
        self.generate.visible_fields = ParamField::visible_fields(
            &self.generate.capabilities,
            self.generate.params.inference_mode,
        );
        self.generate.param_index = 0;
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
                        | (KeyCode::Char('3'), KeyModifiers::ALT)
                        | (KeyCode::Char('4'), KeyModifiers::ALT)
                        | (KeyCode::Left, KeyModifiers::ALT)
                        | (KeyCode::Right, KeyModifiers::ALT) => {
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
                Some(Popup::UpscaleModelSelector {
                    filter,
                    selected,
                    filtered,
                }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        if let Some(model) = filtered.get(*selected).cloned() {
                            self.close_popup();
                            self.spawn_upscale(model);
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
                        self.update_upscale_model_filter();
                    }
                    KeyCode::Backspace => {
                        filter.pop();
                        self.update_upscale_model_filter();
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
                            // Switch to local resource info
                            self.resource_info.clear_server_status();
                            self.resource_info.refresh_local();
                            // Refresh to local catalog and gallery
                            self.models.catalog =
                                mold_core::build_model_catalog(&self.config, None, false);
                            self.gallery.scanning = true;
                            self.spawn_gallery_scan();
                        } else {
                            // Normalize using same logic as CLI/MoldClient
                            let url = mold_core::client::normalize_host(&host);
                            self.generate.params.host = Some(url.clone());
                            self.connecting = true;
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
                Some(Popup::Info { .. }) => {
                    // Dismiss info popup on any key
                    self.close_popup();
                }
                Some(Popup::SettingsInput { key: sk, input, .. }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        let k = *sk;
                        let val = input.trim().to_string();
                        self.close_popup();
                        self.settings_apply_input(k, val);
                    }
                    KeyCode::Char(c) => input.push(c),
                    KeyCode::Backspace => {
                        input.pop();
                    }
                    _ => {}
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
                    // Determine which tab was clicked based on cumulative label widths.
                    // Tab labels: " {i} {label} " with 1-char divider, plus block padding.
                    let x = col.saturating_sub(self.layout.tab_bar.x + 2) as usize; // +2 for border + padding
                    let mut offset = 0;
                    for view in &View::ALL {
                        let tab_width = view.label().len() + 4; // " N Label " = label + " N  "
                        if x < offset + tab_width {
                            self.active_view = *view;
                            return;
                        }
                        offset += tab_width + 1; // +1 for divider
                    }
                    // Click past all tabs — select last view
                    self.active_view = View::Settings;
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
                    Some(Popup::ModelSelector { selected, .. })
                    | Some(Popup::UpscaleModelSelector { selected, .. }) => {
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
                })
                | Some(Popup::UpscaleModelSelector {
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

    /// Return names of all known upscaler models (downloaded or not).
    fn available_upscaler_models(&self) -> Vec<String> {
        let mut models: Vec<String> = mold_core::manifest::known_manifests()
            .iter()
            .filter(|m| m.is_upscaler())
            .map(|m| m.name.clone())
            .collect();
        // Sort: downloaded first, then undownloaded
        let config = &self.config;
        models.sort_by_key(|name| {
            let resolved = mold_core::manifest::resolve_model_name(name);
            let downloaded =
                config.models.contains_key(&resolved) || config.manifest_model_is_downloaded(name);
            if downloaded {
                0
            } else {
                1
            }
        });
        models
    }

    fn update_upscale_model_filter(&mut self) {
        // Collect available models first to avoid conflicting borrows with self.popup
        let all = self.available_upscaler_models();
        if let Some(Popup::UpscaleModelSelector {
            filter,
            selected,
            filtered,
        }) = &mut self.popup
        {
            let query = filter.to_lowercase();
            *filtered = all
                .into_iter()
                .filter(|name| name.to_lowercase().contains(&query))
                .collect();
            if *selected >= filtered.len() {
                *selected = filtered.len().saturating_sub(1);
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
                .filter(|m| m.is_generation_model() && m.name.to_lowercase().contains(&query))
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
                    View::Models => View::Settings,
                    View::Settings => View::Generate,
                };
            }
            Action::ViewPrev => {
                self.active_view = match self.active_view {
                    View::Generate => View::Settings,
                    View::Gallery => View::Generate,
                    View::Models => View::Gallery,
                    View::Settings => View::Models,
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
                View::Settings => self.settings_navigate(-1),
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
                View::Settings => self.settings_navigate(1),
            },
            Action::Increment => {
                if self.active_view == View::Settings {
                    self.settings_increment(1);
                } else {
                    self.increment_param(1);
                }
            }
            Action::Decrement => {
                if self.active_view == View::Settings {
                    self.settings_increment(-1);
                } else {
                    self.increment_param(-1);
                }
            }
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
                View::Settings => self.settings_confirm(),
            },
            Action::PullModel => {
                if self.active_view == View::Models {
                    if let Some(model) = self.models.catalog.get(self.models.selected) {
                        let model_name = model.name.clone();
                        let tx = self.bg_tx.clone();

                        if let Some(ref url) = self.server_url {
                            // Pull via server when connected remotely
                            let url = url.clone();
                            self.tokio_handle.spawn(async move {
                                let client = mold_core::MoldClient::new(&url);
                                let (progress_tx, mut progress_rx) =
                                    mpsc::unbounded_channel::<SseProgressEvent>();
                                let tx_fwd = tx.clone();
                                tokio::spawn(async move {
                                    while let Some(event) = progress_rx.recv().await {
                                        let _ = tx_fwd.send(BackgroundEvent::Progress(event));
                                    }
                                });
                                match client.pull_model_stream(&model_name, progress_tx).await {
                                    Ok(()) => {
                                        let _ = tx.send(BackgroundEvent::PullComplete(model_name));
                                    }
                                    Err(e) => {
                                        let _ = tx.send(BackgroundEvent::Error(format!(
                                            "Server pull failed: {e}"
                                        )));
                                    }
                                }
                            });
                        } else {
                            // Pull locally when no server connected
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
                self.sync_resource_info_mode();
            }
            Action::ShowHelp => {
                self.popup = Some(Popup::Help);
            }
            Action::Cancel => {
                if self.active_view == View::Gallery && self.upscale_in_progress {
                    // Cancel in-progress upscale. abort() cancels the outer async
                    // task so no UpscaleComplete event is sent, but the inner
                    // spawn_blocking thread (GPU inference) runs to completion —
                    // Tokio blocking threads have no cooperative cancellation.
                    if let Some(handle) = self.upscale_task.take() {
                        handle.abort();
                    }
                    self.upscale_in_progress = false;
                    self.upscale_tile_progress = None;
                    self.upscale_progress.clear();
                    self.generate.progress.log.push(ProgressLogEntry {
                        message: "Upscale cancelled".into(),
                        style: ProgressStyle::Warning,
                    });
                } else if self.active_view == View::Gallery
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
            Action::UpscaleImage => {
                if self.active_view == View::Gallery
                    && !self.upscale_in_progress
                    && self.gallery.entries.get(self.gallery.selected).is_some()
                {
                    let models = self.available_upscaler_models();
                    self.popup = Some(Popup::UpscaleModelSelector {
                        filter: String::new(),
                        selected: 0,
                        filtered: models,
                    });
                }
            }
            Action::RemoveModel => {
                if self.active_view == View::Models {
                    if let Some(model) = self.models.catalog.get(self.models.selected) {
                        if !model.downloaded {
                            return;
                        }
                        let name = model.info.name.clone();

                        // Block removal during active generation or pull
                        if self.generate.generating && self.generate.params.model == name {
                            self.generate.error_message =
                                Some("Cannot remove a model while generating".to_string());
                            return;
                        }
                        if mold_core::download::has_pulling_marker(&name) {
                            self.generate.error_message =
                                Some("Cannot remove a model while it is being pulled".to_string());
                            return;
                        }

                        let message = self.build_remove_model_message(&name);
                        self.popup = Some(Popup::Confirm {
                            message,
                            on_confirm: ConfirmAction::RemoveModel(name),
                        });
                    }
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
                p.batch = (p.batch as i32 + delta).max(1) as u32;
            }
            ParamField::Strength => {
                p.strength = (p.strength + delta as f64 * 0.05).clamp(0.0, 1.0);
            }
            ParamField::Frames => {
                p.frames = (p.frames as i32 + delta * 8).clamp(9, 257) as u32;
            }
            ParamField::Fps => {
                p.fps = (p.fps as i32 + delta).clamp(1, 60) as u32;
            }
            ParamField::ControlScale => {
                p.control_scale = (p.control_scale + delta as f64 * 0.1).clamp(0.0, 2.0);
            }
            ParamField::Format => {
                p.format = match p.format {
                    OutputFormat::Png => OutputFormat::Jpeg,
                    OutputFormat::Jpeg => OutputFormat::Gif,
                    OutputFormat::Gif => OutputFormat::Apng,
                    OutputFormat::Apng => OutputFormat::Webp,
                    OutputFormat::Webp => OutputFormat::Mp4,
                    OutputFormat::Mp4 => OutputFormat::Png,
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
            self.sync_resource_info_mode();
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
                // For video files, prefer the cached GIF preview (animated)
                let gif_path = crate::thumbnails::preview_gif_path(&entry.path);
                let load_path = if gif_path.is_file() {
                    &gif_path
                } else {
                    &entry.path
                };
                if let Ok(img) = image::open(load_path) {
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
        if idx < self.gallery.thumb_dimensions.len() {
            self.gallery.thumb_dimensions.remove(idx);
        }
        if idx < self.gallery.thumb_fixed_cache.len() {
            self.gallery.thumb_fixed_cache.remove(idx);
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

    /// Build a multi-line confirmation message for model removal showing
    /// disk space to be freed and any shared-file warnings.
    fn build_remove_model_message(&self, model_name: &str) -> String {
        let mut lines = vec![format!("Remove model '{model_name}'?")];

        // Build ref counts and classify files to compute accurate unique-only size
        let ref_counts = crate::backend::build_ref_counts(&self.config);
        let mut unique_bytes: u64 = 0;
        let mut shared_warnings: Vec<String> = Vec::new();

        if let Some(model_config) = self.config.models.get(model_name) {
            for path in model_config.all_file_paths() {
                let refs = ref_counts.get(&path).cloned().unwrap_or_default();
                let others: Vec<String> = refs
                    .into_iter()
                    .filter(|n| n.as_str() != model_name)
                    .collect();

                if others.is_empty() {
                    // Unique file — will be deleted
                    unique_bytes += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                } else {
                    // Shared file — kept, warn user
                    let filename = std::path::Path::new(&path)
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_else(|| path.clone());
                    shared_warnings.push(format!(
                        "  {} (shared with {})",
                        filename,
                        others.join(", ")
                    ));
                }
            }
        }

        if unique_bytes > 0 {
            lines.push(format!(
                "Disk space to free: ~{}",
                crate::ui::progress::format_bytes(unique_bytes)
            ));
        }

        if !shared_warnings.is_empty() {
            lines.push(String::new());
            lines.push("Shared files (kept):".to_string());
            lines.extend(shared_warnings);
        }

        lines.join("\n")
    }

    /// Dispatch a confirmed popup action.
    fn handle_confirm_action(&mut self, action: ConfirmAction) {
        match action {
            ConfirmAction::DeleteGalleryImage => {
                self.delete_selected_gallery_image();
            }
            ConfirmAction::RemoveModel(name) => {
                let tx = self.bg_tx.clone();
                let model_name = name.clone();
                self.tokio_handle.spawn_blocking(move || {
                    crate::backend::remove_model(model_name, tx);
                });
            }
        }
    }

    fn open_model_selector(&mut self) {
        let mut models: Vec<String> = self
            .models
            .catalog
            .iter()
            .filter(|m| m.is_generation_model())
            .map(|m| m.name.clone())
            .collect();
        // Sort: downloaded first, then undownloaded (preserving order within each group)
        let config = &self.config;
        models.sort_by_key(|name| {
            let resolved = mold_core::manifest::resolve_model_name(name);
            let downloaded =
                config.models.contains_key(&resolved) || config.manifest_model_is_downloaded(name);
            if downloaded {
                0
            } else {
                1
            }
        });
        self.popup = Some(Popup::ModelSelector {
            filter: String::new(),
            selected: 0,
            filtered: models,
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
                self.sync_resource_info_mode();
            }
            // Cycle format
            ParamField::Format => {
                self.generate.params.format = match self.generate.params.format {
                    OutputFormat::Png => OutputFormat::Jpeg,
                    OutputFormat::Jpeg => OutputFormat::Gif,
                    OutputFormat::Gif => OutputFormat::Apng,
                    OutputFormat::Apng => OutputFormat::Webp,
                    OutputFormat::Webp => OutputFormat::Mp4,
                    OutputFormat::Mp4 => OutputFormat::Png,
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

                // Use server catalog defaults when connected, local config otherwise
                if let Some(entry) = self
                    .server_url
                    .as_ref()
                    .and_then(|_| self.models.catalog.iter().find(|m| m.name == model))
                {
                    self.generate.params.width = entry.defaults.default_width;
                    self.generate.params.height = entry.defaults.default_height;
                    self.generate.params.steps = entry.defaults.default_steps;
                    self.generate.params.guidance = entry.defaults.default_guidance;
                } else {
                    let mc = self.config.resolved_model_config(&model);
                    self.generate.params.width = mc.effective_width(&self.config);
                    self.generate.params.height = mc.effective_height(&self.config);
                    self.generate.params.steps = mc.effective_steps(&self.config);
                    self.generate.params.guidance = mc.effective_guidance();
                }
                self.generate.params.seed = None;
                self.generate.params.seed_mode = SeedMode::Random;
                self.generate.params.batch = 1;
                self.generate.params.format = OutputFormat::Png;
                self.generate.params.scheduler = None;
                self.generate.params.lora_path = None;
                self.generate.params.lora_scale = 1.0;
                self.generate.params.expand = false;
                self.generate.params.offload = false;
                self.generate.params.frames = 25;
                self.generate.params.fps = 24;
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

    // ── Settings view helpers ─────────────────────────────────────────

    /// Build the flat list of settings rows from current config state.
    #[allow(clippy::vec_init_then_push)]
    pub fn build_settings_rows(&self) -> Vec<SettingsRow> {
        let mut rows = Vec::new();

        // ── General ─────────────────────────────────────────────
        rows.push(SettingsRow::SectionHeader {
            name: "General".into(),
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::DefaultModel,
            label: "Model",
            field_type: SettingsFieldType::Text,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::ModelsDir,
            label: "Models Dir",
            field_type: SettingsFieldType::Path,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::OutputDir,
            label: "Output Dir",
            field_type: SettingsFieldType::Path,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::ServerPort,
            label: "Port",
            field_type: SettingsFieldType::Number {
                min: 1.0,
                max: 65535.0,
                step: 1.0,
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::DefaultWidth,
            label: "Width",
            field_type: SettingsFieldType::Number {
                min: 64.0,
                max: 8192.0,
                step: 64.0,
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::DefaultHeight,
            label: "Height",
            field_type: SettingsFieldType::Number {
                min: 64.0,
                max: 8192.0,
                step: 64.0,
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::DefaultSteps,
            label: "Steps",
            field_type: SettingsFieldType::Number {
                min: 1.0,
                max: 1000.0,
                step: 1.0,
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::EmbedMetadata,
            label: "Metadata",
            field_type: SettingsFieldType::Bool,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::T5Variant,
            label: "T5 Variant",
            field_type: SettingsFieldType::Toggle {
                options: vec!["auto", "fp16", "q8", "q6", "q5", "q4", "q3"],
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::Qwen3Variant,
            label: "Qwen3 Var.",
            field_type: SettingsFieldType::Toggle {
                options: vec!["auto", "bf16", "q8", "q6", "iq4", "q3"],
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::DefaultNegativePrompt,
            label: "Neg. Prompt",
            field_type: SettingsFieldType::Text,
        });

        // ── Expand ──────────────────────────────────────────────
        rows.push(SettingsRow::SectionHeader {
            name: "Expand".into(),
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::ExpandEnabled,
            label: "Enabled",
            field_type: SettingsFieldType::Bool,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::ExpandBackend,
            label: "Backend",
            field_type: SettingsFieldType::Text,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::ExpandModel,
            label: "Model",
            field_type: SettingsFieldType::Text,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::ExpandApiModel,
            label: "API Model",
            field_type: SettingsFieldType::Text,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::ExpandTemperature,
            label: "Temp.",
            field_type: SettingsFieldType::Number {
                min: 0.0,
                max: 2.0,
                step: 0.1,
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::ExpandTopP,
            label: "Top P",
            field_type: SettingsFieldType::Number {
                min: 0.0,
                max: 1.0,
                step: 0.05,
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::ExpandMaxTokens,
            label: "Max Tokens",
            field_type: SettingsFieldType::Number {
                min: 1.0,
                max: 65535.0,
                step: 64.0,
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::ExpandThinking,
            label: "Thinking",
            field_type: SettingsFieldType::Bool,
        });

        // ── Logging ─────────────────────────────────────────────
        rows.push(SettingsRow::SectionHeader {
            name: "Logging".into(),
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::LogLevel,
            label: "Level",
            field_type: SettingsFieldType::Toggle {
                options: vec!["trace", "debug", "info", "warn", "error"],
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::LogFile,
            label: "File Log",
            field_type: SettingsFieldType::Bool,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::LogDir,
            label: "Log Dir",
            field_type: SettingsFieldType::Path,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::LogMaxDays,
            label: "Max Days",
            field_type: SettingsFieldType::Number {
                min: 1.0,
                max: 3650.0,
                step: 1.0,
            },
        });

        // ── Model Defaults ──────────────────────────────────────
        if !self.config.models.is_empty() {
            let model_name = self.settings.selected_model.clone().unwrap_or_else(|| {
                self.config
                    .models
                    .keys()
                    .next()
                    .cloned()
                    .unwrap_or_default()
            });

            rows.push(SettingsRow::SectionHeader {
                name: format!("Model Defaults \u{2500}\u{2500} {model_name} "),
            });
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelSelector,
                label: "Model",
                field_type: SettingsFieldType::Toggle {
                    options: Vec::new(), // handled specially in cycle logic
                },
            });
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelSteps,
                label: "Steps",
                field_type: SettingsFieldType::Number {
                    min: 1.0,
                    max: 1000.0,
                    step: 1.0,
                },
            });
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelGuidance,
                label: "Guidance",
                field_type: SettingsFieldType::Number {
                    min: 0.0,
                    max: 100.0,
                    step: 0.5,
                },
            });
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelWidth,
                label: "Width",
                field_type: SettingsFieldType::Number {
                    min: 64.0,
                    max: 8192.0,
                    step: 64.0,
                },
            });
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelHeight,
                label: "Height",
                field_type: SettingsFieldType::Number {
                    min: 64.0,
                    max: 8192.0,
                    step: 64.0,
                },
            });
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelScheduler,
                label: "Scheduler",
                field_type: SettingsFieldType::Toggle {
                    options: vec!["(none)", "ddim", "euler-ancestral", "uni-pc"],
                },
            });
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelNegativePrompt,
                label: "Neg. Prompt",
                field_type: SettingsFieldType::Text,
            });
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelLora,
                label: "LoRA",
                field_type: SettingsFieldType::Path,
            });
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelLoraScale,
                label: "LoRA Scale",
                field_type: SettingsFieldType::Number {
                    min: 0.0,
                    max: 2.0,
                    step: 0.1,
                },
            });

            // Read-only paths
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelTransformer,
                label: "Transformer",
                field_type: SettingsFieldType::ReadOnly,
            });
            rows.push(SettingsRow::Field {
                key: SettingsKey::ModelVae,
                label: "VAE",
                field_type: SettingsFieldType::ReadOnly,
            });
        }

        rows
    }

    /// Get the display value for a settings key.
    pub fn settings_display_value(&self, key: &SettingsKey) -> String {
        let cfg = &self.config;
        // For model defaults, use resolved config (merges manifest defaults)
        // so the display shows effective runtime values, not raw None/Some.
        let resolved_model = self
            .settings
            .selected_model
            .as_ref()
            .map(|name| cfg.resolved_model_config(name));
        // Raw model config for path fields (those come from config, not manifest)
        let model_cfg = self
            .settings
            .selected_model
            .as_ref()
            .and_then(|name| cfg.models.get(name));

        match key {
            SettingsKey::DefaultModel => cfg.default_model.clone(),
            SettingsKey::ModelsDir => cfg.models_dir.clone(),
            SettingsKey::OutputDir => cfg
                .output_dir
                .as_deref()
                .unwrap_or("~/.mold/output")
                .to_string(),
            SettingsKey::ServerPort => cfg.server_port.to_string(),
            SettingsKey::DefaultWidth => cfg.default_width.to_string(),
            SettingsKey::DefaultHeight => cfg.default_height.to_string(),
            SettingsKey::DefaultSteps => cfg.default_steps.to_string(),
            SettingsKey::EmbedMetadata => if cfg.embed_metadata { "on" } else { "off" }.into(),
            SettingsKey::T5Variant => cfg.t5_variant.as_deref().unwrap_or("auto").to_string(),
            SettingsKey::Qwen3Variant => cfg.qwen3_variant.as_deref().unwrap_or("auto").to_string(),
            SettingsKey::DefaultNegativePrompt => cfg
                .default_negative_prompt
                .as_deref()
                .unwrap_or("(none)")
                .to_string(),
            // Expand
            SettingsKey::ExpandEnabled => if cfg.expand.enabled { "on" } else { "off" }.into(),
            SettingsKey::ExpandBackend => cfg.expand.backend.clone(),
            SettingsKey::ExpandModel => cfg.expand.model.clone(),
            SettingsKey::ExpandApiModel => cfg.expand.api_model.clone(),
            SettingsKey::ExpandTemperature => format!("{:.1}", cfg.expand.temperature),
            SettingsKey::ExpandTopP => format!("{:.2}", cfg.expand.top_p),
            SettingsKey::ExpandMaxTokens => cfg.expand.max_tokens.to_string(),
            SettingsKey::ExpandThinking => if cfg.expand.thinking { "on" } else { "off" }.into(),
            // Logging
            SettingsKey::LogLevel => cfg.logging.level.clone(),
            SettingsKey::LogFile => if cfg.logging.file { "on" } else { "off" }.into(),
            SettingsKey::LogDir => cfg
                .logging
                .dir
                .as_deref()
                .unwrap_or("~/.mold/logs")
                .to_string(),
            SettingsKey::LogMaxDays => cfg.logging.max_days.to_string(),
            // Model defaults
            SettingsKey::ModelSelector => self
                .settings
                .selected_model
                .as_deref()
                .unwrap_or("(none)")
                .to_string(),
            SettingsKey::ModelSteps => resolved_model
                .as_ref()
                .and_then(|m| m.default_steps)
                .map(|v| v.to_string())
                .unwrap_or_else(|| cfg.default_steps.to_string()),
            SettingsKey::ModelGuidance => resolved_model
                .as_ref()
                .and_then(|m| m.default_guidance)
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "0.0".into()),
            SettingsKey::ModelWidth => resolved_model
                .as_ref()
                .and_then(|m| m.default_width)
                .map(|v| v.to_string())
                .unwrap_or_else(|| cfg.default_width.to_string()),
            SettingsKey::ModelHeight => resolved_model
                .as_ref()
                .and_then(|m| m.default_height)
                .map(|v| v.to_string())
                .unwrap_or_else(|| cfg.default_height.to_string()),
            SettingsKey::ModelScheduler => resolved_model
                .as_ref()
                .and_then(|m| m.scheduler)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "(none)".into()),
            SettingsKey::ModelNegativePrompt => model_cfg
                .and_then(|m| m.negative_prompt.as_deref())
                .unwrap_or("(none)")
                .to_string(),
            SettingsKey::ModelLora => model_cfg
                .and_then(|m| m.lora.as_deref())
                .unwrap_or("(none)")
                .to_string(),
            SettingsKey::ModelLoraScale => model_cfg
                .and_then(|m| m.lora_scale)
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "1.0".into()),
            SettingsKey::ModelTransformer => model_cfg
                .and_then(|m| m.transformer.as_deref())
                .unwrap_or("(not set)")
                .to_string(),
            SettingsKey::ModelVae => model_cfg
                .and_then(|m| m.vae.as_deref())
                .unwrap_or("(not set)")
                .to_string(),
        }
    }

    /// Return the env var name if it overrides the given settings key.
    pub fn settings_env_override(key: &SettingsKey) -> Option<&'static str> {
        let var = match key {
            SettingsKey::DefaultModel => "MOLD_DEFAULT_MODEL",
            SettingsKey::ModelsDir => "MOLD_MODELS_DIR",
            SettingsKey::OutputDir => "MOLD_OUTPUT_DIR",
            SettingsKey::EmbedMetadata => "MOLD_EMBED_METADATA",
            SettingsKey::T5Variant => "MOLD_T5_VARIANT",
            SettingsKey::Qwen3Variant => "MOLD_QWEN3_VARIANT",
            SettingsKey::ExpandEnabled => "MOLD_EXPAND",
            SettingsKey::ExpandBackend => "MOLD_EXPAND_BACKEND",
            SettingsKey::ExpandModel | SettingsKey::ExpandApiModel => "MOLD_EXPAND_MODEL",
            SettingsKey::ExpandTemperature => "MOLD_EXPAND_TEMPERATURE",
            SettingsKey::ExpandThinking => "MOLD_EXPAND_THINKING",
            _ => return None,
        };
        if std::env::var(var).is_ok() {
            Some(var)
        } else {
            None
        }
    }

    /// Navigate up (delta=-1) or down (delta=1) in the settings list, skipping headers.
    fn settings_navigate(&mut self, delta: i32) {
        let rows = self.build_settings_rows();
        if rows.is_empty() {
            return;
        }
        let current = self.settings.row_index;
        let len = rows.len();
        let mut next = current;
        loop {
            let candidate = next as i32 + delta;
            if candidate < 0 || candidate >= len as i32 {
                break;
            }
            next = candidate as usize;
            if rows[next].is_field() {
                self.settings.row_index = next;
                break;
            }
        }
    }

    /// Adjust the current settings field by delta (+1 or -1).
    fn settings_increment(&mut self, delta: i32) {
        let rows = self.build_settings_rows();
        let row = match rows.get(self.settings.row_index) {
            Some(r) => r,
            None => return,
        };
        let (key, field_type) = match row {
            SettingsRow::Field {
                key, field_type, ..
            } => (*key, field_type.clone()),
            _ => return,
        };

        // Handle ModelSelector specially — cycles through configured models
        if key == SettingsKey::ModelSelector {
            self.settings_cycle_model(delta);
            return;
        }

        match field_type {
            SettingsFieldType::Number { min, max, step } => {
                self.settings_adjust_number(key, delta as f64 * step, min, max);
            }
            SettingsFieldType::Toggle { options } => {
                if !options.is_empty() {
                    self.settings_cycle_toggle(key, &options, delta);
                }
            }
            SettingsFieldType::Bool => {
                self.settings_toggle_bool(key);
            }
            _ => {}
        }
    }

    fn settings_adjust_number(&mut self, key: SettingsKey, delta: f64, min: f64, max: f64) {
        let cfg = &mut self.config;
        match key {
            SettingsKey::ServerPort => {
                cfg.server_port = (cfg.server_port as f64 + delta).clamp(min, max) as u16;
            }
            SettingsKey::DefaultWidth => {
                cfg.default_width = (cfg.default_width as f64 + delta).clamp(min, max) as u32;
            }
            SettingsKey::DefaultHeight => {
                cfg.default_height = (cfg.default_height as f64 + delta).clamp(min, max) as u32;
            }
            SettingsKey::DefaultSteps => {
                cfg.default_steps = (cfg.default_steps as f64 + delta).clamp(min, max) as u32;
            }
            SettingsKey::ExpandTemperature => {
                cfg.expand.temperature = (cfg.expand.temperature + delta).clamp(min, max);
            }
            SettingsKey::ExpandTopP => {
                cfg.expand.top_p = (cfg.expand.top_p + delta).clamp(min, max);
            }
            SettingsKey::ExpandMaxTokens => {
                cfg.expand.max_tokens =
                    (cfg.expand.max_tokens as f64 + delta).clamp(min, max) as u32;
            }
            SettingsKey::LogMaxDays => {
                cfg.logging.max_days = (cfg.logging.max_days as f64 + delta).clamp(min, max) as u32;
            }
            SettingsKey::ModelSteps => {
                if let Some(name) = &self.settings.selected_model {
                    let resolved = self.config.resolved_model_config(name);
                    let cur = resolved.effective_steps(&self.config) as f64;
                    if let Some(mc) = self.config.models.get_mut(name) {
                        mc.default_steps = Some((cur + delta).clamp(min, max) as u32);
                    }
                }
                self.save_config();
                return;
            }
            SettingsKey::ModelGuidance => {
                if let Some(name) = &self.settings.selected_model {
                    let resolved = self.config.resolved_model_config(name);
                    let cur = resolved.effective_guidance();
                    if let Some(mc) = self.config.models.get_mut(name) {
                        mc.default_guidance = Some((cur + delta).clamp(min, max));
                    }
                }
                self.save_config();
                return;
            }
            SettingsKey::ModelWidth => {
                if let Some(name) = &self.settings.selected_model {
                    let resolved = self.config.resolved_model_config(name);
                    let cur = resolved.effective_width(&self.config) as f64;
                    if let Some(mc) = self.config.models.get_mut(name) {
                        mc.default_width = Some((cur + delta).clamp(min, max) as u32);
                    }
                }
                self.save_config();
                return;
            }
            SettingsKey::ModelHeight => {
                if let Some(name) = &self.settings.selected_model {
                    let resolved = self.config.resolved_model_config(name);
                    let cur = resolved.effective_height(&self.config) as f64;
                    if let Some(mc) = self.config.models.get_mut(name) {
                        mc.default_height = Some((cur + delta).clamp(min, max) as u32);
                    }
                }
                self.save_config();
                return;
            }
            SettingsKey::ModelLoraScale => {
                if let Some(name) = &self.settings.selected_model {
                    if let Some(mc) = self.config.models.get_mut(name) {
                        let cur = mc.lora_scale.unwrap_or(1.0);
                        mc.lora_scale = Some((cur + delta).clamp(min, max));
                    }
                }
                self.save_config();
                return;
            }
            _ => return,
        }
        self.save_config();
    }

    fn settings_cycle_toggle(&mut self, key: SettingsKey, options: &[&str], delta: i32) {
        let current = self.settings_display_value(&key);
        let idx = options.iter().position(|&o| o == current).unwrap_or(0);
        let next = (idx as i32 + delta).rem_euclid(options.len() as i32) as usize;
        let value = options[next].to_string();

        match key {
            SettingsKey::T5Variant => {
                self.config.t5_variant = if value == "auto" { None } else { Some(value) };
            }
            SettingsKey::Qwen3Variant => {
                self.config.qwen3_variant = if value == "auto" { None } else { Some(value) };
            }
            SettingsKey::LogLevel => {
                self.config.logging.level = value;
            }
            SettingsKey::ModelScheduler => {
                if let Some(name) = &self.settings.selected_model {
                    if let Some(mc) = self.config.models.get_mut(name) {
                        mc.scheduler = match options[next] {
                            "ddim" => Some(Scheduler::Ddim),
                            "euler-ancestral" => Some(Scheduler::EulerAncestral),
                            "uni-pc" => Some(Scheduler::UniPc),
                            _ => None,
                        };
                    }
                }
            }
            _ => return,
        }
        self.save_config();
    }

    fn settings_toggle_bool(&mut self, key: SettingsKey) {
        match key {
            SettingsKey::EmbedMetadata => self.config.embed_metadata = !self.config.embed_metadata,
            SettingsKey::ExpandEnabled => self.config.expand.enabled = !self.config.expand.enabled,
            SettingsKey::ExpandThinking => {
                self.config.expand.thinking = !self.config.expand.thinking;
            }
            SettingsKey::LogFile => self.config.logging.file = !self.config.logging.file,
            _ => return,
        }
        self.save_config();
    }

    fn settings_cycle_model(&mut self, delta: i32) {
        let names: Vec<String> = self.config.models.keys().cloned().collect();
        if names.is_empty() {
            return;
        }
        let idx = self
            .settings
            .selected_model
            .as_ref()
            .and_then(|current| names.iter().position(|n| n == current))
            .unwrap_or(0);
        let next = (idx as i32 + delta).rem_euclid(names.len() as i32) as usize;
        self.settings.selected_model = Some(names[next].clone());
    }

    /// Handle Enter on the current settings field.
    fn settings_confirm(&mut self) {
        let rows = self.build_settings_rows();
        let row = match rows.get(self.settings.row_index) {
            Some(r) => r,
            None => return,
        };
        let (key, field_type) = match row {
            SettingsRow::Field {
                key, field_type, ..
            } => (*key, field_type.clone()),
            _ => return,
        };

        match field_type {
            SettingsFieldType::Text | SettingsFieldType::Path => {
                let label = match row {
                    SettingsRow::Field { label, .. } => *label,
                    _ => "",
                };
                let current = self.settings_display_value(&key);
                let input = if current == "(none)" || current == "(not set)" {
                    String::new()
                } else {
                    current
                };
                self.popup = Some(Popup::SettingsInput {
                    key,
                    input,
                    label: label.to_string(),
                });
            }
            SettingsFieldType::Bool => {
                self.settings_toggle_bool(key);
            }
            SettingsFieldType::Toggle { ref options } => {
                if key == SettingsKey::ModelSelector {
                    self.settings_cycle_model(1);
                } else if !options.is_empty() {
                    self.settings_cycle_toggle(key, options, 1);
                }
            }
            SettingsFieldType::Number { .. } => {
                // No-op for Enter on numeric fields (use +/-)
            }
            SettingsFieldType::ReadOnly => {}
        }
    }

    /// Apply a text/path popup result to the config and save.
    fn settings_apply_input(&mut self, key: SettingsKey, value: String) {
        let val = if value.is_empty() { None } else { Some(value) };
        match key {
            SettingsKey::DefaultModel => {
                if let Some(v) = val {
                    self.config.default_model = v;
                }
            }
            SettingsKey::ModelsDir => {
                if let Some(v) = val {
                    self.config.models_dir = v;
                }
            }
            SettingsKey::OutputDir => {
                self.config.output_dir = val;
            }
            SettingsKey::DefaultNegativePrompt => {
                self.config.default_negative_prompt = val;
            }
            SettingsKey::ExpandBackend => {
                if let Some(v) = val {
                    self.config.expand.backend = v;
                }
            }
            SettingsKey::ExpandModel => {
                if let Some(v) = val {
                    self.config.expand.model = v;
                }
            }
            SettingsKey::ExpandApiModel => {
                if let Some(v) = val {
                    self.config.expand.api_model = v;
                }
            }
            SettingsKey::LogDir => {
                self.config.logging.dir = val;
            }
            SettingsKey::ModelNegativePrompt => {
                if let Some(name) = &self.settings.selected_model {
                    if let Some(mc) = self.config.models.get_mut(name) {
                        mc.negative_prompt = val;
                    }
                }
            }
            SettingsKey::ModelLora => {
                if let Some(name) = &self.settings.selected_model {
                    if let Some(mc) = self.config.models.get_mut(name) {
                        mc.lora = val;
                    }
                }
            }
            _ => return,
        }
        self.save_config();
    }

    /// Save config to disk, storing any error in settings state.
    fn save_config(&mut self) {
        #[cfg(test)]
        if self.settings.skip_save {
            return;
        }
        if let Err(e) = self.config.save() {
            self.settings.save_error = Some(format!("Save failed: {e}"));
        } else {
            self.settings.save_error = None;
        }
    }

    fn start_generation(&mut self) {
        let prompt_text = self.generate.prompt.lines().join("\n").trim().to_string();
        if prompt_text.is_empty() {
            self.generate.error_message = Some("Prompt is empty".to_string());
            return;
        }

        self.generate.generating = true;
        self.generate.batch_remaining = self.generate.params.batch;
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
                    self.generate.batch_remaining = self.generate.batch_remaining.saturating_sub(1);
                    if self.generate.batch_remaining == 0 {
                        self.generate.generating = false;
                    }
                    self.generate.last_seed = Some(response.seed_used);
                    self.generate.last_generation_time_ms = Some(response.generation_time_ms);

                    // Use the model name from the response (server is source of
                    // truth). The UI params may have changed if the user switched
                    // models while generation was running.
                    let actual_model = response.model.clone();

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
                        let ext = img_data.format.extension();
                        let filename = mold_core::default_output_filename(
                            &actual_model,
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

                    // Handle video output: save primary file + cache GIF preview
                    if let Some(ref video) = response.video {
                        let ext = video.format.extension();
                        let filename =
                            mold_core::default_output_filename(&actual_model, ts_secs, ext, 1, 0);
                        if let Some(ref dir) = output_dir {
                            let path = dir.join(&filename);
                            if std::fs::write(&path, &video.data).is_ok() {
                                saved_path = path.clone();
                                // Cache the GIF preview for gallery detail view
                                if !video.gif_preview.is_empty() {
                                    crate::thumbnails::save_preview_gif(&video.gif_preview, &path)
                                        .ok();
                                }
                                // Generate a still thumbnail for the gallery grid
                                if !video.thumbnail.is_empty() {
                                    crate::thumbnails::save_thumbnail_bytes(
                                        &video.thumbnail,
                                        &path,
                                    )
                                    .ok();
                                }
                            }
                        }
                        // Show GIF preview in the generate viewport (animated)
                        if !video.gif_preview.is_empty() {
                            if let Ok(img) = image::load_from_memory(&video.gif_preview) {
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
                    self.save_session();
                    Config::write_last_model(&actual_model);

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
                        model: actual_model.clone(),
                        timestamp: ts,
                    });

                    // Add to gallery (most recent first) with full metadata
                    // Use video dimensions if no images (video-only response)
                    let (entry_width, entry_height) = if let Some(img) = response.images.first() {
                        (img.width, img.height)
                    } else if let Some(ref video) = response.video {
                        (video.width, video.height)
                    } else {
                        (self.generate.params.width, self.generate.params.height)
                    };
                    if !response.images.is_empty() || response.video.is_some() {
                        let meta = mold_core::OutputMetadata {
                            prompt: prompt_text,
                            negative_prompt: if neg_text.is_empty() {
                                None
                            } else {
                                Some(neg_text)
                            },
                            original_prompt: None,
                            model: actual_model,
                            seed: response.seed_used,
                            steps: self.generate.params.steps,
                            guidance: self.generate.params.guidance,
                            width: entry_width,
                            height: entry_height,
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
                            frames: response.video.as_ref().map(|v| v.frames),
                            fps: response.video.as_ref().map(|v| v.fps),
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
                        self.gallery.thumb_dimensions.insert(0, None);
                        self.gallery.thumb_fixed_cache.insert(0, None);

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
                    self.generate.batch_remaining = 0;
                    self.generate.error_message = Some(msg);
                }
                BackgroundEvent::GalleryScanComplete(entries) => {
                    self.gallery.thumbnail_states = vec![None; entries.len()];
                    self.gallery.thumb_dimensions = vec![None; entries.len()];
                    self.gallery.thumb_fixed_cache = vec![None; entries.len()];
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
                                    let fetched = if let Ok(data) =
                                        client.get_gallery_thumbnail(&filename).await
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
                                        let cache_path =
                                            crate::gallery_scan::image_cache_dir().join(&filename);
                                        if cache_path.is_file() {
                                            let key = path;
                                            tokio::task::spawn_blocking(move || {
                                                crate::thumbnails::generate_thumbnail_from_cached(
                                                    &cache_path,
                                                    &key,
                                                )
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
                    self.gallery.thumb_dimensions = vec![None; len];
                    self.gallery.thumb_fixed_cache = vec![None; len];
                }
                BackgroundEvent::ServerConnected { url, models } => {
                    self.connecting = false;
                    self.server_url = Some(url.clone());
                    self.models.catalog = models.clone();
                    self.models.selected = 0;
                    // Auto-switch to auto mode
                    if self.generate.params.inference_mode == InferenceMode::Local {
                        self.generate.params.inference_mode = InferenceMode::Auto;
                    }
                    self.generate.visible_fields = ParamField::visible_fields(
                        &self.generate.capabilities,
                        self.generate.params.inference_mode,
                    );
                    // Apply model defaults from server catalog
                    self.apply_remote_model_defaults(&models);
                    self.generate.progress.log.push(ProgressLogEntry {
                        message: format!("Connected to {url}"),
                        style: ProgressStyle::Done,
                    });
                    // Re-scan gallery from the (now-connected) server
                    self.gallery.scanning = true;
                    self.spawn_gallery_scan();
                    // Trigger immediate server status fetch for resource info
                    self.spawn_server_status_fetch();
                }
                BackgroundEvent::ServerUnreachable(msg) => {
                    self.connecting = false;
                    self.generate.progress.log.push(ProgressLogEntry {
                        message: format!("Server unreachable: {msg}"),
                        style: ProgressStyle::Error,
                    });
                    // Revert host — don't set server_url
                    self.generate.params.host = self.server_url.clone();
                    // Fall back to local resource info
                    self.resource_info.clear_server_status();
                    self.resource_info.refresh_local();
                }
                BackgroundEvent::PullComplete(model) => {
                    self.generate.progress.log.push(ProgressLogEntry {
                        message: format!("Pull complete: {model}"),
                        style: ProgressStyle::Done,
                    });
                    // Refresh the catalog (from server when connected, local otherwise)
                    if let Some(ref url) = self.server_url {
                        let url = url.clone();
                        let tx = self.bg_tx.clone();
                        self.tokio_handle.spawn(async move {
                            let client = mold_core::MoldClient::new(&url);
                            if let Ok(models) = client.list_models_extended().await {
                                let _ = tx.send(BackgroundEvent::ServerConnected { url, models });
                            }
                        });
                    } else {
                        self.config = Config::load_or_default();
                        self.models.catalog =
                            mold_core::build_model_catalog(&self.config, None, false);
                    }
                }
                BackgroundEvent::ModelRemoveComplete(model) => {
                    self.generate.progress.log.push(ProgressLogEntry {
                        message: format!("Removed model: {model}"),
                        style: ProgressStyle::Done,
                    });
                    // Refresh config and catalog
                    self.config = Config::load_or_default();
                    self.models.catalog = mold_core::build_model_catalog(&self.config, None, false);
                    // Clamp selected index
                    if !self.models.catalog.is_empty()
                        && self.models.selected >= self.models.catalog.len()
                    {
                        self.models.selected = self.models.catalog.len() - 1;
                    }
                }
                BackgroundEvent::ModelRemoveFailed(msg) => {
                    self.generate.progress.log.push(ProgressLogEntry {
                        message: format!("Remove failed: {msg}"),
                        style: ProgressStyle::Error,
                    });
                }
                BackgroundEvent::UpscaleDownloadProgress(event) => {
                    reduce_progress_state(&mut self.upscale_progress, event);
                }
                BackgroundEvent::UpscaleProgress { tile, total } => {
                    self.upscale_tile_progress = Some((tile, total));
                }
                BackgroundEvent::UpscaleComplete {
                    image_data,
                    source_path,
                    model,
                    scale_factor,
                    original_width,
                    original_height,
                    upscale_time_ms,
                } => {
                    self.upscale_in_progress = false;
                    self.upscale_task = None;
                    self.upscale_tile_progress = None;
                    self.upscale_progress.clear();

                    let upscaled_w = original_width * scale_factor;
                    let upscaled_h = original_height * scale_factor;

                    // Save upscaled image to output directory
                    let output_dir = if self.config.is_output_disabled() {
                        None
                    } else {
                        let dir = self.config.effective_output_dir();
                        let _ = std::fs::create_dir_all(&dir);
                        Some(dir)
                    };

                    let stem = source_path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy();
                    let filename = format!("{stem}_upscaled_{scale_factor}x.png");

                    let saved_path = if let Some(ref dir) = output_dir {
                        let path = dir.join(&filename);
                        if let Err(e) = std::fs::write(&path, &image_data) {
                            self.generate.error_message =
                                Some(format!("Failed to save upscaled image: {e}"));
                            return;
                        }
                        path
                    } else {
                        // No output dir — nowhere to save
                        self.generate.progress.log.push(ProgressLogEntry {
                            message: format!(
                                "Upscaled {original_width}x{original_height} -> {upscaled_w}x{upscaled_h} ({scale_factor}x, {:.1}s) — output dir disabled",
                                upscale_time_ms as f64 / 1000.0
                            ),
                            style: ProgressStyle::Warning,
                        });
                        return;
                    };

                    self.generate.progress.log.push(ProgressLogEntry {
                        message: format!(
                            "Upscaled {original_width}x{original_height} -> {upscaled_w}x{upscaled_h} ({scale_factor}x, {:.1}s)",
                            upscale_time_ms as f64 / 1000.0
                        ),
                        style: ProgressStyle::Done,
                    });

                    // Insert new gallery entry at position 0
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);

                    // Carry over source metadata where applicable
                    let source_meta = self
                        .gallery
                        .entries
                        .iter()
                        .find(|e| e.path == source_path)
                        .map(|e| e.metadata.clone());

                    let meta = mold_core::OutputMetadata {
                        prompt: source_meta
                            .as_ref()
                            .map(|m| m.prompt.clone())
                            .unwrap_or_default(),
                        negative_prompt: source_meta
                            .as_ref()
                            .and_then(|m| m.negative_prompt.clone()),
                        original_prompt: source_meta
                            .as_ref()
                            .and_then(|m| m.original_prompt.clone()),
                        model,
                        seed: source_meta.as_ref().map(|m| m.seed).unwrap_or(0),
                        steps: source_meta.as_ref().map(|m| m.steps).unwrap_or(0),
                        guidance: source_meta.as_ref().map(|m| m.guidance).unwrap_or(0.0),
                        width: upscaled_w,
                        height: upscaled_h,
                        strength: None,
                        scheduler: source_meta.as_ref().and_then(|m| m.scheduler),
                        lora: source_meta.as_ref().and_then(|m| m.lora.clone()),
                        lora_scale: source_meta.as_ref().and_then(|m| m.lora_scale),
                        frames: None,
                        fps: None,
                        version: mold_core::build_info::VERSION.to_string(),
                    };

                    self.gallery.entries.insert(
                        0,
                        GalleryEntry {
                            path: saved_path.clone(),
                            metadata: meta,
                            generation_time_ms: Some(upscale_time_ms),
                            timestamp: ts,
                            server_url: None,
                        },
                    );
                    self.gallery.thumbnail_states.insert(0, None);
                    self.gallery.thumb_dimensions.insert(0, None);
                    self.gallery.thumb_fixed_cache.insert(0, None);
                    self.gallery.selected = 0;

                    // Generate thumbnail in background
                    self.tokio_handle.spawn(async move {
                        tokio::task::spawn_blocking(move || {
                            crate::thumbnails::generate_thumbnail(&saved_path).ok();
                        })
                        .await
                        .ok();
                    });
                }
                BackgroundEvent::UpscaleFailed(msg) => {
                    self.upscale_in_progress = false;
                    self.upscale_task = None;
                    self.upscale_tile_progress = None;
                    self.upscale_progress.clear();
                    self.generate.error_message = Some(format!("Upscale failed: {msg}"));
                }
                BackgroundEvent::ServerStatusUpdate(Some(status)) => {
                    self.resource_info.update_from_server_status(*status);
                }
                BackgroundEvent::ServerStatusUpdate(None) => {
                    // Server became unreachable — clear stale remote info
                    self.resource_info.clear_server_status();
                }
            }
        }
    }

    fn handle_progress(&mut self, event: SseProgressEvent) {
        let refresh_catalog = reduce_progress_state(&mut self.generate.progress, event);
        if refresh_catalog {
            // Refresh config and catalog after pull
            self.config = Config::load_or_default();
            self.models.catalog = mold_core::build_model_catalog(&self.config, None, false);
        }
    }
}

fn reduce_progress_state(progress: &mut ProgressState, event: SseProgressEvent) -> bool {
    match event {
        SseProgressEvent::StageStart { name } => {
            progress.current_stage = Some(name);
            // Reset transient bars when the stream moves into a new phase.
            progress.clear_download();
            progress.clear_weight();
        }
        SseProgressEvent::StageDone { name, elapsed_ms } => {
            progress.current_stage = None;
            progress.log.push(ProgressLogEntry {
                message: format!("{name} [{:.1}s]", elapsed_ms as f64 / 1000.0),
                style: ProgressStyle::Done,
            });
        }
        SseProgressEvent::Info { message } => {
            // Download status messages go to the stage spinner only (not the log)
            // to avoid duplicate display.
            if message.contains("pulling") || message.contains("Checking") {
                // These are status-only messages — show as spinner, not log
                progress.downloading = true;
                progress.current_stage = Some(message);
            } else if message.contains("Verifying") {
                // Verification messages: show as spinner AND log entry
                progress.downloading = true;
                progress.current_stage = Some(message.clone());
                progress.log.push(ProgressLogEntry {
                    message,
                    style: ProgressStyle::Info,
                });
            } else {
                progress.log.push(ProgressLogEntry {
                    message,
                    style: ProgressStyle::Info,
                });
            }
        }
        SseProgressEvent::CacheHit { resource } => {
            progress.log.push(ProgressLogEntry {
                message: format!("{resource} [cache hit]"),
                style: ProgressStyle::Done,
            });
        }
        SseProgressEvent::DenoiseStep {
            step,
            total,
            elapsed_ms,
        } => {
            progress.denoise_step = step;
            progress.denoise_total = total;
            progress.denoise_elapsed_ms = elapsed_ms;
        }
        SseProgressEvent::WeightLoad {
            bytes_loaded,
            bytes_total,
            component,
        } => {
            progress.weight_loaded = bytes_loaded;
            progress.weight_total = bytes_total;
            progress.weight_component = component;
        }
        SseProgressEvent::DownloadProgress {
            filename,
            bytes_downloaded,
            bytes_total,
            batch_bytes_downloaded,
            batch_bytes_total,
            batch_elapsed_ms,
            file_index,
            total_files,
        } => {
            progress.downloading = true;
            // Clear status spinners when actual download data arrives
            if progress.current_stage.is_some() {
                progress.current_stage = None;
            }
            progress.download_filename = filename;
            progress.download_bytes = bytes_downloaded;
            progress.download_total = bytes_total;
            progress.download_batch_bytes = batch_bytes_downloaded;
            progress.download_batch_total = batch_bytes_total;
            progress.download_batch_elapsed_ms = batch_elapsed_ms;
            progress.record_download_sample(batch_elapsed_ms, batch_bytes_downloaded);
            progress.download_file_index = file_index;
            if total_files > 0 {
                progress.download_total_files = total_files;
            }
        }
        SseProgressEvent::DownloadDone {
            filename,
            file_index,
            total_files,
            batch_bytes_downloaded,
            batch_bytes_total,
            batch_elapsed_ms,
        } => {
            progress.log.push(ProgressLogEntry {
                message: format!("[{}/{}] {filename}", file_index + 1, total_files),
                style: ProgressStyle::Done,
            });
            if file_index + 1 < total_files {
                // More files to go — keep batch progress visible and show
                // a spinner while hf-hub validates the next file's cache.
                progress.download_filename.clear();
                progress.download_bytes = 0;
                progress.download_total = 0;
                progress.download_batch_bytes = batch_bytes_downloaded;
                progress.download_batch_total = batch_bytes_total;
                progress.download_batch_elapsed_ms = batch_elapsed_ms;
                progress.download_file_index = file_index + 1;
                // Keep total_files and rate/eta for continuity
                progress.current_stage = Some(format!(
                    "Preparing file [{}/{}]...",
                    file_index + 2,
                    total_files
                ));
            } else {
                // Last file done — clear everything (PullComplete follows shortly)
                progress.clear_download();
            }
        }
        SseProgressEvent::PullComplete { model } => {
            progress.clear_download();
            progress.log.push(ProgressLogEntry {
                message: format!("Pull complete: {model}"),
                style: ProgressStyle::Done,
            });
            return true;
        }
        SseProgressEvent::Queued { position } => {
            progress.current_stage = Some(format!("Queued (position {position})"));
        }
    }
    false
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
        let mut state = ProgressState {
            denoise_step: 10,
            denoise_total: 20,
            weight_loaded: 1000,
            download_filename: "test.gguf".to_string(),
            download_bytes: 500,
            download_batch_bytes: 750,
            download_batch_total: 1500,
            download_batch_elapsed_ms: 250,
            download_file_index: 2,
            download_total_files: 5,
            ..Default::default()
        };
        state.log.push(ProgressLogEntry {
            message: "test".to_string(),
            style: ProgressStyle::Done,
        });
        state.clear();
        assert_eq!(state.denoise_step, 0);
        assert_eq!(state.denoise_total, 0);
        assert_eq!(state.weight_loaded, 0);
        assert_eq!(state.download_bytes, 0);
        assert_eq!(state.download_batch_bytes, 0);
        assert_eq!(state.download_batch_total, 0);
        assert_eq!(state.download_batch_elapsed_ms, 0);
        assert!(state.download_filename.is_empty());
        assert_eq!(state.download_file_index, 0);
        assert_eq!(state.download_total_files, 0);
        assert!(state.log.is_empty());
    }

    #[test]
    fn progress_state_download_tracks_file_index() {
        let mut state = ProgressState {
            download_filename: "model.safetensors".to_string(),
            download_bytes: 16_384,
            download_total: 2_900_000_000,
            download_file_index: 1,
            download_total_files: 5,
            ..Default::default()
        };

        assert_eq!(state.download_file_index, 1);
        assert_eq!(state.download_total_files, 5);

        // Simulate DownloadDone resetting download state
        state.download_bytes = 0;
        state.download_total = 0;
        state.download_filename.clear();
        // file_index/total_files stay until next download or clear
        assert_eq!(state.download_file_index, 1);
    }

    #[test]
    fn progress_state_default_has_zero_file_counters() {
        let state = ProgressState::default();
        assert_eq!(state.download_file_index, 0);
        assert_eq!(state.download_total_files, 0);
    }

    #[test]
    fn download_progress_preserves_total_file_count_across_chunk_updates() {
        let mut state = ProgressState::default();

        reduce_progress_state(
            &mut state,
            SseProgressEvent::DownloadProgress {
                filename: "text_encoder_2/model.safetensors".to_string(),
                bytes_downloaded: 0,
                bytes_total: 2_600_000_000,
                batch_bytes_downloaded: 3_000_000_000,
                batch_bytes_total: 8_800_000_000,
                batch_elapsed_ms: 60_000,
                file_index: 2,
                total_files: 6,
            },
        );
        reduce_progress_state(
            &mut state,
            SseProgressEvent::DownloadProgress {
                filename: "text_encoder_2/model.safetensors".to_string(),
                bytes_downloaded: 16_384,
                bytes_total: 2_600_000_000,
                batch_bytes_downloaded: 3_000_016_384,
                batch_bytes_total: 8_800_000_000,
                batch_elapsed_ms: 60_100,
                file_index: 2,
                total_files: 0,
            },
        );

        assert_eq!(state.download_filename, "text_encoder_2/model.safetensors");
        assert_eq!(state.download_bytes, 16_384);
        assert_eq!(state.download_total, 2_600_000_000);
        assert_eq!(state.download_batch_bytes, 3_000_016_384);
        assert_eq!(state.download_batch_total, 8_800_000_000);
        assert_eq!(state.download_batch_elapsed_ms, 60_100);
        assert!(state.download_rate_bps.is_none());
        assert!(state.download_eta_secs.is_none());
        assert_eq!(state.download_file_index, 2);
        assert_eq!(state.download_total_files, 6);
    }

    #[test]
    fn download_rate_and_eta_require_multiple_samples() {
        let mut state = ProgressState::default();

        reduce_progress_state(
            &mut state,
            SseProgressEvent::DownloadProgress {
                filename: "model.safetensors".to_string(),
                bytes_downloaded: 128,
                bytes_total: 1024,
                batch_bytes_downloaded: 128,
                batch_bytes_total: 4096,
                batch_elapsed_ms: 100,
                file_index: 0,
                total_files: 2,
            },
        );
        assert!(state.download_rate_bps.is_none());
        assert!(state.download_eta_secs.is_none());

        reduce_progress_state(
            &mut state,
            SseProgressEvent::DownloadProgress {
                filename: "model.safetensors".to_string(),
                bytes_downloaded: 256,
                bytes_total: 1024,
                batch_bytes_downloaded: 256,
                batch_bytes_total: 4096,
                batch_elapsed_ms: 300,
                file_index: 0,
                total_files: 0,
            },
        );
        assert!(state.download_rate_bps.is_none());
        assert!(state.download_eta_secs.is_none());

        reduce_progress_state(
            &mut state,
            SseProgressEvent::DownloadProgress {
                filename: "model.safetensors".to_string(),
                bytes_downloaded: 1_024,
                bytes_total: 1024,
                batch_bytes_downloaded: 1_536,
                batch_bytes_total: 4096,
                batch_elapsed_ms: 1_300,
                file_index: 0,
                total_files: 0,
            },
        );
        assert!(state.download_rate_bps.is_some());
        assert!(state.download_eta_secs.is_some());
    }

    #[test]
    fn stage_start_clears_stale_download_bar_from_previous_pull() {
        let mut state = ProgressState::default();

        reduce_progress_state(
            &mut state,
            SseProgressEvent::DownloadProgress {
                filename: "vae/model.safetensors".to_string(),
                bytes_downloaded: 512,
                bytes_total: 1024,
                batch_bytes_downloaded: 2048,
                batch_bytes_total: 8192,
                batch_elapsed_ms: 500,
                file_index: 0,
                total_files: 3,
            },
        );
        reduce_progress_state(
            &mut state,
            SseProgressEvent::StageStart {
                name: "Loading model".to_string(),
            },
        );

        assert_eq!(state.current_stage.as_deref(), Some("Loading model"));
        assert!(state.download_filename.is_empty());
        assert_eq!(state.download_bytes, 0);
        assert_eq!(state.download_total, 0);
        assert_eq!(state.download_batch_bytes, 0);
        assert_eq!(state.download_batch_total, 0);
        assert_eq!(state.download_batch_elapsed_ms, 0);
        assert_eq!(state.download_file_index, 0);
        assert_eq!(state.download_total_files, 0);
    }

    #[test]
    fn pull_complete_clears_active_download_bar() {
        let mut state = ProgressState::default();

        reduce_progress_state(
            &mut state,
            SseProgressEvent::DownloadProgress {
                filename: "diffusion_pytorch_model.safetensors".to_string(),
                bytes_downloaded: 2048,
                bytes_total: 4096,
                batch_bytes_downloaded: 2048,
                batch_bytes_total: 4096,
                batch_elapsed_ms: 250,
                file_index: 0,
                total_files: 1,
            },
        );

        let refresh_catalog = reduce_progress_state(
            &mut state,
            SseProgressEvent::PullComplete {
                model: "flux2-klein:q8".to_string(),
            },
        );

        assert!(refresh_catalog);
        assert!(state.download_filename.is_empty());
        assert_eq!(state.download_bytes, 0);
        assert_eq!(state.download_total, 0);
        assert_eq!(state.download_batch_bytes, 0);
        assert_eq!(state.download_batch_total, 0);
        assert_eq!(state.download_batch_elapsed_ms, 0);
        assert_eq!(state.download_total_files, 0);
        assert!(state
            .log
            .iter()
            .any(|entry| entry.message == "Pull complete: flux2-klein:q8"));
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
        let _ = SeedMode::Random.resolve(None);
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
    fn model_actions_are_implemented() {
        // These model actions should exist and NOT be in the unimplemented list
        let implemented = vec![Action::PullModel, Action::RemoveModel, Action::UnloadModel];
        for action in &implemented {
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
                frames: None,
                fps: None,
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
                frames: None,
                fps: None,
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
            frames: None,
            fps: None,
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
            thumb_dimensions: Vec::new(),
            thumb_fixed_cache: Vec::new(),
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
        let entries = [make_test_entry(), make_test_entry()];
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

    // ── Settings view tests ────────────────────────────────

    /// Build a minimal App for settings tests, bypassing server checks.
    /// Config mutations are tested in-memory; save_config() may fail
    /// (save_error is set) but that's fine for mutation tests.
    fn make_settings_test_app() -> App {
        let mut config = Config {
            // Pin default model so the test doesn't depend on downloaded models
            default_model: "flux2-klein:q8".to_string(),
            ..Default::default()
        };
        // Insert a test model so the Model Defaults section appears
        config.models.insert(
            "test-model:q8".to_string(),
            mold_core::config::ModelConfig {
                transformer: Some("/path/to/transformer.gguf".into()),
                vae: Some("/path/to/vae.safetensors".into()),
                default_steps: Some(20),
                default_guidance: Some(3.5),
                default_width: Some(1024),
                default_height: Some(1024),
                lora: Some("/path/to/lora.safetensors".into()),
                lora_scale: Some(0.8),
                negative_prompt: Some("blurry, low quality".into()),
                scheduler: Some(Scheduler::EulerAncestral),
                ..Default::default()
            },
        );

        let picker = ratatui_image::picker::Picker::from_fontsize((8, 16));
        let params = GenerateParams::from_config(&config);
        let family = crate::model_info::family_for_model(&params.model, &config);
        let caps = crate::model_info::capabilities_for_family(&family);
        let visible = ParamField::visible_fields(&caps, params.inference_mode);
        let (bg_tx, bg_rx) = mpsc::unbounded_channel();

        App {
            active_view: View::Settings,
            generate: GenerateState {
                prompt: TextArea::default(),
                negative_prompt: TextArea::default(),
                params,
                focus: GenerateFocus::Navigation,
                param_index: 0,
                visible_fields: visible,
                capabilities: caps,
                progress: ProgressState::default(),
                preview_image: None,
                image_state: None,
                generating: false,
                batch_remaining: 0,
                last_seed: None,
                last_generation_time_ms: None,
                error_message: None,
                model_description: String::new(),
            },
            gallery: GalleryState {
                entries: Vec::new(),
                selected: 0,
                preview_image: None,
                image_state: None,
                scanning: false,
                view_mode: GalleryViewMode::Grid,
                thumbnail_states: Vec::new(),
                thumb_dimensions: Vec::new(),
                thumb_fixed_cache: Vec::new(),
                grid_cols: 3,
                grid_scroll: 0,
            },
            models: ModelsState {
                catalog: Vec::new(),
                selected: 0,
                filter: String::new(),
                filtering: false,
            },
            settings: SettingsState {
                selected_model: Some("test-model:q8".to_string()),
                row_index: 1,
                skip_save: true,
                ..Default::default()
            },
            config,
            server_url: None,
            picker,
            theme: crate::ui::theme::Theme::default(),
            popup: None,
            should_quit: false,
            bg_tx,
            bg_rx,
            tokio_handle: tokio::runtime::Handle::current(),
            resource_info: crate::ui::info::ResourceInfo::default(),
            history: crate::history::PromptHistory::load(),
            layout: LayoutAreas::default(),
            server_process: None,
            upscale_in_progress: false,
            upscale_task: None,
            upscale_tile_progress: None,
            upscale_progress: ProgressState::default(),
            connecting: false,
        }
    }

    /// Helper: find the row_index for a given SettingsKey.
    fn find_settings_row(app: &App, key: SettingsKey) -> usize {
        let rows = app.build_settings_rows();
        rows.iter()
            .position(|r| matches!(r, SettingsRow::Field { key: k, .. } if *k == key))
            .unwrap_or_else(|| panic!("SettingsKey {key:?} not found in rows"))
    }

    #[test]
    fn settings_state_default_values() {
        let state = SettingsState::default();
        assert_eq!(state.row_index, 0);
        assert_eq!(state.scroll_offset, 0);
        assert!(state.selected_model.is_none());
        assert!(state.save_error.is_none());
    }

    #[test]
    fn settings_row_is_field_and_read_only() {
        let header = SettingsRow::SectionHeader {
            name: "General".into(),
        };
        assert!(!header.is_field());

        let field = SettingsRow::Field {
            key: SettingsKey::DefaultModel,
            label: "Model",
            field_type: SettingsFieldType::Text,
        };
        assert!(field.is_field());
        assert!(!field.is_read_only());

        let ro = SettingsRow::Field {
            key: SettingsKey::ModelTransformer,
            label: "Transformer",
            field_type: SettingsFieldType::ReadOnly,
        };
        assert!(ro.is_read_only());
    }

    #[test]
    fn settings_env_override_returns_none_for_unset() {
        assert!(App::settings_env_override(&SettingsKey::ServerPort).is_none());
        assert!(App::settings_env_override(&SettingsKey::DefaultWidth).is_none());
        assert!(App::settings_env_override(&SettingsKey::LogLevel).is_none());
    }

    #[test]
    fn settings_input_popup_variant_exists() {
        let popup = Popup::SettingsInput {
            key: SettingsKey::DefaultModel,
            input: "test".to_string(),
            label: "Model".to_string(),
        };
        match popup {
            Popup::SettingsInput { key, input, label } => {
                assert_eq!(key, SettingsKey::DefaultModel);
                assert_eq!(input, "test");
                assert_eq!(label, "Model");
            }
            _ => panic!("expected SettingsInput"),
        }
    }

    #[test]
    fn view_settings_label_and_index() {
        assert_eq!(View::Settings.label(), "Settings");
        assert_eq!(View::Settings.index(), 3);
        assert_eq!(View::ALL.len(), 4);
        assert_eq!(View::ALL[3], View::Settings);
    }

    // ── Settings E2E: display values ──────────────────────

    #[tokio::test]
    async fn settings_display_all_global_defaults() {
        let app = make_settings_test_app();
        assert_eq!(
            app.settings_display_value(&SettingsKey::DefaultModel),
            "flux2-klein:q8"
        );
        assert_eq!(app.settings_display_value(&SettingsKey::ServerPort), "7680");
        assert_eq!(
            app.settings_display_value(&SettingsKey::DefaultWidth),
            "768"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::DefaultHeight),
            "768"
        );
        assert_eq!(app.settings_display_value(&SettingsKey::DefaultSteps), "4");
        assert_eq!(
            app.settings_display_value(&SettingsKey::EmbedMetadata),
            "on"
        );
        assert_eq!(app.settings_display_value(&SettingsKey::T5Variant), "auto");
        assert_eq!(
            app.settings_display_value(&SettingsKey::Qwen3Variant),
            "auto"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::DefaultNegativePrompt),
            "(none)"
        );
    }

    #[tokio::test]
    async fn settings_display_all_expand_defaults() {
        let app = make_settings_test_app();
        assert_eq!(
            app.settings_display_value(&SettingsKey::ExpandEnabled),
            "off"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::ExpandBackend),
            "local"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::ExpandModel),
            "qwen3-expand:q8"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::ExpandApiModel),
            "qwen2.5:3b"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::ExpandTemperature),
            "0.7"
        );
        assert_eq!(app.settings_display_value(&SettingsKey::ExpandTopP), "0.90");
        assert_eq!(
            app.settings_display_value(&SettingsKey::ExpandMaxTokens),
            "300"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::ExpandThinking),
            "off"
        );
    }

    #[tokio::test]
    async fn settings_display_all_logging_defaults() {
        let app = make_settings_test_app();
        assert_eq!(app.settings_display_value(&SettingsKey::LogLevel), "info");
        assert_eq!(app.settings_display_value(&SettingsKey::LogFile), "off");
        assert_eq!(app.settings_display_value(&SettingsKey::LogMaxDays), "7");
    }

    #[tokio::test]
    async fn settings_display_all_model_defaults() {
        let app = make_settings_test_app();
        assert_eq!(
            app.settings_display_value(&SettingsKey::ModelSelector),
            "test-model:q8"
        );
        assert_eq!(app.settings_display_value(&SettingsKey::ModelSteps), "20");
        assert_eq!(
            app.settings_display_value(&SettingsKey::ModelGuidance),
            "3.5"
        );
        assert_eq!(app.settings_display_value(&SettingsKey::ModelWidth), "1024");
        assert_eq!(
            app.settings_display_value(&SettingsKey::ModelHeight),
            "1024"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::ModelScheduler),
            "euler-ancestral"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::ModelNegativePrompt),
            "blurry, low quality"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::ModelLora),
            "/path/to/lora.safetensors"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::ModelLoraScale),
            "0.8"
        );
        // Read-only paths
        assert_eq!(
            app.settings_display_value(&SettingsKey::ModelTransformer),
            "/path/to/transformer.gguf"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::ModelVae),
            "/path/to/vae.safetensors"
        );
    }

    // ── Settings E2E: numeric adjustments ─────────────────

    #[tokio::test]
    async fn settings_adjust_server_port() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ServerPort, 1.0, 1024.0, 65535.0);
        assert_eq!(app.config.server_port, 7681);
        app.settings_adjust_number(SettingsKey::ServerPort, -2.0, 1024.0, 65535.0);
        assert_eq!(app.config.server_port, 7679);
    }

    #[tokio::test]
    async fn settings_adjust_default_width() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::DefaultWidth, 64.0, 64.0, 4096.0);
        assert_eq!(app.config.default_width, 832);
    }

    #[tokio::test]
    async fn settings_adjust_default_height() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::DefaultHeight, -64.0, 64.0, 4096.0);
        assert_eq!(app.config.default_height, 704);
    }

    #[tokio::test]
    async fn settings_adjust_default_steps() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::DefaultSteps, 1.0, 1.0, 200.0);
        assert_eq!(app.config.default_steps, 5);
    }

    #[tokio::test]
    async fn settings_adjust_expand_temperature() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ExpandTemperature, 0.1, 0.0, 2.0);
        assert!((app.config.expand.temperature - 0.8).abs() < 0.001);
    }

    #[tokio::test]
    async fn settings_adjust_expand_top_p() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ExpandTopP, -0.05, 0.0, 1.0);
        assert!((app.config.expand.top_p - 0.85).abs() < 0.001);
    }

    #[tokio::test]
    async fn settings_adjust_expand_max_tokens() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ExpandMaxTokens, 64.0, 64.0, 4096.0);
        assert_eq!(app.config.expand.max_tokens, 364);
    }

    #[tokio::test]
    async fn settings_adjust_log_max_days() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::LogMaxDays, 1.0, 1.0, 365.0);
        assert_eq!(app.config.logging.max_days, 8);
    }

    #[tokio::test]
    async fn settings_adjust_model_steps() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ModelSteps, 1.0, 1.0, 200.0);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert_eq!(mc.default_steps, Some(21));
    }

    #[tokio::test]
    async fn settings_adjust_model_guidance() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ModelGuidance, 0.5, 0.0, 30.0);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert!((mc.default_guidance.unwrap() - 4.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn settings_adjust_model_width() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ModelWidth, 64.0, 64.0, 4096.0);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert_eq!(mc.default_width, Some(1088));
    }

    #[tokio::test]
    async fn settings_adjust_model_height() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ModelHeight, -64.0, 64.0, 4096.0);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert_eq!(mc.default_height, Some(960));
    }

    #[tokio::test]
    async fn settings_adjust_model_lora_scale() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ModelLoraScale, 0.1, 0.0, 2.0);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert!((mc.lora_scale.unwrap() - 0.9).abs() < 0.001);
    }

    #[tokio::test]
    async fn settings_numeric_clamps_at_min() {
        let mut app = make_settings_test_app();
        // Steps = 4, try decrementing by 100
        app.settings_adjust_number(SettingsKey::DefaultSteps, -100.0, 1.0, 200.0);
        assert_eq!(app.config.default_steps, 1);
    }

    #[tokio::test]
    async fn settings_numeric_clamps_at_max() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::DefaultSteps, 500.0, 1.0, 200.0);
        assert_eq!(app.config.default_steps, 200);
    }

    // ── Settings E2E: boolean toggles ─────────────────────

    #[tokio::test]
    async fn settings_toggle_embed_metadata() {
        let mut app = make_settings_test_app();
        assert!(app.config.embed_metadata);
        app.settings_toggle_bool(SettingsKey::EmbedMetadata);
        assert!(!app.config.embed_metadata);
        app.settings_toggle_bool(SettingsKey::EmbedMetadata);
        assert!(app.config.embed_metadata);
    }

    #[tokio::test]
    async fn settings_toggle_expand_enabled() {
        let mut app = make_settings_test_app();
        assert!(!app.config.expand.enabled);
        app.settings_toggle_bool(SettingsKey::ExpandEnabled);
        assert!(app.config.expand.enabled);
    }

    #[tokio::test]
    async fn settings_toggle_expand_thinking() {
        let mut app = make_settings_test_app();
        assert!(!app.config.expand.thinking);
        app.settings_toggle_bool(SettingsKey::ExpandThinking);
        assert!(app.config.expand.thinking);
    }

    #[tokio::test]
    async fn settings_toggle_log_file() {
        let mut app = make_settings_test_app();
        assert!(!app.config.logging.file);
        app.settings_toggle_bool(SettingsKey::LogFile);
        assert!(app.config.logging.file);
    }

    // ── Settings E2E: toggle cycles ───────────────────────

    #[tokio::test]
    async fn settings_cycle_t5_variant() {
        let mut app = make_settings_test_app();
        let opts = &["auto", "fp16", "q8", "q6", "q5", "q4", "q3"];
        assert_eq!(app.settings_display_value(&SettingsKey::T5Variant), "auto");
        app.settings_cycle_toggle(SettingsKey::T5Variant, opts, 1);
        assert_eq!(app.config.t5_variant, Some("fp16".into()));
        app.settings_cycle_toggle(SettingsKey::T5Variant, opts, 1);
        assert_eq!(app.config.t5_variant, Some("q8".into()));
        // Cycle backward wraps around: q8 (idx 2) - 2 = auto (idx 0)
        app.settings_cycle_toggle(SettingsKey::T5Variant, opts, -2);
        assert!(app.config.t5_variant.is_none()); // "auto" → None
    }

    #[tokio::test]
    async fn settings_cycle_qwen3_variant() {
        let mut app = make_settings_test_app();
        let opts = &["auto", "bf16", "q8", "q6", "iq4", "q3"];
        app.settings_cycle_toggle(SettingsKey::Qwen3Variant, opts, 1);
        assert_eq!(app.config.qwen3_variant, Some("bf16".into()));
        app.settings_cycle_toggle(SettingsKey::Qwen3Variant, opts, -1);
        assert!(app.config.qwen3_variant.is_none());
    }

    #[tokio::test]
    async fn settings_cycle_log_level() {
        let mut app = make_settings_test_app();
        let opts = &["trace", "debug", "info", "warn", "error"];
        assert_eq!(app.config.logging.level, "info");
        app.settings_cycle_toggle(SettingsKey::LogLevel, opts, 1);
        assert_eq!(app.config.logging.level, "warn");
        app.settings_cycle_toggle(SettingsKey::LogLevel, opts, 1);
        assert_eq!(app.config.logging.level, "error");
        app.settings_cycle_toggle(SettingsKey::LogLevel, opts, 1); // wraps
        assert_eq!(app.config.logging.level, "trace");
    }

    #[tokio::test]
    async fn settings_cycle_model_scheduler() {
        let mut app = make_settings_test_app();
        let opts = &["(none)", "ddim", "euler-ancestral", "uni-pc"];
        // Current is euler-ancestral
        assert_eq!(
            app.settings_display_value(&SettingsKey::ModelScheduler),
            "euler-ancestral"
        );
        app.settings_cycle_toggle(SettingsKey::ModelScheduler, opts, 1);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert_eq!(mc.scheduler, Some(Scheduler::UniPc));
        app.settings_cycle_toggle(SettingsKey::ModelScheduler, opts, 1); // wraps to (none)
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert!(mc.scheduler.is_none());
    }

    // ── Settings E2E: text/path apply ─────────────────────

    #[tokio::test]
    async fn settings_apply_default_model() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::DefaultModel, "sd15:fp16".into());
        assert_eq!(app.config.default_model, "sd15:fp16");
    }

    #[tokio::test]
    async fn settings_apply_models_dir() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::ModelsDir, "/tmp/models".into());
        assert_eq!(app.config.models_dir, "/tmp/models");
    }

    #[tokio::test]
    async fn settings_apply_output_dir() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::OutputDir, "/tmp/output".into());
        assert_eq!(app.config.output_dir, Some("/tmp/output".into()));
        // Empty clears
        app.settings_apply_input(SettingsKey::OutputDir, String::new());
        assert!(app.config.output_dir.is_none());
    }

    #[tokio::test]
    async fn settings_apply_default_negative_prompt() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::DefaultNegativePrompt, "ugly, deformed".into());
        assert_eq!(
            app.config.default_negative_prompt,
            Some("ugly, deformed".into())
        );
        app.settings_apply_input(SettingsKey::DefaultNegativePrompt, String::new());
        assert!(app.config.default_negative_prompt.is_none());
    }

    #[tokio::test]
    async fn settings_apply_expand_backend() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::ExpandBackend, "http://localhost:11434".into());
        assert_eq!(app.config.expand.backend, "http://localhost:11434");
    }

    #[tokio::test]
    async fn settings_apply_expand_model() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::ExpandModel, "qwen3-expand:q4".into());
        assert_eq!(app.config.expand.model, "qwen3-expand:q4");
    }

    #[tokio::test]
    async fn settings_apply_expand_api_model() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::ExpandApiModel, "gpt-4o".into());
        assert_eq!(app.config.expand.api_model, "gpt-4o");
    }

    #[tokio::test]
    async fn settings_apply_log_dir() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::LogDir, "/tmp/logs".into());
        assert_eq!(app.config.logging.dir, Some("/tmp/logs".into()));
        app.settings_apply_input(SettingsKey::LogDir, String::new());
        assert!(app.config.logging.dir.is_none());
    }

    #[tokio::test]
    async fn settings_apply_model_negative_prompt() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::ModelNegativePrompt, "watermark".into());
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert_eq!(mc.negative_prompt, Some("watermark".into()));
    }

    #[tokio::test]
    async fn settings_apply_model_lora() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(
            SettingsKey::ModelLora,
            "/new/path/to/lora.safetensors".into(),
        );
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert_eq!(mc.lora, Some("/new/path/to/lora.safetensors".into()));
        // Clear
        app.settings_apply_input(SettingsKey::ModelLora, String::new());
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert!(mc.lora.is_none());
    }

    // ── Settings E2E: model selector cycling ──────────────

    #[tokio::test]
    async fn settings_cycle_model_selector() {
        let mut app = make_settings_test_app();
        // Add a second model
        app.config.models.insert(
            "second-model:fp16".to_string(),
            mold_core::config::ModelConfig::default(),
        );
        assert_eq!(app.settings.selected_model, Some("test-model:q8".into()));
        app.settings_cycle_model(1);
        // Should have moved to the other model (HashMap order is not guaranteed,
        // but it should be a different model)
        assert!(app.settings.selected_model.is_some());
        let selected = app.settings.selected_model.clone().unwrap();
        app.settings_cycle_model(1); // cycle back
        assert_ne!(
            app.settings.selected_model.as_deref(),
            Some(selected.as_str())
        );
    }

    // ── Settings E2E: navigation ──────────────────────────

    #[tokio::test]
    async fn settings_navigate_skips_headers() {
        let mut app = make_settings_test_app();
        // Start at index 0, which is a section header
        app.settings.row_index = 0;
        app.settings_navigate(1); // should skip header, land on first field
        let rows = app.build_settings_rows();
        assert!(rows[app.settings.row_index].is_field());
    }

    #[tokio::test]
    async fn settings_navigate_clamps_at_boundaries() {
        let mut app = make_settings_test_app();
        app.settings.row_index = 0;
        app.settings_navigate(-1); // can't go above 0
        assert_eq!(app.settings.row_index, 0);
    }

    // ── Settings E2E: build_settings_rows structure ───────

    #[tokio::test]
    async fn settings_rows_have_all_sections() {
        let app = make_settings_test_app();
        let rows = app.build_settings_rows();
        let headers: Vec<String> = rows
            .iter()
            .filter_map(|r| match r {
                SettingsRow::SectionHeader { name } => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert!(headers.iter().any(|h| h == "General"));
        assert!(headers.iter().any(|h| h == "Expand"));
        assert!(headers.iter().any(|h| h == "Logging"));
        assert!(headers.iter().any(|h| h.starts_with("Model Defaults")));
    }

    #[tokio::test]
    async fn settings_rows_contain_read_only_paths() {
        let app = make_settings_test_app();
        let rows = app.build_settings_rows();
        let has_ro = rows.iter().any(|r| {
            matches!(
                r,
                SettingsRow::Field {
                    key: SettingsKey::ModelTransformer,
                    field_type: SettingsFieldType::ReadOnly,
                    ..
                }
            )
        });
        assert!(has_ro, "ModelTransformer should be ReadOnly");
    }

    // ── Settings E2E: full increment via row_index ────────

    #[tokio::test]
    async fn settings_increment_via_row_index_adjusts_width() {
        let mut app = make_settings_test_app();
        let idx = find_settings_row(&app, SettingsKey::DefaultWidth);
        app.settings.row_index = idx;
        app.settings_increment(1);
        assert_eq!(app.config.default_width, 832); // 768 + 64
    }

    #[tokio::test]
    async fn settings_increment_via_row_index_toggles_bool() {
        let mut app = make_settings_test_app();
        let idx = find_settings_row(&app, SettingsKey::EmbedMetadata);
        app.settings.row_index = idx;
        assert!(app.config.embed_metadata);
        app.settings_increment(1);
        assert!(!app.config.embed_metadata);
    }

    #[tokio::test]
    async fn settings_increment_via_row_index_cycles_toggle() {
        let mut app = make_settings_test_app();
        let idx = find_settings_row(&app, SettingsKey::LogLevel);
        app.settings.row_index = idx;
        assert_eq!(app.config.logging.level, "info");
        app.settings_increment(1);
        assert_eq!(app.config.logging.level, "warn");
    }

    // ── Settings E2E: confirm opens popup for text fields ─

    #[tokio::test]
    async fn settings_confirm_opens_popup_for_text_field() {
        let mut app = make_settings_test_app();
        let idx = find_settings_row(&app, SettingsKey::DefaultModel);
        app.settings.row_index = idx;
        app.settings_confirm();
        assert!(matches!(app.popup, Some(Popup::SettingsInput { .. })));
        if let Some(Popup::SettingsInput { key, input, .. }) = &app.popup {
            assert_eq!(*key, SettingsKey::DefaultModel);
            assert_eq!(input, "flux2-klein:q8");
        }
    }

    #[tokio::test]
    async fn settings_confirm_toggles_bool_field() {
        let mut app = make_settings_test_app();
        let idx = find_settings_row(&app, SettingsKey::ExpandEnabled);
        app.settings.row_index = idx;
        assert!(!app.config.expand.enabled);
        app.settings_confirm();
        assert!(app.config.expand.enabled);
        assert!(app.popup.is_none()); // no popup for bools
    }

    #[tokio::test]
    async fn settings_confirm_cycles_toggle_field() {
        let mut app = make_settings_test_app();
        let idx = find_settings_row(&app, SettingsKey::T5Variant);
        app.settings.row_index = idx;
        app.settings_confirm();
        assert_eq!(app.config.t5_variant, Some("fp16".into()));
        assert!(app.popup.is_none());
    }

    // ── Regression: metadata uses response model, not UI state (#161) ────

    #[tokio::test]
    async fn generation_complete_metadata_uses_response_model() {
        // Simulates: user starts generation with model A, then switches UI
        // to model B before generation completes. Metadata must record
        // model A (from the response), not model B (current UI state).
        let mut app = make_settings_test_app();
        app.active_view = View::Generate;
        app.generate.generating = true;
        app.generate.batch_remaining = 1;

        // UI currently shows model B (user switched mid-generation)
        app.generate.params.model = "flux-dev:q4".to_string();
        // Set a non-empty prompt so the history entry is recorded
        app.generate.prompt = TextArea::from(["a test prompt"]);

        // Inject a GenerationComplete with model A (the model that actually ran)
        let response = GenerateResponse {
            images: vec![mold_core::ImageData {
                data: vec![0u8; 4],
                format: OutputFormat::Png,
                width: 64,
                height: 64,
                index: 0,
            }],
            generation_time_ms: 100,
            model: "flux-schnell:q8".to_string(),
            seed_used: 42,
            video: None,
        };
        app.bg_tx
            .send(BackgroundEvent::GenerationComplete(Box::new(response)))
            .unwrap();

        // Process the event through the real handler
        app.process_background_events();

        // History entry must record the *response* model, not the UI model
        assert!(!app.history.is_empty());
        let results = app.history.search("a test prompt");
        assert!(!results.is_empty(), "history should contain our prompt");
        assert_eq!(
            results[0].model, "flux-schnell:q8",
            "history should record response model, not UI model"
        );

        // Gallery metadata (if an entry was created) should also use response model.
        // The test image bytes aren't a valid PNG so the gallery entry may not be
        // created (image::load_from_memory fails), but the history entry is the
        // authoritative check for this regression.
        if let Some(entry) = app.gallery.entries.first() {
            assert_eq!(
                entry.metadata.model, "flux-schnell:q8",
                "gallery metadata should record response model, not UI model"
            );
        }
    }

    // ── Regression: batch_remaining tracks multi-image generation (#162) ────

    #[test]
    fn batch_remaining_decrements_on_generation_complete() {
        // Verify batch tracking: generating stays true until all batch
        // images are received.
        let mut gen = GenerateState {
            prompt: TextArea::default(),
            negative_prompt: TextArea::default(),
            params: GenerateParams::from_config(&Config::load_or_default()),
            focus: GenerateFocus::Prompt,
            param_index: 0,
            visible_fields: vec![],
            capabilities: capabilities_for_family("flux"),
            progress: ProgressState::default(),
            preview_image: None,
            image_state: None,
            generating: true,
            batch_remaining: 3,
            last_seed: None,
            last_generation_time_ms: None,
            error_message: None,
            model_description: String::new(),
        };

        // Simulate receiving first image — still 2 more to go
        gen.batch_remaining = gen.batch_remaining.saturating_sub(1);
        assert_eq!(gen.batch_remaining, 2);
        if gen.batch_remaining == 0 {
            gen.generating = false;
        }
        assert!(
            gen.generating,
            "should still be generating with 2 images left"
        );

        // Second image
        gen.batch_remaining = gen.batch_remaining.saturating_sub(1);
        assert_eq!(gen.batch_remaining, 1);
        if gen.batch_remaining == 0 {
            gen.generating = false;
        }
        assert!(
            gen.generating,
            "should still be generating with 1 image left"
        );

        // Third (final) image
        gen.batch_remaining = gen.batch_remaining.saturating_sub(1);
        assert_eq!(gen.batch_remaining, 0);
        if gen.batch_remaining == 0 {
            gen.generating = false;
        }
        assert!(
            !gen.generating,
            "should stop generating when batch is complete"
        );
    }

    #[test]
    fn batch_remaining_resets_on_error() {
        let mut gen = GenerateState {
            prompt: TextArea::default(),
            negative_prompt: TextArea::default(),
            params: GenerateParams::from_config(&Config::load_or_default()),
            focus: GenerateFocus::Prompt,
            param_index: 0,
            visible_fields: vec![],
            capabilities: capabilities_for_family("flux"),
            progress: ProgressState::default(),
            preview_image: None,
            image_state: None,
            generating: true,
            batch_remaining: 4,
            last_seed: None,
            last_generation_time_ms: None,
            error_message: None,
            model_description: String::new(),
        };

        // Simulate error mid-batch
        gen.generating = false;
        gen.batch_remaining = 0;
        gen.error_message = Some("connection lost".to_string());

        assert!(!gen.generating);
        assert_eq!(gen.batch_remaining, 0);
        assert!(gen.error_message.is_some());
    }

    #[test]
    fn start_generation_sets_batch_remaining() {
        let config = Config::load_or_default();
        let params = GenerateParams::from_config(&config);
        // batch defaults to 1
        assert_eq!(params.batch, 1);

        let mut gen = GenerateState {
            prompt: TextArea::default(),
            negative_prompt: TextArea::default(),
            params,
            focus: GenerateFocus::Prompt,
            param_index: 0,
            visible_fields: vec![],
            capabilities: capabilities_for_family("flux"),
            progress: ProgressState::default(),
            preview_image: None,
            image_state: None,
            generating: false,
            batch_remaining: 0,
            last_seed: None,
            last_generation_time_ms: None,
            error_message: None,
            model_description: String::new(),
        };

        // Simulate setting batch to 4 and starting generation
        gen.params.batch = 4;
        gen.generating = true;
        gen.batch_remaining = gen.params.batch;
        assert_eq!(gen.batch_remaining, 4);
        assert!(gen.generating);
    }

    // ── Regression: batch size unlimited in TUI (#194) ─────────────────

    #[tokio::test]
    async fn batch_increment_no_upper_cap() {
        let mut app = make_settings_test_app();
        // Switch to Generate view with Parameters focus
        app.active_view = View::Generate;
        app.generate.focus = GenerateFocus::Parameters;
        // Point param_index at the Batch field
        let batch_idx = app
            .generate
            .visible_fields
            .iter()
            .position(|f| *f == ParamField::Batch)
            .expect("Batch field should be in visible_fields");
        app.generate.param_index = batch_idx;

        // Set batch to 16 and increment — should exceed old cap of 16
        app.generate.params.batch = 16;
        app.increment_param(1);
        assert_eq!(
            app.generate.params.batch, 17,
            "batch should exceed old cap of 16"
        );

        // Set to a large value and increment further
        app.generate.params.batch = 100;
        app.increment_param(1);
        assert_eq!(
            app.generate.params.batch, 101,
            "batch should have no upper bound"
        );

        // Minimum should still be 1
        app.generate.params.batch = 1;
        app.increment_param(-1);
        assert_eq!(app.generate.params.batch, 1, "batch should not go below 1");
    }

    #[test]
    fn available_upscaler_models_returns_all_known() {
        // All known upscaler models should be listed (downloaded or not)
        let models: Vec<String> = mold_core::manifest::known_manifests()
            .iter()
            .filter(|m| m.is_upscaler())
            .map(|m| m.name.clone())
            .collect();
        // There should be 7 upscaler models in the manifest
        assert_eq!(models.len(), 7);
        assert!(models.iter().all(|n| !n.is_empty()));
    }

    #[test]
    fn upscale_model_selector_popup_variant() {
        let popup = Popup::UpscaleModelSelector {
            filter: String::new(),
            selected: 0,
            filtered: vec![
                "real-esrgan-x4plus:fp16".into(),
                "real-esrgan-x2:fp16".into(),
            ],
        };
        // Verify the variant can be pattern-matched and fields accessed
        if let Popup::UpscaleModelSelector {
            filter,
            selected,
            filtered,
        } = &popup
        {
            assert!(filter.is_empty());
            assert_eq!(*selected, 0);
            assert_eq!(filtered.len(), 2);
        } else {
            panic!("expected UpscaleModelSelector");
        }
    }

    #[test]
    fn upscale_background_event_variants() {
        // Verify the new BackgroundEvent variants can be constructed
        let progress = BackgroundEvent::UpscaleProgress { tile: 3, total: 9 };
        if let BackgroundEvent::UpscaleProgress { tile, total } = progress {
            assert_eq!(tile, 3);
            assert_eq!(total, 9);
        } else {
            panic!("expected UpscaleProgress");
        }

        let complete = BackgroundEvent::UpscaleComplete {
            image_data: vec![0u8; 100],
            source_path: std::path::PathBuf::from("/tmp/test.png"),
            model: "real-esrgan-x4plus:fp16".into(),
            scale_factor: 4,
            original_width: 512,
            original_height: 512,
            upscale_time_ms: 1500,
        };
        if let BackgroundEvent::UpscaleComplete {
            scale_factor,
            original_width,
            original_height,
            ..
        } = complete
        {
            assert_eq!(scale_factor, 4);
            assert_eq!(original_width, 512);
            assert_eq!(original_height, 512);
        } else {
            panic!("expected UpscaleComplete");
        }

        let failed = BackgroundEvent::UpscaleFailed("OOM".into());
        if let BackgroundEvent::UpscaleFailed(msg) = failed {
            assert_eq!(msg, "OOM");
        } else {
            panic!("expected UpscaleFailed");
        }
    }

    #[test]
    fn upscale_model_filter_narrows_list() {
        let all = vec![
            "real-esrgan-x4plus:fp16".to_string(),
            "real-esrgan-x2:fp16".to_string(),
            "realesrgan-anime:fp16".to_string(),
        ];
        let query = "x4".to_lowercase();
        let filtered: Vec<String> = all
            .into_iter()
            .filter(|name| name.to_lowercase().contains(&query))
            .collect();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], "real-esrgan-x4plus:fp16");
    }

    #[test]
    fn upscale_model_filter_empty_returns_all() {
        let all = vec![
            "real-esrgan-x4plus:fp16".to_string(),
            "real-esrgan-x2:fp16".to_string(),
        ];
        let query = "".to_lowercase();
        let filtered: Vec<String> = all
            .into_iter()
            .filter(|name| name.to_lowercase().contains(&query))
            .collect();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn model_list_sorts_downloaded_first() {
        // Simulate the sorting logic used by open_model_selector / available_upscaler_models
        let config = Config::default();
        let mut models = [
            "not-downloaded-model:q8".to_string(),
            "also-not-downloaded:fp16".to_string(),
        ];
        // With empty config, none are "downloaded" — order should be preserved
        models.sort_by_key(|name| {
            let resolved = mold_core::manifest::resolve_model_name(name);
            let downloaded =
                config.models.contains_key(&resolved) || config.manifest_model_is_downloaded(name);
            if downloaded {
                0
            } else {
                1
            }
        });
        assert_eq!(models[0], "not-downloaded-model:q8");
        assert_eq!(models[1], "also-not-downloaded:fp16");
    }

    #[test]
    fn model_list_downloaded_sorts_before_undownloaded() {
        let mut config = Config::default();
        // Mark one model as "downloaded" by adding it to config.models
        config.models.insert(
            "second-model:fp16".to_string(),
            mold_core::config::ModelConfig {
                transformer: Some("/fake/path.safetensors".into()),
                ..Default::default()
            },
        );

        let mut models = [
            "first-model:q8".to_string(),
            "second-model:fp16".to_string(),
            "third-model:q4".to_string(),
        ];
        models.sort_by_key(|name| {
            let resolved = mold_core::manifest::resolve_model_name(name);
            let downloaded =
                config.models.contains_key(&resolved) || config.manifest_model_is_downloaded(name);
            if downloaded {
                0
            } else {
                1
            }
        });
        // "second-model:fp16" is downloaded, so it should be first
        assert_eq!(models[0], "second-model:fp16");
        // The other two remain in their original relative order
        assert_eq!(models[1], "first-model:q8");
        assert_eq!(models[2], "third-model:q4");
    }

    #[test]
    fn default_model_resolution_for_selector() {
        let config = Config::default();
        let default = mold_core::manifest::resolve_model_name(&config.resolved_default_model());
        // Default should resolve to a known model name
        assert!(!default.is_empty());
    }

    #[test]
    fn upscale_download_progress_event_variant() {
        let event = BackgroundEvent::UpscaleDownloadProgress(SseProgressEvent::DownloadProgress {
            filename: "weights.safetensors".into(),
            file_index: 0,
            total_files: 1,
            bytes_downloaded: 50_000_000,
            bytes_total: 100_000_000,
            batch_bytes_downloaded: 50_000_000,
            batch_bytes_total: 100_000_000,
            batch_elapsed_ms: 5_000,
        });
        if let BackgroundEvent::UpscaleDownloadProgress(SseProgressEvent::DownloadProgress {
            filename,
            bytes_downloaded,
            bytes_total,
            ..
        }) = event
        {
            assert_eq!(filename, "weights.safetensors");
            assert_eq!(bytes_downloaded, 50_000_000);
            assert_eq!(bytes_total, 100_000_000);
        } else {
            panic!("expected UpscaleDownloadProgress(DownloadProgress)");
        }
    }

    #[test]
    fn upscale_progress_state_tracks_download() {
        let mut progress = ProgressState::default();
        assert!(!progress.is_downloading());

        // Simulate download progress events via reduce_progress_state
        reduce_progress_state(
            &mut progress,
            SseProgressEvent::Info {
                message: "Model 'real-esrgan-x4plus:fp16' not found locally, pulling...".into(),
            },
        );
        assert!(progress.is_downloading());

        reduce_progress_state(
            &mut progress,
            SseProgressEvent::DownloadProgress {
                filename: "RealESRGAN_x4plus.pth".into(),
                file_index: 0,
                total_files: 1,
                bytes_downloaded: 30_000_000,
                bytes_total: 67_000_000,
                batch_bytes_downloaded: 30_000_000,
                batch_bytes_total: 67_000_000,
                batch_elapsed_ms: 3_000,
            },
        );
        assert!(progress.is_downloading());
        assert_eq!(progress.download_batch_bytes, 30_000_000);
        assert_eq!(progress.download_batch_total, 67_000_000);
        assert_eq!(progress.download_filename, "RealESRGAN_x4plus.pth");
        assert_eq!(progress.download_total_files, 1);
    }

    #[test]
    fn upscale_progress_transitions_download_to_tiles() {
        let mut progress = ProgressState::default();

        // Download phase
        reduce_progress_state(
            &mut progress,
            SseProgressEvent::DownloadProgress {
                filename: "RealESRGAN_x4plus.pth".into(),
                file_index: 0,
                total_files: 1,
                bytes_downloaded: 67_000_000,
                bytes_total: 67_000_000,
                batch_bytes_downloaded: 67_000_000,
                batch_bytes_total: 67_000_000,
                batch_elapsed_ms: 6_000,
            },
        );
        assert!(progress.is_downloading());

        // Download done
        reduce_progress_state(
            &mut progress,
            SseProgressEvent::DownloadDone {
                filename: "RealESRGAN_x4plus.pth".into(),
                file_index: 0,
                total_files: 1,
                batch_bytes_downloaded: 67_000_000,
                batch_bytes_total: 67_000_000,
                batch_elapsed_ms: 6_000,
            },
        );

        // Pull complete clears download state
        reduce_progress_state(
            &mut progress,
            SseProgressEvent::PullComplete {
                model: "real-esrgan-x4plus:fp16".into(),
            },
        );
        assert!(!progress.is_downloading());
        assert_eq!(progress.download_batch_bytes, 0);

        // Now tile progress would come via separate UpscaleProgress events,
        // not through this progress state. Verify the state is clean for
        // the next phase.
        assert_eq!(progress.denoise_step, 0);
    }

    #[test]
    fn upscale_progress_cleared_on_completion() {
        let mut progress = ProgressState::default();

        // Simulate some download state
        reduce_progress_state(
            &mut progress,
            SseProgressEvent::DownloadProgress {
                filename: "model.pth".into(),
                file_index: 0,
                total_files: 1,
                bytes_downloaded: 10_000,
                bytes_total: 20_000,
                batch_bytes_downloaded: 10_000,
                batch_bytes_total: 20_000,
                batch_elapsed_ms: 1_000,
            },
        );
        assert!(progress.is_downloading());

        // clear() should reset everything
        progress.clear();
        assert!(!progress.is_downloading());
        assert_eq!(progress.download_batch_bytes, 0);
        assert_eq!(progress.download_batch_total, 0);
        assert!(progress.download_filename.is_empty());
    }

    #[test]
    fn model_selector_excludes_upscalers() {
        // The generation model selector should never include upscaler models.
        let catalog = mold_core::build_model_catalog(&Config::default(), None, false);
        let generation_models: Vec<String> = catalog
            .iter()
            .filter(|m| m.is_generation_model())
            .map(|m| m.name.clone())
            .collect();

        for name in &generation_models {
            assert!(
                !name.starts_with("real-esrgan"),
                "model selector should not include upscaler model '{name}'"
            );
        }
        // Verify we actually have generation models
        assert!(
            !generation_models.is_empty(),
            "should have generation models after filtering"
        );
    }

    #[test]
    fn model_selector_excludes_utility_models() {
        // The generation model selector should never include utility models like qwen3-expand.
        let catalog = mold_core::build_model_catalog(&Config::default(), None, false);
        let generation_models: Vec<String> = catalog
            .iter()
            .filter(|m| m.is_generation_model())
            .map(|m| m.name.clone())
            .collect();

        for name in &generation_models {
            assert!(
                !name.starts_with("qwen3-expand"),
                "model selector should not include utility model '{name}'"
            );
        }
    }

    #[test]
    fn full_catalog_still_includes_upscalers_and_utility() {
        // The full catalog (for Models tab / mold list) should still include everything.
        let catalog = mold_core::build_model_catalog(&Config::default(), None, false);
        assert!(
            catalog.iter().any(|m| m.is_upscaler()),
            "full catalog should include upscaler models"
        );
    }

    // ── Remote server awareness tests ─────────────────────────────

    fn make_test_catalog_entry(
        name: &str,
        steps: u32,
        guidance: f64,
        width: u32,
        height: u32,
        desc: &str,
    ) -> ModelInfoExtended {
        ModelInfoExtended {
            info: mold_core::ModelInfo {
                name: name.to_string(),
                family: "flux".to_string(),
                size_gb: 4.5,
                is_loaded: false,
                last_used: None,
                hf_repo: "test/repo".to_string(),
            },
            defaults: mold_core::ModelDefaults {
                default_steps: steps,
                default_guidance: guidance,
                default_width: width,
                default_height: height,
                description: desc.to_string(),
            },
            downloaded: true,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
        }
    }

    #[test]
    fn remote_catalog_defaults_applied_to_matching_model() {
        // Simulates apply_remote_model_defaults logic
        let mut params = GenerateParams::from_config(&Config::load_or_default());
        params.model = "flux-dev:q4".to_string();
        params.steps = 1;

        let catalog = vec![make_test_catalog_entry(
            "flux-dev:q4",
            20,
            3.5,
            1024,
            1024,
            "FLUX Dev Q4 GGUF",
        )];

        // Apply defaults from catalog (same logic as apply_remote_model_defaults)
        if let Some(entry) = catalog.iter().find(|m| m.name == params.model) {
            params.steps = entry.defaults.default_steps;
            params.guidance = entry.defaults.default_guidance;
            params.width = entry.defaults.default_width;
            params.height = entry.defaults.default_height;
        }

        assert_eq!(params.steps, 20);
        assert!((params.guidance - 3.5).abs() < f64::EPSILON);
        assert_eq!(params.width, 1024);
        assert_eq!(params.height, 1024);
    }

    #[test]
    fn remote_catalog_defaults_no_match_is_noop() {
        let mut params = GenerateParams::from_config(&Config::load_or_default());
        let original_steps = params.steps;
        params.model = "nonexistent-model".to_string();

        let catalog = vec![make_test_catalog_entry(
            "flux-dev:q4",
            99,
            9.9,
            512,
            512,
            "test",
        )];

        if let Some(entry) = catalog.iter().find(|m| m.name == params.model) {
            params.steps = entry.defaults.default_steps;
        }

        assert_eq!(
            params.steps, original_steps,
            "should not change for non-matching model"
        );
    }

    #[test]
    fn server_status_update_populates_resource_info() {
        let mut ri = crate::ui::info::ResourceInfo::default();
        let status = mold_core::ServerStatus {
            version: "0.6.3".to_string(),
            git_sha: None,
            build_date: None,
            models_loaded: vec!["flux-dev:q4".to_string()],
            busy: true,
            current_generation: None,
            gpu_info: Some(mold_core::GpuInfo {
                name: "RTX 4090".to_string(),
                vram_total_mb: 24564,
                vram_used_mb: 8192,
            }),
            uptime_secs: 3600,
            hostname: Some("hal9000".to_string()),
            memory_status: Some("VRAM: 16.0 GB free".to_string()),
        };
        ri.update_from_server_status(status);
        assert_eq!(ri.memory_line.as_deref(), Some("VRAM: 16.0 GB free"));
        assert_eq!(ri.process_memory_mb, 0);
        let ss = ri.server_status.as_ref().unwrap();
        assert_eq!(ss.hostname.as_deref(), Some("hal9000"));
        assert!(ss.busy);
        assert_eq!(ss.gpu_info.as_ref().unwrap().name, "RTX 4090");
    }

    #[test]
    fn clear_server_status_reverts_to_local() {
        let mut ri = crate::ui::info::ResourceInfo::default();
        ri.server_status = Some(mold_core::ServerStatus {
            version: "0.6.3".to_string(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: false,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 0,
            hostname: Some("remote".to_string()),
            memory_status: Some("VRAM: 16.0 GB free".to_string()),
        });
        ri.clear_server_status();
        assert!(ri.server_status.is_none());
        ri.refresh_local();
        // After refresh_local, process_memory_mb is populated (may be 0 if no mold process)
        // The point is it doesn't panic and switches to local info
    }

    #[test]
    fn background_event_server_status_variant_exists() {
        // Compile-time check that the variant exists
        let status = mold_core::ServerStatus {
            version: "0.6.3".to_string(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: false,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 0,
            hostname: None,
            memory_status: None,
        };
        let _event = BackgroundEvent::ServerStatusUpdate(Some(Box::new(status)));
        // None variant for server-unreachable
        let _event_none = BackgroundEvent::ServerStatusUpdate(None);
    }
}
