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
    /// Generation completed successfully. `from_local` is true for any
    /// response produced by the in-process inference engine — including
    /// Auto-mode fallbacks after the remote server goes unreachable —
    /// so the completion handler can still write the file locally even
    /// when `server_url` remains set.
    GenerationComplete {
        response: Box<GenerateResponse>,
        from_local: bool,
    },
    /// Generation or background task failed.
    Error(String),
    /// Merged all-hosts gallery scan completed.
    GalleryScanComplete(crate::gallery_scan::MergedScan),
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
    /// Server catalog refreshed (e.g., after a pull). Updates the model list
    /// without the mode-switching side effects of `ServerConnected`.
    CatalogRefreshed(Vec<ModelInfoExtended>),
    /// A server-side gallery delete failed. Carries the server's error
    /// message so the UI can surface it and re-sync the local list with
    /// whatever state remains on the server.
    GalleryDeleteFailed(String),
    /// Chain progress update from server SSE.
    ChainProgress(mold_core::ChainProgressEvent),
    /// Chain generation completed — video bytes ready.
    ChainComplete {
        response: Box<mold_core::ChainResponse>,
    },
    /// Chain generation failed.
    ChainError(String),
    /// Per-host `/api/status` poll result for the Machines workspace.
    /// `None` marks the row Offline — it stays listed and self-heals.
    HostStatusUpdate {
        host_id: String,
        status: Option<Box<ServerStatus>>,
    },
    /// Per-host authoritative device inventory for Machines controls.
    HostDevicesUpdate {
        host_id: String,
        devices: Option<mold_core::DeviceState>,
    },
    /// Accepted device mutation response. Applied immediately before the
    /// follow-up inventory poll so restart-only success is never discarded.
    HostDeviceMutationApplied {
        host_id: String,
        device: Box<mold_core::DeviceInfo>,
    },
    HostCapabilitiesUpdate {
        host_id: String,
        capabilities: Option<mold_core::ServerCapabilities>,
    },
    /// Per-host queue snapshot for the selected Machines row.
    HostQueueUpdate {
        host_id: String,
        queue: Option<mold_core::QueueListingWire>,
    },
    /// Result of the connect-a-machine test fetch.
    MachineConnectTested {
        url: String,
        api_key: Option<String>,
        result: Result<Box<ServerStatus>, String>,
    },
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

/// Cap the Timeline log to avoid unbounded growth during long runs (e.g.
/// multi-hour video generations push hundreds of weight-load / download /
/// denoise entries). The Timeline panel only ever shows the tail, so older
/// rows are invisible — dropping them is a pure memory win.
pub(crate) const MAX_LOG_ENTRIES: usize = 500;

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
    /// Wall-clock start of the current generation. Set by
    /// [`ProgressState::mark_generation_start`] when the user triggers a
    /// run and cleared when generation finishes. Drives the always-visible
    /// "Overall" row in the Timeline panel.
    pub generation_started_at: Option<std::time::Instant>,
    /// Wall-clock start of the currently-active pipeline stage.
    /// Set on each `StageStart` event, cleared on the matching `StageDone`
    /// (or when generation finishes). Drives the per-stage elapsed suffix
    /// appended to the spinner row.
    pub stage_started_at: Option<std::time::Instant>,
    /// One-based index of the currently-active pipeline stage — useful when
    /// we want to show "step N" without knowing the total up front.
    pub stage_index: usize,
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
        self.generation_started_at = None;
        self.stage_started_at = None;
        self.stage_index = 0;
    }

    /// Mark the start of a new generation — called from `start_generation`
    /// right after the rest of the progress state is cleared. Stamping the
    /// start here keeps the "Overall" Timeline row accurate even before any
    /// SSE events arrive from the server.
    pub fn mark_generation_start(&mut self) {
        self.generation_started_at = Some(std::time::Instant::now());
        self.stage_started_at = None;
        self.stage_index = 0;
    }

    /// Wall-clock duration since [`mark_generation_start`], or `None` if a
    /// generation isn't in flight.
    pub fn generation_elapsed(&self) -> Option<std::time::Duration> {
        self.generation_started_at.map(|t| t.elapsed())
    }

    /// Wall-clock duration since the active stage began.
    pub fn stage_elapsed(&self) -> Option<std::time::Duration> {
        self.stage_started_at.map(|t| t.elapsed())
    }

    /// Append a log entry, trimming the oldest rows when the buffer would
    /// exceed [`MAX_LOG_ENTRIES`]. Use this in place of `progress.log.push(…)`
    /// at every event-driven append site so the buffer stays bounded.
    pub fn push_log(&mut self, entry: ProgressLogEntry) {
        self.log.push(entry);
        if self.log.len() > MAX_LOG_ENTRIES {
            let overflow = self.log.len() - MAX_LOG_ENTRIES;
            self.log.drain(..overflow);
        }
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

/// Which sub-mode the Create view is in.
///
/// The chain composer is nested under Create (mirroring the desktop's
/// `/create/chain` route) rather than being a tab of its own. Switching
/// workspaces does not reset the mode — a chain in progress survives a
/// round-trip through Library and back; only [`Action::ChainExit`]
/// returns to Compose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CreateMode {
    #[default]
    Compose,
    Chain,
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
///
/// The Create redesign trimmed this to the essentials + the Advanced
/// accordion's section fields (see `ui::create_form::visible_rows`):
/// `Width`/`Height` merged into `Size`, `SeedValue` merged into `Seed`,
/// and `Mode`/`Host` were deleted — routing comes from the Machines
/// generation target (`GenTarget`). `UnloadModel` moved to Models (`u`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamField {
    // Essentials
    Model,
    Size,
    Steps,
    Guidance,
    Seed,
    Batch,
    // Advanced — Sampling
    Scheduler,
    Expand,
    Offload,
    // Advanced — Source
    SourceImage,
    Strength,
    MaskImage,
    ControlImage,
    ControlModel,
    ControlScale,
    // Advanced — LoRA / Upscale / Output
    Lora,
    Upscale,
    Format,
    // Advanced — Video
    Frames,
    Fps,
}

impl ParamField {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Model => "Model",
            Self::Size => "Size",
            Self::Steps => "Detail",
            Self::Guidance => "Prompt strength",
            Self::Seed => "Seed",
            Self::Batch => "Batch",
            Self::Format => "Format",
            Self::Scheduler => "Scheduler",
            Self::Lora => "LoRA",
            Self::Upscale => "Upscale",
            Self::Expand => "Expand prompt",
            Self::Offload => "Offload",
            Self::SourceImage => "Source",
            Self::Strength => "Strength",
            Self::MaskImage => "Mask",
            Self::ControlImage => "Control",
            Self::ControlModel => "CNet Mdl",
            Self::Frames => "Frames",
            Self::Fps => "FPS",
            Self::ControlScale => "Scale",
        }
    }
}

fn qwen_image_edit_dimensions_for_path(path: &str) -> Option<(u32, u32)> {
    const TARGET_AREA: u32 = 1024 * 1024;
    const ALIGN: u32 = 16;

    let bytes = std::fs::read(path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    let orig_w = img.width().max(1);
    let orig_h = img.height().max(1);
    Some(mold_core::fit_to_target_area(
        orig_w,
        orig_h,
        TARGET_AREA,
        ALIGN,
    ))
}

/// Parse the Size popup's free-text `WxH` entry (`1024x768`, `1024 × 768`,
/// …). Both numbers clamp into the same 256–4096 range the old Width/Height
/// rows enforced. Returns `None` when the text isn't two numbers.
pub(crate) fn parse_size_input(text: &str) -> Option<(u32, u32)> {
    let mut parts = text.split(['x', 'X', '\u{00d7}']).map(str::trim);
    let w: u32 = parts.next()?.parse().ok()?;
    let h: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((w.clamp(256, 4096), h.clamp(256, 4096)))
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
    /// Display name of the Machines generation target when this run was
    /// routed at a specific registered host — used only to compose the
    /// no-fallback "Can't reach {name} ({url})" error.
    pub target_host_name: Option<String>,
    // Advanced
    pub lora_path: Option<String>,
    pub lora_scale: f64,
    pub expand: bool,
    pub offload: bool,
    /// Upscaler to run after generation (wired to the existing
    /// `GenerateRequest.upscale_model` field). `None` = off.
    pub upscale_model: Option<String>,
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
            target_host_name: None,
            lora_path: None,
            lora_scale: 1.0,
            expand: false,
            offload: false,
            upscale_model: None,
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
            ParamField::Size => format!("{} \u{00d7} {}", self.width, self.height),
            ParamField::Steps => self.steps.to_string(),
            ParamField::Guidance => format!("{:.1}", self.guidance),
            // The Seed essentials row absorbs the old SeedValue row: the
            // mode alone when Random, `mode · value` when the seed is
            // pinned (Fixed/Increment).
            ParamField::Seed => match self.seed_mode {
                SeedMode::Random => "random".to_string(),
                mode => format!(
                    "{} \u{00b7} {}",
                    mode.label(),
                    self.seed
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "\u{27e8}random\u{27e9}".to_string())
                ),
            },
            ParamField::Batch => self.batch.to_string(),
            ParamField::Format => format!("{:?}", self.format).to_lowercase(),
            ParamField::Upscale => self.upscale_model.clone().unwrap_or_else(|| "off".into()),
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
    /// The flat Create-form row list (`param_index` indexes into it).
    /// Rebuilt via [`App::refresh_create_rows`] whenever capabilities or
    /// the accordion state change.
    pub rows: Vec<crate::ui::create_form::CreateRow>,
    /// Advanced accordion disclosure state (persisted at
    /// `tui.advanced_open` / `tui.advanced_section`).
    pub advanced: crate::ui::create_form::AdvancedState,
    /// Scroll offset (in panel lines) of the parameters list — written by
    /// the renderer, read by mouse hit-testing.
    pub param_scroll: usize,
    pub capabilities: ModelCapabilities,
    pub progress: ProgressState,
    pub preview_image: Option<image::DynamicImage>,
    pub image_state: Option<StatefulProtocol>,
    /// When the preview is an animated GIF/APNG/WebP, holds the decoded
    /// frame list and current playback cursor. `image_state` always shows
    /// the frame at `animation.current`.
    pub animation: Option<crate::animation::AnimationState>,
    pub generating: bool,
    /// Number of images remaining in the current batch (0 when not batching).
    pub batch_remaining: u32,
    pub last_seed: Option<u64>,
    pub last_generation_time_ms: Option<u64>,
    pub error_message: Option<String>,
    pub model_description: String,
    /// Path of the most recently saved output — drives the activity
    /// strip's "done · saved to …" line. None when saving is disabled or
    /// the server kept the file.
    pub last_output_path: Option<std::path::PathBuf>,
}

impl GenerateState {
    /// Whether the inline Negative prompt editor is currently rendered and
    /// therefore focusable: the model supports it, the Advanced accordion
    /// is open, and the Negative section is the expanded one. This is the
    /// predicate every focus-routing or hit-test site should consult —
    /// checking `supports_negative_prompt` alone lets focus land on a row
    /// that isn't drawn.
    pub fn negative_visible(&self) -> bool {
        self.capabilities.supports_negative_prompt
            && self.advanced.open
            && self.advanced.expanded == Some(crate::ui::create_form::AdvSection::Negative)
    }
}

/// Which gallery view is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GalleryViewMode {
    #[default]
    Grid,
    Detail,
}

/// How long a Library scan stays fresh before re-entering the view
/// triggers a merged rescan.
pub(crate) const LIBRARY_RESCAN_STALE: std::time::Duration = std::time::Duration::from_secs(30);

/// State for the Gallery view.
pub struct GalleryState {
    pub entries: Vec<GalleryEntry>,
    pub selected: usize,
    pub preview_image: Option<image::DynamicImage>,
    pub image_state: Option<StatefulProtocol>,
    /// Frame loop for animated previews (GIF/APNG/WebP).
    pub animation: Option<crate::animation::AnimationState>,
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
    /// The `/` filter query (matches prompt, model, filename).
    pub filter: String,
    /// True while the filter line is being edited (keys route to it).
    pub filtering: bool,
    /// Indices into `entries` that pass the filter, in display order.
    /// Identity when the filter is empty. Thumbnail caches stay keyed by
    /// the underlying entry index, so filtering never invalidates them.
    pub filtered: Vec<usize>,
    /// One-slot fixed-protocol cache for the details side panel's
    /// thumbnail: (entry index, area width, area height, protocol).
    pub details_thumb: Option<(usize, u16, u16, ratatui_image::protocol::Protocol)>,
    /// When the last merged scan finished (None = never).
    pub last_scan: Option<std::time::Instant>,
    /// Set when the host registry changed — the next Library visit
    /// rescans regardless of staleness.
    pub dirty: bool,
    /// Remote sources that failed the last merged scan (header honesty).
    pub offline_hosts: usize,
}

impl Default for GalleryState {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            selected: 0,
            preview_image: None,
            image_state: None,
            animation: None,
            scanning: false,
            view_mode: GalleryViewMode::Grid,
            thumbnail_states: Vec::new(),
            thumb_dimensions: Vec::new(),
            thumb_fixed_cache: Vec::new(),
            grid_cols: 3,
            grid_scroll: 0,
            filter: String::new(),
            filtering: false,
            filtered: Vec::new(),
            details_thumb: None,
            last_scan: None,
            dirty: false,
            offline_hosts: 0,
        }
    }
}

impl GalleryState {
    /// Recompute `filtered` from the current entries + query, keeping the
    /// selection when it still matches (else snapping to the first match).
    pub fn refresh_filter(&mut self) {
        self.filtered = crate::gallery_scan::filter_entries(&self.entries, &self.filter);
        if !self.filtered.contains(&self.selected) {
            if let Some(&first) = self.filtered.first() {
                self.selected = first;
            }
        }
    }

    /// Position of the selected entry within the filtered list.
    pub fn selected_pos(&self) -> Option<usize> {
        self.filtered.iter().position(|&i| i == self.selected)
    }

    /// Whether entering the Library should kick a merged rescan: never
    /// while one is already running, always when the host registry
    /// changed, otherwise when the last scan is missing or stale.
    pub fn rescan_due(&self) -> bool {
        if self.scanning {
            return false;
        }
        self.dirty
            || self
                .last_scan
                .is_none_or(|t| t.elapsed() >= LIBRARY_RESCAN_STALE)
    }

    /// Handle one key while the `/` filter line is being edited. Returns
    /// true when the key was consumed. Esc clears the filter, Enter
    /// confirms it (keeps it applied), typed chars/Backspace edit it;
    /// modified keys (e.g. Ctrl+C) fall through to the normal map.
    pub fn handle_filter_key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> bool {
        match code {
            KeyCode::Esc => {
                self.filter.clear();
                self.filtering = false;
                self.refresh_filter();
                true
            }
            KeyCode::Enter => {
                self.filtering = false;
                true
            }
            KeyCode::Backspace => {
                self.filter.pop();
                self.refresh_filter();
                true
            }
            KeyCode::Char(c)
                if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.filter.push(c);
                self.refresh_filter();
                true
            }
            _ => false,
        }
    }
}

/// One machine a gallery print exists on. The first origin in
/// `GalleryEntry::origins` is the primary fetch route (the local copy
/// when one exists, else the first host that listed the print).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GalleryOrigin {
    /// Host id — `hosts::LOCAL_HOST_ID` for this machine, the registry
    /// slug (or URL slug for the unregistered `--host` server) otherwise.
    pub host_id: String,
    /// Origin URL; `None` for the local machine.
    pub url: Option<String>,
    /// Display name for the details panel ("This Mac", "hal9000", …).
    pub name: String,
}

impl GalleryOrigin {
    pub(crate) fn local() -> Self {
        Self {
            host_id: crate::hosts::LOCAL_HOST_ID.to_string(),
            url: None,
            name: crate::hosts::local_display_name().to_string(),
        }
    }

    pub(crate) fn from_host_entry(entry: &crate::hosts::HostEntry) -> Self {
        Self {
            host_id: entry.id.clone(),
            url: Some(entry.url.clone()),
            name: entry.display_name(),
        }
    }

    /// Synthesize an origin for a bare URL (the connected `--host` server
    /// when it isn't a registered machine, or legacy entries).
    pub(crate) fn remote_from_url(url: &str) -> Self {
        Self {
            host_id: crate::hosts::host_id_from_url(url),
            url: Some(url.to_string()),
            name: crate::hosts::host_port_label(url),
        }
    }

    pub fn is_local(&self) -> bool {
        self.url.is_none()
    }

    /// Whether this origin is THIS machine — either the plain local
    /// output dir, or the connected loopback server whose gallery is
    /// authoritative for it (one box, not a separate "machine").
    pub fn is_this_machine(&self) -> bool {
        self.host_id == crate::hosts::LOCAL_HOST_ID
    }
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
    /// Every machine this print exists on (first = primary fetch route).
    /// Empty is the legacy form — interpreted via `server_url`.
    pub origins: Vec<GalleryOrigin>,
}

impl GalleryEntry {
    pub fn filename(&self) -> String {
        self.path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".into())
    }

    /// Every machine this print exists on. Legacy entries (empty
    /// `origins`) synthesize one origin from `server_url`.
    pub(crate) fn owning_origins(&self) -> Vec<GalleryOrigin> {
        if !self.origins.is_empty() {
            return self.origins.clone();
        }
        match &self.server_url {
            Some(url) => vec![GalleryOrigin::remote_from_url(url)],
            None => vec![GalleryOrigin::local()],
        }
    }

    /// The primary origin — where previews/opens/upscales fetch from.
    pub(crate) fn primary_origin(&self) -> GalleryOrigin {
        self.owning_origins()
            .into_iter()
            .next()
            .unwrap_or_else(GalleryOrigin::local)
    }

    /// The details panel's Machine value: "This Mac" / host display name,
    /// or a count once the print exists on more than one machine.
    pub(crate) fn machine_label(&self) -> String {
        let origins = self.owning_origins();
        if origins.len() > 1 {
            format!("{} machines", origins.len())
        } else {
            origins[0].name.clone()
        }
    }
}

/// Confirm copy for deleting a print. Names every machine it will be
/// removed from when it exists on more than one.
pub(crate) fn delete_confirm_message(filename: &str, machines: usize) -> String {
    if machines > 1 {
        format!("Delete {filename}? It exists on {machines} machines. This can't be undone.")
    } else {
        format!("Delete {filename}?")
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
    // Preferences (DB-backed, see crate::prefs)
    PrefDefaultFormat,
    PrefReduceMotion,
    PrefShowTimeline,
    PrefConfirmDestructive,
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

/// Which pane has keyboard focus within the Settings view.
///
/// The Settings view is split into the Appearance swatch picker at the top and
/// the scrollable Configuration list below. Exactly one of them owns the
/// keyboard at any time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SettingsFocus {
    Appearance,
    #[default]
    Configuration,
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
    /// Active theme preset (drives [`App::theme`]).
    pub theme_preset: crate::ui::theme::ThemePreset,
    /// Which pane (Appearance vs Configuration) holds focus.
    pub focus: SettingsFocus,
    /// Columns in the last-rendered theme card grid — set by the
    /// Appearance renderer so ↑/↓ move by exactly one visual row (same
    /// pattern as `GalleryState::grid_cols`). 0 until first render;
    /// navigation clamps it to at least 1.
    pub appearance_cols: usize,
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
    /// Stepped connect-a-machine flow (Machines workspace):
    /// Url → optional ApiKey → Testing → saved or Failed with retry.
    MachineConnect {
        form: crate::hosts::ConnectForm,
    },
    SeedInput {
        input: String,
    },
    /// Free-text `WxH` entry for the Size essentials row.
    SizeInput {
        input: String,
    },
    HistorySearch {
        filter: String,
        selected: usize,
        results: Vec<String>,
    },
    /// The ^K command palette (see `crate::palette`).
    CommandPalette {
        filter: String,
        selected: usize,
        filtered: Vec<crate::palette::CommandId>,
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
        purpose: UpscalePickerPurpose,
    },
}

/// What selecting an entry in [`Popup::UpscaleModelSelector`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscalePickerPurpose {
    /// Library flow: upscale the selected gallery image now.
    RunNow,
    /// Create → Advanced → Upscale: set `GenerateParams::upscale_model`
    /// for the next generation (the `(off)` entry clears it).
    SetGenerateParam,
}

/// Label of the synthetic "clear" entry offered by the Create-side picker.
pub(crate) const UPSCALE_OFF_ENTRY: &str = "(off)";

#[derive(Debug, Clone)]
pub enum ConfirmAction {
    /// Delete a gallery image by index.
    DeleteGalleryImage,
    RemoveModel(String),
    /// Delete the currently selected script stage.
    DeleteScriptStage,
    /// Forget a registered machine (also deletes its saved API key).
    ForgetHost(String),
    /// Cancel a queued job on a registered machine.
    CancelHostJob {
        host_id: String,
        job_id: String,
    },
}

/// The root application state.
pub struct App {
    pub active_view: View,
    /// Compose vs chain-composer sub-mode of the Create view.
    pub create_mode: CreateMode,
    pub generate: GenerateState,
    pub gallery: GalleryState,
    pub models: ModelsState,
    /// Machines workspace state: host registry, per-host telemetry,
    /// selected-host queue snapshot. Logic lives in `crate::hosts`.
    pub machines: crate::hosts::MachinesState,
    /// Sticky generation target (persisted at `tui.generate_target`).
    pub target: crate::hosts::GenTarget,
    pub settings: SettingsState,
    /// DB-backed user preferences (Settings → Preferences), loaded once
    /// at boot and persisted per-key on toggle.
    pub prefs: crate::prefs::TuiPrefs,
    pub script: crate::ui::script_composer::ScriptComposerState,
    pub config: Config,
    pub server_url: Option<String>,
    pub picker: Picker,
    /// Studio motion effects (workspace fade, completion sweep) — gated
    /// behind reduce-motion; see `crate::motion`.
    pub motion: crate::motion::MotionState,
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
    /// Whether the Create view renders the Timeline panel. Backed by the
    /// `tui.show_timeline` settings key (owned by the Settings redesign —
    /// we read the raw key with a default of `true` so the panel honors
    /// the pref without a cross-PR dependency).
    pub show_timeline: bool,
}

/// Stored layout rectangles for mouse click hit-testing.
#[derive(Debug, Default, Clone)]
pub struct LayoutAreas {
    pub tab_bar: ratatui::layout::Rect,
    /// The main content region between the tab strip and the activity strip.
    pub content: ratatui::layout::Rect,
    /// The one-line activity strip above the status bar.
    pub activity: ratatui::layout::Rect,
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
    let mut cmd = std::process::Command::new(exe);
    configure_background_server_command(&mut cmd, port);
    cmd.spawn().ok()
}

/// Pure hit-test for the tab bar. Given an absolute column and the tab
/// bar's left edge, return the tab the click lands on, or `None` when the
/// click is past the last rendered tab (e.g. on the right-aligned
/// version/host indicator or blank space). The math mirrors
/// `ui::render_tab_bar`: one column of horizontal padding on the left,
/// then each tab is drawn as `" N Label "` (label length + 4 columns),
/// with a single-column divider between adjacent tabs.
pub(crate) fn tab_at_column(col: u16, tab_bar_x: u16) -> Option<View> {
    // Geometry is single-sourced in `ui::chrome::tab_spans` so the
    // renderer, the underline row, and this hit-test can never drift.
    crate::ui::chrome::tab_at_column(col, tab_bar_x)
}

/// Configure the background `mold serve` command — pure helper so tests
/// can inspect the args and env without actually spawning a process.
pub(crate) fn configure_background_server_command(cmd: &mut std::process::Command, port: u16) {
    cmd.args(["serve", "--port", &port.to_string(), "--log-file"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
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

        // Boot-time user preferences (Settings → Preferences).
        let prefs = crate::prefs::TuiPrefs::load();

        let mut params = GenerateParams::from_config(&config);
        // `tui.default_format` seeds a fresh session's Format param; the
        // session / per-model overlays below still win when they carry a
        // saved format.
        params.format = prefs.default_format;
        params.inference_mode = initial_mode;
        // Store the server URL in params.host so it's visible/editable
        if let Some(ref url) = server_url {
            params.host = Some(url.clone());
        }

        let family = family_for_model(&params.model, &config);
        let mut capabilities = capabilities_for_family(&family);
        // Restore the Advanced accordion where the user left it.
        let advanced = crate::ui::create_form::AdvancedState::load();

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
            capabilities = capabilities_for_family(&fam);
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

        let initial_preset = session
            .theme
            .as_deref()
            .map(crate::ui::theme::ThemePreset::from_slug)
            .unwrap_or_default();

        // `tui.show_timeline` is owned by the Settings redesign; read the
        // raw key (default true) so there's no cross-PR dependency.
        let show_timeline = mold_db::open_default()
            .ok()
            .flatten()
            .and_then(|db| {
                mold_db::Settings::new(&db)
                    .get_bool("tui.show_timeline")
                    .ok()
                    .flatten()
            })
            .unwrap_or(true);

        let rows = crate::ui::create_form::visible_rows(&capabilities, &advanced);
        let app = Ok(Self {
            active_view: View::Create,
            create_mode: CreateMode::default(),
            generate: GenerateState {
                prompt,
                negative_prompt,
                params,
                focus: GenerateFocus::Prompt,
                param_index: 0,
                rows,
                advanced,
                param_scroll: 0,
                capabilities,
                progress: ProgressState::default(),
                preview_image: None,
                image_state: None,
                animation: None,
                generating: false,
                batch_remaining: 0,
                last_seed: None,
                last_generation_time_ms: None,
                error_message: None,
                model_description,
                last_output_path: None,
            },
            gallery: GalleryState::default(),
            models: ModelsState {
                catalog,
                selected: 0,
                filter: String::new(),
                filtering: false,
            },
            machines: crate::hosts::MachinesState::load(),
            target: if local {
                // `mold tui --local` pins this run to the local engine;
                // the persisted target is left untouched.
                crate::hosts::GenTarget::Local
            } else {
                crate::hosts::GenTarget::load()
            },
            settings: {
                let first_model = config.models.keys().next().cloned();
                SettingsState {
                    selected_model: first_model,
                    row_index: 1, // skip first section header
                    theme_preset: initial_preset,
                    ..Default::default()
                }
            },
            prefs,
            script: crate::ui::script_composer::ScriptComposerState::default(),
            config,
            server_url,
            motion: crate::motion::MotionState::from_env_and_prefs(),
            picker,
            theme: initial_preset.build(),
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
            show_timeline,
        });

        // Spawn background gallery scan
        if let Ok(ref app) = app {
            app.spawn_gallery_scan();
        }

        app
    }

    /// Spawn a merged all-hosts gallery scan: the local output dir, the
    /// connected server (when set), and every registered Machines host,
    /// fetched concurrently with per-host API keys. Offline hosts
    /// contribute nothing but don't break the scan.
    pub fn spawn_gallery_scan(&self) {
        let tx = self.bg_tx.clone();
        let sources =
            crate::gallery_scan::scan_sources(self.server_url.as_deref(), &self.machines.registry);
        self.tokio_handle.spawn(async move {
            let scan = crate::gallery_scan::scan_all_hosts(sources).await;
            let _ = tx.send(BackgroundEvent::GalleryScanComplete(scan));
        });
    }

    /// Whether the event loop should poll `/api/status` instead of local sysinfo.
    /// True when connected to a server AND not forced into local mode.
    pub fn should_poll_remote(&self) -> bool {
        self.server_url.is_some() && self.generate.params.inference_mode != InferenceMode::Local
    }

    /// Whether a completed generation should be written to the local output
    /// dir by the TUI. False when the server (running on the same or a
    /// different machine) has already saved the file to its own output dir
    /// — otherwise the TUI writes a second copy with a slightly-different
    /// timestamp suffix, and the next gallery scan surfaces both as
    /// duplicate tiles. Also false when output is explicitly disabled.
    pub fn should_save_output_locally(&self) -> bool {
        if self.config.is_output_disabled() {
            return false;
        }
        !self.should_poll_remote()
    }

    /// Per-response variant of [`should_save_output_locally`]: in Auto
    /// mode the backend transparently falls back to local inference when
    /// the connected server becomes unreachable, but `server_url` is left
    /// set so `should_save_output_locally()` — which only looks at the
    /// mode and connection — classifies the completion as remote and
    /// drops the file. The completion event carries `from_local`, set by
    /// the backend on every local-path emission; when it's true we must
    /// write the file locally even if the TUI still thinks it is
    /// remote-connected. `is_output_disabled` still wins because the
    /// user explicitly opted out of saving anywhere.
    pub fn should_persist_response_locally(&self, from_local: bool) -> bool {
        if self.config.is_output_disabled() {
            return false;
        }
        if from_local {
            return true;
        }
        !self.should_poll_remote()
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
                    let _ = tx.send(BackgroundEvent::HostDevicesUpdate {
                        host_id: crate::hosts::LOCAL_HOST_ID.to_string(),
                        devices: client.devices().await.ok(),
                    });
                    let _ = tx.send(BackgroundEvent::HostCapabilitiesUpdate {
                        host_id: crate::hosts::LOCAL_HOST_ID.to_string(),
                        capabilities: client.server_capabilities().await.ok(),
                    });
                }
                Err(_) => {
                    // Server became unreachable — clear stale status so the UI
                    // stops showing the last-known hostname/memory.
                    let _ = tx.send(BackgroundEvent::ServerStatusUpdate(None));
                    let _ = tx.send(BackgroundEvent::HostDevicesUpdate {
                        host_id: crate::hosts::LOCAL_HOST_ID.to_string(),
                        devices: None,
                    });
                    let _ = tx.send(BackgroundEvent::HostCapabilitiesUpdate {
                        host_id: crate::hosts::LOCAL_HOST_ID.to_string(),
                        capabilities: None,
                    });
                }
            }
        });
    }

    /// Multi-host polling tick — driven by the 2 s resource-refresh timer.
    /// The plan (which hosts, whether a queue fetch rides along) lives in
    /// `hosts::MachinesState::poll_plan`.
    pub fn tick_host_polling(&mut self) {
        if !self.machines.poll_due() {
            return;
        }
        let plan = self
            .machines
            .poll_plan(self.active_view == View::Machines, &self.target);
        for entry in plan.status_hosts {
            self.spawn_host_status_fetch(entry);
        }
        if let Some(entry) = plan.queue_host {
            self.spawn_host_queue_fetch(entry);
        }
    }

    /// Spawn a `/api/status` fetch for one registered host.
    pub fn spawn_host_status_fetch(&self, entry: crate::hosts::HostEntry) {
        self.tokio_handle
            .spawn(crate::hosts::fetch_host_status(entry, self.bg_tx.clone()));
    }

    /// Spawn a queue-listing fetch for one registered host.
    pub fn spawn_host_queue_fetch(&self, entry: crate::hosts::HostEntry) {
        self.tokio_handle
            .spawn(crate::hosts::fetch_host_queue(entry, self.bg_tx.clone()));
    }

    /// Spawn the connect-flow test fetch against `url`.
    fn spawn_machine_connect_test(&self, url: String, api_key: Option<String>) {
        self.tokio_handle.spawn(crate::hosts::test_connection(
            url,
            api_key,
            self.bg_tx.clone(),
        ));
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
            self.gallery.animation = None;
        }

        let tx = self.bg_tx.clone();
        // Route the upscale to the print's OWNING host when it is remote
        // (with that host's saved API key); local prints keep the
        // connected-server-then-local-fallback behavior.
        let route = entry.primary_origin();
        let server_url = match &route.url {
            Some(url) => Some(url.clone()),
            None => self.server_url.clone(),
        };
        let server_host_id = route.url.as_ref().map(|_| route.host_id.clone());
        let config = self.config.clone();
        let source_path = entry.path.clone();

        let handle = self.tokio_handle.spawn(async move {
            // Read image bytes
            let image_bytes = if let Some(ref url) = route.url {
                let filename = entry.filename();
                match crate::gallery_scan::fetch_and_cache_image(url, &route.host_id, &filename)
                    .await
                {
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
                metadata: None,
            };

            // Try server first — use SSE streaming for tile progress.
            // Remote-owned prints upscale on their owning host (with its
            // saved API key); local prints use the connected server.
            if let Some(ref url) = server_url {
                let api_key = server_host_id
                    .as_deref()
                    .and_then(crate::hosts::api_key_for);
                let client = crate::hosts::client_for(url, api_key.as_deref());

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
                    0,
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
            crate::session::TuiSession::from_params(&prompt_text, &neg_text, &self.generate.params)
                .with_theme(self.settings.theme_preset);
        session.save();
    }

    /// Apply a theme preset — rebuilds [`App::theme`], records the
    /// selection, and persists the change to the session file right
    /// away. Persisting on every apply (rather than only on shutdown or
    /// after a generation) means a crash, force-quit, or quick
    /// theme-change-then-close all keep the user's selection.
    pub fn apply_theme_preset(&mut self, preset: crate::ui::theme::ThemePreset) {
        self.settings.theme_preset = preset;
        self.theme = preset.build();
        self.save_session();
    }

    pub fn update_model(&mut self, model_name: &str) {
        let model_name = model_name.to_string();
        let outgoing_model = self.generate.params.model.clone();
        // No-op when switching to the same model — avoids clobbering
        // current in-memory params with an older DB snapshot.
        if outgoing_model == model_name {
            return;
        }

        // Snapshot the outgoing model's current params into model_prefs
        // so they're restored when the user comes back to this model.
        // Keyed on the *outgoing* name — not the incoming one.
        if !outgoing_model.is_empty() {
            self.save_prefs_for_model(&outgoing_model);
        }

        self.generate.params.model = model_name.clone();

        // Use server catalog defaults when connected to a remote server,
        // local config otherwise.
        let used_remote = if self.should_poll_remote() {
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
        if family == "qwen-image-edit" {
            if let Some(path) = self.generate.params.source_image_path.as_deref() {
                if let Some((width, height)) = qwen_image_edit_dimensions_for_path(path) {
                    self.generate.params.width = width;
                    self.generate.params.height = height;
                }
            }
        }
        self.generate.capabilities = capabilities_for_family(&family);
        self.refresh_create_rows();
        self.generate.param_index = 0;

        // Apply saved per-model prefs last, so a user's explicit choices
        // override the manifest/catalog defaults we just restored. Only
        // generation params move — the prompt textareas stay as-is so a
        // model flip mid-typing doesn't wipe what the user is writing.
        self.apply_prefs_for_model(&model_name);
    }

    /// Persist the *currently-displayed* generation params under `model`.
    /// Used by `update_model` before switching away, and by any future
    /// explicit "save as default for this model" flow.
    fn save_prefs_for_model(&self, model: &str) {
        let db = match mold_db::open_default() {
            Ok(Some(db)) => db,
            _ => return,
        };
        let p = &self.generate.params;
        let prefs = mold_db::ModelPrefs {
            width: Some(p.width),
            height: Some(p.height),
            steps: Some(p.steps),
            guidance: Some(p.guidance),
            // Canonical Display form: "ddim" / "euler-ancestral" /
            // "uni-pc". Matches `mold_core::Scheduler::Display` and what
            // `mold-db::config_sync` writes via `mold config set`, so
            // rows written by either surface round-trip cleanly.
            scheduler: p.scheduler.map(|s| s.to_string()),
            seed_mode: Some(p.seed_mode.label().to_string()),
            batch: Some(p.batch),
            format: Some(format!("{:?}", p.format).to_lowercase()),
            lora_path: p.lora_path.clone(),
            lora_scale: Some(p.lora_scale),
            expand: Some(p.expand),
            offload: Some(p.offload),
            strength: Some(p.strength),
            control_scale: Some(p.control_scale),
            frames: None,
            fps: None,
            last_prompt: None,
            last_negative: None,
        };
        if let Err(e) = prefs.save(&db, model) {
            tracing::warn!(error = %e, model, "save_prefs_for_model failed");
        }
    }

    /// If the DB has a `model_prefs` row for `model`, overlay its saved
    /// generation params onto `self.generate.params`. Skips prompt text —
    /// those stay under the user's control.
    fn apply_prefs_for_model(&mut self, model: &str) {
        let db = match mold_db::open_default() {
            Ok(Some(db)) => db,
            _ => return,
        };
        let Some(prefs) = mold_db::ModelPrefs::load(&db, model).ok().flatten() else {
            return;
        };
        let p = &mut self.generate.params;
        if let Some(w) = prefs.width {
            p.width = w;
        }
        if let Some(h) = prefs.height {
            p.height = h;
        }
        if let Some(s) = prefs.steps {
            p.steps = s;
        }
        if let Some(g) = prefs.guidance {
            p.guidance = g;
        }
        if let Some(ref sched) = prefs.scheduler {
            // `Scheduler::FromStr` accepts both the canonical Display
            // form ("euler-ancestral", "uni-pc") that we and
            // `mold-db::config_sync` write, and the legacy debug-lower
            // form ("eulerancestral", "unipc") written by pre-#265 TUI
            // builds, so existing DBs don't lose their saved choice.
            p.scheduler = sched.parse().ok();
        }
        if let Some(ref sm) = prefs.seed_mode {
            p.seed_mode = match sm.as_str() {
                "fixed" => SeedMode::Fixed,
                "increment" => SeedMode::Increment,
                _ => SeedMode::Random,
            };
        }
        if let Some(b) = prefs.batch {
            p.batch = b;
        }
        if let Some(ref f) = prefs.format {
            p.format = match f.as_str() {
                "jpeg" => mold_core::OutputFormat::Jpeg,
                _ => mold_core::OutputFormat::Png,
            };
        }
        if prefs.lora_path.is_some() {
            p.lora_path = prefs.lora_path.clone();
        }
        if let Some(ls) = prefs.lora_scale {
            p.lora_scale = ls;
        }
        if let Some(e) = prefs.expand {
            p.expand = e;
        }
        if let Some(o) = prefs.offload {
            p.offload = o;
        }
        if let Some(s) = prefs.strength {
            p.strength = s;
        }
        if let Some(cs) = prefs.control_scale {
            p.control_scale = cs;
        }
    }

    /// Rebuild the Create-form row list from the current capabilities and
    /// accordion state, clamping the selection into range.
    pub(crate) fn refresh_create_rows(&mut self) {
        self.generate.rows = crate::ui::create_form::visible_rows(
            &self.generate.capabilities,
            &self.generate.advanced,
        );
        if self.generate.param_index >= self.generate.rows.len() {
            self.generate.param_index = self.generate.rows.len().saturating_sub(1);
        }
    }

    /// Expand one accordion section (collapsing any other) or collapse all
    /// with `None`. Persists the accordion state, rebuilds the rows, and
    /// keeps the selection on the section's own row. Focus escapes the
    /// inline Negative editor if the change hides it.
    fn set_advanced_expanded(&mut self, sec: Option<crate::ui::create_form::AdvSection>) {
        self.generate.advanced.expanded = sec;
        self.generate.advanced.save();
        self.refresh_create_rows();
        if let Some(sec) = sec {
            if let Some(idx) = self
                .generate
                .rows
                .iter()
                .position(|r| *r == crate::ui::create_form::CreateRow::Section(sec))
            {
                self.generate.param_index = idx;
            }
        }
        if self.generate.focus == GenerateFocus::NegativePrompt && !self.generate.negative_visible()
        {
            self.generate.focus = GenerateFocus::Parameters;
        }
    }

    /// The selected model's default canvas size — the anchor area for the
    /// Size essentials row's aspect presets. Prefers the server catalog
    /// defaults when remote-connected, mirroring `update_model`.
    fn model_default_size(&self) -> (u32, u32) {
        if self.should_poll_remote() {
            if let Some(entry) = self
                .models
                .catalog
                .iter()
                .find(|m| m.name == self.generate.params.model)
            {
                return (entry.defaults.default_width, entry.defaults.default_height);
            }
        }
        let mc = self
            .config
            .resolved_model_config(&self.generate.params.model);
        (
            mc.effective_width(&self.config),
            mc.effective_height(&self.config),
        )
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

        // Library `/` filter editing swallows its editing keys; anything
        // it doesn't consume (e.g. Ctrl+C) falls through to the map.
        if self.active_view == View::Library && self.gallery.filtering {
            if let CrosstermEvent::Key(key) = &event {
                if self.gallery.handle_filter_key(key.code, key.modifiers) {
                    return;
                }
            }
        }

        // If we're in a text input field, let the textarea handle the event first
        if self.active_view == View::Create {
            let in_text_field = matches!(
                self.generate.focus,
                GenerateFocus::Prompt | GenerateFocus::NegativePrompt
            );
            if in_text_field {
                if let CrosstermEvent::Key(key) = &event {
                    // Let certain keys bypass the textarea
                    match (key.code, key.modifiers) {
                        // TUI-global shortcuts that bypass the textarea.
                        // ^K deliberately no longer performs the emacs
                        // kill-to-end-of-line in prompt fields — it opens
                        // the command palette everywhere instead.
                        (KeyCode::Tab, KeyModifiers::NONE)
                        | (KeyCode::BackTab, KeyModifiers::SHIFT)
                        | (KeyCode::Char('c'), KeyModifiers::CONTROL) // quit
                        | (KeyCode::Char('g'), KeyModifiers::CONTROL) // generate
                        | (KeyCode::Char('m'), KeyModifiers::CONTROL) // model selector
                        | (KeyCode::Char('r'), KeyModifiers::CONTROL) // seed mode
                        | (KeyCode::Char('k'), KeyModifiers::CONTROL) // command palette
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
                        | (KeyCode::Char('5'), KeyModifiers::ALT)
                        | (KeyCode::Left, KeyModifiers::ALT)
                        | (KeyCode::Right, KeyModifiers::ALT) => {
                            // Fall through for view switching (Alt+1..5).
                        }
                        (KeyCode::Char('n'), KeyModifiers::ALT)
                        | (KeyCode::Char('N'), KeyModifiers::ALT) => {
                            // Fall through so Alt+N reaches
                            // `Action::ToggleNegativePrompt` even while the
                            // Prompt or Negative textarea has focus.
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
                    KeyCode::Up | KeyCode::Char('k') if *selected > 0 => {
                        *selected -= 1;
                    }
                    KeyCode::Down | KeyCode::Char('j') if *selected + 1 < filtered.len() => {
                        *selected += 1;
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
                    purpose,
                }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        let purpose = *purpose;
                        if let Some(model) = filtered.get(*selected).cloned() {
                            self.close_popup();
                            match purpose {
                                UpscalePickerPurpose::RunNow => self.spawn_upscale(model),
                                UpscalePickerPurpose::SetGenerateParam => {
                                    self.generate.params.upscale_model =
                                        (model != UPSCALE_OFF_ENTRY).then_some(model);
                                }
                            }
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k') if *selected > 0 => {
                        *selected -= 1;
                    }
                    KeyCode::Down | KeyCode::Char('j') if *selected + 1 < filtered.len() => {
                        *selected += 1;
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
                Some(Popup::SizeInput { input }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        let text = input.trim().to_string();
                        self.close_popup();
                        if let Some((w, h)) = parse_size_input(&text) {
                            self.generate.params.width = w;
                            self.generate.params.height = h;
                        }
                    }
                    KeyCode::Char(c)
                        if c.is_ascii_digit() || matches!(c, 'x' | 'X' | '\u{00d7}' | ' ') =>
                    {
                        input.push(c)
                    }
                    KeyCode::Backspace => {
                        input.pop();
                    }
                    _ => {}
                },
                Some(Popup::MachineConnect { form }) => {
                    use crate::hosts::{connect_advance, ConnectEffect, ConnectInput};
                    let input = match key.code {
                        KeyCode::Esc => Some(ConnectInput::Esc),
                        KeyCode::Enter => Some(ConnectInput::Enter),
                        KeyCode::Backspace => Some(ConnectInput::Backspace),
                        KeyCode::Char(c) => Some(ConnectInput::Char(c)),
                        _ => None,
                    };
                    if let Some(input) = input {
                        match connect_advance(form, input) {
                            ConnectEffect::None => {}
                            ConnectEffect::Close => self.close_popup(),
                            ConnectEffect::StartTest => {
                                let url = form.url.clone();
                                let api_key = Some(form.api_key.clone()).filter(|k| !k.is_empty());
                                self.spawn_machine_connect_test(url, api_key);
                            }
                            // Save only arises from the async TestOk input,
                            // which is fed in by the background handler.
                            ConnectEffect::Save => {}
                        }
                    }
                }
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
                        if (key.modifiers == KeyModifiers::NONE || key.code == KeyCode::Up)
                            && *selected > 0 =>
                    {
                        *selected -= 1;
                    }
                    KeyCode::Down | KeyCode::Char('j')
                        if (key.modifiers == KeyModifiers::NONE || key.code == KeyCode::Down)
                            && *selected + 1 < results.len() =>
                    {
                        *selected += 1;
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
                Some(Popup::CommandPalette {
                    filter,
                    selected,
                    filtered,
                }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        if let Some(id) = filtered.get(*selected).copied() {
                            self.close_popup();
                            self.dispatch_action(crate::palette::command_action(id));
                        } else {
                            self.close_popup();
                        }
                    }
                    // No j/k aliases — labels contain those letters.
                    KeyCode::Up if *selected > 0 => {
                        *selected -= 1;
                    }
                    KeyCode::Down if *selected + 1 < filtered.len() => {
                        *selected += 1;
                    }
                    KeyCode::Char(c) => {
                        filter.push(c);
                        *filtered = crate::palette::filter_commands(filter)
                            .into_iter()
                            .map(|cmd| cmd.id)
                            .collect();
                        if *selected >= filtered.len() {
                            *selected = filtered.len().saturating_sub(1);
                        }
                    }
                    KeyCode::Backspace => {
                        filter.pop();
                        *filtered = crate::palette::filter_commands(filter)
                            .into_iter()
                            .map(|cmd| cmd.id)
                            .collect();
                        if *selected >= filtered.len() {
                            *selected = filtered.len().saturating_sub(1);
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

                // Tab bar clicks — switch views.
                // Clicks on empty space to the right of the last rendered
                // tab (e.g. the version/host indicator) must be a no-op,
                // not a stealth jump to Settings.
                if self.layout.tab_bar.contains((col, row).into()) {
                    if let Some(view) = tab_at_column(col, self.layout.tab_bar.x) {
                        self.set_active_view(view);
                        return;
                    }
                    // Click past all tabs — leave the active view alone.
                    return;
                }

                // Generate view clicks
                if self.active_view == View::Create {
                    let pos: ratatui::layout::Position = (col, row).into();
                    if self.layout.prompt.contains(pos) {
                        self.generate.focus = GenerateFocus::Prompt;
                    } else if self.layout.negative_prompt.contains(pos)
                        && self.generate.negative_visible()
                    {
                        self.generate.focus = GenerateFocus::NegativePrompt;
                    } else if self.layout.parameters.contains(pos) {
                        self.generate.focus = GenerateFocus::Parameters;
                        // Select and activate the row under the click,
                        // accounting for multi-line rows and scroll.
                        let line = (row - self.layout.parameters.y).saturating_sub(1) as usize
                            + self.generate.param_scroll;
                        let has_desc = !self.generate.model_description.is_empty();
                        if let Some(idx) =
                            crate::ui::param_form::row_at_line(&self.generate.rows, has_desc, line)
                        {
                            self.generate.param_index = idx;
                            self.activate_current_param();
                        }
                    } else {
                        // Click on preview or elsewhere — go to navigation
                        self.generate.focus = GenerateFocus::Navigation;
                    }
                }

                // Gallery view clicks
                if self.active_view == View::Library
                    && self.gallery.view_mode == GalleryViewMode::Grid
                {
                    let pos: ratatui::layout::Position = (col, row).into();
                    if self.layout.gallery_grid.contains(pos) {
                        // Source cell dimensions from the renderer so the
                        // hit-test can't drift from the rendered grid —
                        // the previous hard-coded `cell_h = 14` was the
                        // "finicky clicks" bug after CELL_H was shrunk
                        // to 12 when filename labels were removed.
                        let cell_w = crate::ui::gallery::CELL_W;
                        let cell_h = crate::ui::gallery::CELL_H;
                        let cols = self.gallery.grid_cols.max(1);
                        let rel_x = col.saturating_sub(self.layout.gallery_grid.x);
                        let rel_y = row.saturating_sub(self.layout.gallery_grid.y);
                        let grid_col = (rel_x / cell_w) as usize;
                        let grid_row = (rel_y / cell_h) as usize + self.gallery.grid_scroll;
                        // The grid renders the *filtered* list — a click
                        // position maps through it to the entry index.
                        let pos = grid_row * cols + grid_col;
                        if let Some(&idx) = self.gallery.filtered.get(pos) {
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
                                self.active_view = View::Create;
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
            purpose,
        }) = &mut self.popup
        {
            let query = filter.to_lowercase();
            let mut list: Vec<String> = Vec::new();
            // The Create-side picker keeps its "(off)" clear entry on top.
            if *purpose == UpscalePickerPurpose::SetGenerateParam
                && UPSCALE_OFF_ENTRY.to_lowercase().contains(&query)
            {
                list.push(UPSCALE_OFF_ENTRY.to_string());
            }
            list.extend(
                all.into_iter()
                    .filter(|name| name.to_lowercase().contains(&query)),
            );
            *filtered = list;
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

    /// Switch the active workspace, running per-view entry hooks:
    /// Machines polls immediately; Library kicks a merged rescan when the
    /// last one is stale (>30 s), missing, or the host registry changed.
    fn set_active_view(&mut self, view: View) {
        if self.active_view != view {
            self.motion
                .trigger_workspace_fade(self.layout.content, self.theme.bg);
        }
        self.active_view = view;
        match view {
            View::Machines => {
                // Entering Machines polls immediately instead of
                // waiting out the background cadence.
                self.machines.force_poll();
                self.tick_host_polling();
            }
            View::Library if self.gallery.rescan_due() => {
                self.gallery.scanning = true;
                self.spawn_gallery_scan();
            }
            _ => {}
        }
    }

    /// Dispatch a semantic action.
    pub fn dispatch_action(&mut self, action: Action) {
        match action {
            Action::Quit => self.should_quit = true,
            Action::SwitchView(view) => {
                self.set_active_view(view);
            }
            Action::ViewNext => {
                let i = self.active_view.index();
                self.set_active_view(View::ALL[(i + 1) % View::ALL.len()]);
            }
            Action::ViewPrev => {
                let i = self.active_view.index();
                self.set_active_view(View::ALL[(i + View::ALL.len() - 1) % View::ALL.len()]);
            }
            Action::ChainEnter => {
                let switched =
                    self.active_view != View::Create || self.create_mode != CreateMode::Chain;
                self.active_view = View::Create;
                self.create_mode = CreateMode::Chain;
                if switched {
                    self.motion
                        .trigger_workspace_fade(self.layout.content, self.theme.bg);
                }
            }
            Action::ChainExit => {
                self.create_mode = CreateMode::Compose;
            }
            Action::ToggleAdvanced => {
                self.generate.advanced.open = !self.generate.advanced.open;
                self.generate.advanced.save();
                self.refresh_create_rows();
                // Closing the disclosure hides the inline Negative editor;
                // don't leave focus in a textarea that isn't drawn.
                if self.generate.focus == GenerateFocus::NegativePrompt
                    && !self.generate.negative_visible()
                {
                    self.generate.focus = GenerateFocus::Parameters;
                }
            }
            Action::OpenPalette => {
                self.popup = Some(Popup::CommandPalette {
                    filter: String::new(),
                    selected: 0,
                    filtered: crate::palette::all_commands()
                        .into_iter()
                        .map(|cmd| cmd.id)
                        .collect(),
                });
            }
            Action::SetTheme(preset) => {
                self.apply_theme_preset(preset);
            }
            Action::FocusNext if self.active_view == View::Create => {
                // Use `negative_visible()` instead of `supports_negative_prompt`
                // alone so Tab skips the Negative pane when the user has
                // collapsed it. Otherwise focus can land on a hidden textarea
                // and keystrokes get routed nowhere.
                self.generate.focus = self.generate.focus.next(self.generate.negative_visible());
            }
            Action::FocusPrev if self.active_view == View::Create => {
                self.generate.focus = self.generate.focus.prev(self.generate.negative_visible());
            }
            Action::FocusNext | Action::FocusPrev if self.active_view == View::Settings => {
                // Two panes → next and prev are the same flip. Tab is the
                // deterministic way to leave the Appearance card grid
                // without moving (and live-applying) the theme selection.
                self.settings.focus = match self.settings.focus {
                    SettingsFocus::Appearance => SettingsFocus::Configuration,
                    SettingsFocus::Configuration => SettingsFocus::Appearance,
                };
            }
            Action::FocusNext | Action::FocusPrev if self.active_view == View::Machines => {
                use crate::hosts::MachinesFocus;
                self.machines.focus = match self.machines.focus {
                    MachinesFocus::HostList => MachinesFocus::Detail,
                    MachinesFocus::Detail => MachinesFocus::HostList,
                };
            }
            Action::Up => match self.active_view {
                View::Create => {
                    if self.generate.focus == GenerateFocus::Parameters
                        && self.generate.param_index > 0
                    {
                        self.generate.param_index -= 1;
                    }
                }
                View::Library => {
                    let cols = self.gallery.grid_cols.max(1);
                    match self.gallery.view_mode {
                        GalleryViewMode::Grid => {
                            if let Some(pos) = self.gallery.selected_pos() {
                                if pos >= cols {
                                    self.gallery.selected = self.gallery.filtered[pos - cols];
                                }
                            }
                        }
                        GalleryViewMode::Detail => {
                            if let Some(pos) = self.gallery.selected_pos() {
                                if pos > 0 {
                                    self.gallery.selected = self.gallery.filtered[pos - 1];
                                    self.load_gallery_preview();
                                }
                            }
                        }
                    }
                }
                View::Models => {
                    if self.models.selected > 0 {
                        self.models.selected -= 1;
                    }
                }
                View::Machines => match self.machines.focus {
                    crate::hosts::MachinesFocus::HostList => {
                        if self.machines.select_prev() {
                            self.spawn_selected_host_queue_fetch();
                        }
                    }
                    crate::hosts::MachinesFocus::Detail => self.machines.queue_select_prev(),
                },
                View::Settings => self.settings_navigate(-1),
            },
            Action::Down => match self.active_view {
                View::Create => {
                    if self.generate.focus == GenerateFocus::Parameters
                        && self.generate.param_index + 1 < self.generate.rows.len()
                    {
                        self.generate.param_index += 1;
                    }
                }
                View::Library => {
                    let cols = self.gallery.grid_cols.max(1);
                    let len = self.gallery.filtered.len();
                    match self.gallery.view_mode {
                        GalleryViewMode::Grid => {
                            if let Some(pos) = self.gallery.selected_pos() {
                                if pos + cols < len {
                                    self.gallery.selected = self.gallery.filtered[pos + cols];
                                }
                            }
                        }
                        GalleryViewMode::Detail => {
                            if let Some(pos) = self.gallery.selected_pos() {
                                if pos + 1 < len {
                                    self.gallery.selected = self.gallery.filtered[pos + 1];
                                    self.load_gallery_preview();
                                }
                            }
                        }
                    }
                }
                View::Models => {
                    if self.models.selected + 1 < self.models.catalog.len() {
                        self.models.selected += 1;
                    }
                }
                View::Machines => match self.machines.focus {
                    crate::hosts::MachinesFocus::HostList => {
                        if self.machines.select_next() {
                            self.spawn_selected_host_queue_fetch();
                        }
                    }
                    crate::hosts::MachinesFocus::Detail => self.machines.queue_select_next(),
                },
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
            Action::Generate if self.active_view == View::Create && !self.generate.generating => {
                self.start_generation();
            }
            // Alt+N muscle memory: open Advanced, expand the Negative
            // section, and focus the inline editor in one stroke. A no-op
            // for models without negative-prompt support.
            Action::ToggleNegativePrompt if self.generate.capabilities.supports_negative_prompt => {
                self.active_view = View::Create;
                self.generate.advanced.open = true;
                self.set_advanced_expanded(Some(crate::ui::create_form::AdvSection::Negative));
                if let Some(idx) = self
                    .generate
                    .rows
                    .iter()
                    .position(|r| *r == crate::ui::create_form::CreateRow::NegativeEditor)
                {
                    self.generate.param_index = idx;
                }
                self.generate.focus = GenerateFocus::NegativePrompt;
            }
            Action::Confirm => match self.active_view {
                View::Create => {
                    if self.generate.focus == GenerateFocus::Parameters {
                        self.activate_current_param();
                    } else if !self.generate.generating {
                        self.start_generation();
                    }
                }
                View::Library => match self.gallery.view_mode {
                    GalleryViewMode::Grid => {
                        if !self.gallery.filtered.is_empty() {
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
                        self.active_view = View::Create;
                        self.generate.focus = GenerateFocus::Prompt;
                    }
                }
                View::Machines => {}
                View::Settings => {
                    // Enter only edits a Configuration row. When the
                    // Appearance swatch grid holds focus the preset is
                    // already live-applied, so Enter has no work to do —
                    // falling through to `settings_confirm()` would read
                    // `row_index` from the Configuration list and open
                    // the popup for whichever field happens to be
                    // selected there.
                    if self.settings.focus == SettingsFocus::Configuration {
                        self.settings_confirm();
                    }
                }
            },
            Action::PullModel if self.active_view == View::Models => {
                if let Some(model) = self.models.catalog.get(self.models.selected) {
                    let model_name = model.name.clone();
                    let tx = self.bg_tx.clone();

                    if self.should_poll_remote() {
                        // Pull via server when connected remotely
                        let url = self.server_url.clone().unwrap();
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
                if self.active_view == View::Library && self.upscale_in_progress {
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
                    self.generate.progress.push_log(ProgressLogEntry {
                        message: "Upscale cancelled".into(),
                        style: ProgressStyle::Warning,
                    });
                } else if self.active_view == View::Library
                    && self.gallery.view_mode == GalleryViewMode::Detail
                {
                    self.gallery.view_mode = GalleryViewMode::Grid;
                    self.gallery.preview_image = None;
                    self.gallery.image_state = None;
                    self.gallery.animation = None;
                } else {
                    self.generate.error_message = None;
                }
            }
            Action::HistoryPrev
                if self.active_view == View::Create
                    && self.generate.focus == GenerateFocus::Prompt =>
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
            Action::HistoryNext
                if self.active_view == View::Create
                    && self.generate.focus == GenerateFocus::Prompt =>
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
            Action::Unfocus if self.active_view == View::Create => {
                self.generate.focus = GenerateFocus::Navigation;
            }
            Action::GridLeft
                if self.active_view == View::Library
                    && self.gallery.view_mode == GalleryViewMode::Grid =>
            {
                if let Some(pos) = self.gallery.selected_pos() {
                    if pos > 0 {
                        self.gallery.selected = self.gallery.filtered[pos - 1];
                    }
                }
            }
            Action::GridRight
                if self.active_view == View::Library
                    && self.gallery.view_mode == GalleryViewMode::Grid =>
            {
                if let Some(pos) = self.gallery.selected_pos() {
                    if pos + 1 < self.gallery.filtered.len() {
                        self.gallery.selected = self.gallery.filtered[pos + 1];
                    }
                }
            }
            Action::EditAndGenerate if self.active_view == View::Library => {
                self.load_gallery_into_generate();
            }
            Action::Regenerate if self.active_view == View::Library => {
                self.load_gallery_into_generate();
                if !self.generate.generating {
                    self.start_generation();
                }
            }
            Action::DeleteImage if self.active_view == View::Library => {
                if let Some(entry) = self.gallery.entries.get(self.gallery.selected) {
                    let filename = entry.filename();
                    let machines = entry.owning_origins().len();
                    self.request_confirm(
                        delete_confirm_message(&filename, machines),
                        ConfirmAction::DeleteGalleryImage,
                    );
                }
            }
            Action::FilterLibrary
                if self.active_view == View::Library
                    && self.gallery.view_mode == GalleryViewMode::Grid =>
            {
                self.gallery.filtering = true;
            }
            Action::FilterLibraryClear if self.active_view == View::Library => {
                self.gallery.filter.clear();
                self.gallery.filtering = false;
                self.gallery.refresh_filter();
            }
            Action::OpenFile => {
                self.open_gallery_file();
            }
            Action::UpscaleImage
                if self.active_view == View::Library
                    && !self.upscale_in_progress
                    && self.gallery.entries.get(self.gallery.selected).is_some() =>
            {
                let models = self.available_upscaler_models();
                self.popup = Some(Popup::UpscaleModelSelector {
                    filter: String::new(),
                    selected: 0,
                    filtered: models,
                    purpose: UpscalePickerPurpose::RunNow,
                });
            }
            Action::RemoveModel if self.active_view == View::Models => {
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
                    self.request_confirm(message, ConfirmAction::RemoveModel(name));
                }
            }
            Action::MachinesConnect => {
                self.active_view = View::Machines;
                self.popup = Some(Popup::MachineConnect {
                    form: crate::hosts::ConnectForm::default(),
                });
            }
            Action::MachinesSetTarget if self.active_view == View::Machines => {
                self.target = self.machines.target_for_selected(&self.target);
                self.target.save();
            }
            Action::MachinesForget if self.active_view == View::Machines => {
                if let crate::hosts::MachineRowId::Host(id) = self.machines.selected_row() {
                    let name = self
                        .machines
                        .registry
                        .get(&id)
                        .map(|e| e.display_name())
                        .unwrap_or_else(|| id.clone());
                    self.popup = Some(Popup::Confirm {
                        message: format!("Forget {name}? Its saved API key is deleted too."),
                        on_confirm: ConfirmAction::ForgetHost(id),
                    });
                }
            }
            Action::MachinesRefresh if self.active_view == View::Machines => {
                self.machines.force_poll();
                self.tick_host_polling();
            }
            Action::MachinesCancelJob
                if self.active_view == View::Machines
                    && self.machines.focus == crate::hosts::MachinesFocus::Detail =>
            {
                if let Some((host_id, job)) = self.machines.selected_queued_job() {
                    self.popup = Some(Popup::Confirm {
                        message: format!("Cancel queued job for {}?", job.model),
                        on_confirm: ConfirmAction::CancelHostJob {
                            host_id,
                            job_id: job.id,
                        },
                    });
                }
            }
            Action::MachinesNextDevice if self.active_view == View::Machines => {
                self.machines.select_next_device();
            }
            Action::MachinesToggleDevice
                if self.active_view == View::Machines
                    && self.machines.focus == crate::hosts::MachinesFocus::Detail =>
            {
                let Some(device) = self.machines.selected_device().cloned() else {
                    return;
                };
                if device.admin_state == mold_core::DeviceAdminState::StartupExcluded {
                    self.generate.error_message =
                        Some("This GPU was excluded at startup and requires a restart".to_string());
                    return;
                }
                if device.restart_required {
                    return;
                }
                if !self.machines.can_mutate_selected_device() {
                    self.generate.error_message = Some(
                        "Live GPU controls require Scheduler V2; only a disabled GPU can be enabled for the next restart"
                            .to_string(),
                    );
                    return;
                }
                let enabled = !device.desired_enabled;
                let tx = self.bg_tx.clone();
                match self.machines.selected_row() {
                    crate::hosts::MachineRowId::Local => {
                        if let Some(url) = self.server_url.clone() {
                            self.tokio_handle
                                .spawn(crate::hosts::set_local_device_enabled(
                                    url, device.id, enabled, tx,
                                ));
                        }
                    }
                    crate::hosts::MachineRowId::Host(id) => {
                        if let Some(entry) = self.machines.registry.get(&id).cloned() {
                            self.tokio_handle
                                .spawn(crate::hosts::set_host_device_enabled(
                                    entry, device.id, enabled, tx,
                                ));
                        }
                    }
                }
            }
            Action::ScriptMoveDown => self.script.move_down(),
            Action::ScriptMoveUp => self.script.move_up(),
            Action::ScriptReorderDown => self.script.reorder_down(),
            Action::ScriptReorderUp => self.script.reorder_up(),
            Action::ScriptAddAfter => self.script.add_stage_after(),
            Action::ScriptAddBefore => self.script.add_stage_before(),
            Action::ScriptDelete if self.script.script.stages.len() > 1 => {
                self.request_confirm(
                    format!("Delete stage {}?", self.script.selected + 1),
                    ConfirmAction::DeleteScriptStage,
                );
            }
            Action::ScriptCycleTransition => self.script.cycle_transition(),
            Action::ScriptSave => self.script.open_save_dialog(),
            Action::ScriptLoad => self.script.open_load_dialog(),
            Action::ScriptSubmit if !self.generate.generating => {
                let req = self.script.build_chain_request();
                self.generate.generating = true;
                self.generate.error_message = None;
                self.generate.progress.clear();
                self.generate.progress.mark_generation_start();
                self.generate.preview_image = None;
                self.generate.image_state = None;
                self.generate.animation = None;

                let tx = self.bg_tx.clone();
                let server_url = self.server_url.clone();

                self.tokio_handle.spawn(async move {
                    crate::backend::run_chain_generation(server_url, req, tx).await;
                });
            }
            Action::ScriptOpenPromptEditor => self.script.open_prompt_editor(),
            Action::ScriptOpenFramesEditor => self.script.open_frames_editor(),
            Action::ScriptModalSubmit => {
                use crate::ui::script_composer::ScriptModal;
                match self.script.modal {
                    ScriptModal::PromptEdit { .. } => self.script.commit_prompt(),
                    ScriptModal::FramesEdit { .. } => self.script.commit_frames(),
                    ScriptModal::SavePath { .. } => self.script.save_to_path(),
                    ScriptModal::LoadPath { .. } => self.script.load_from_path(),
                    ScriptModal::Closed => {}
                }
            }
            Action::ScriptModalCancel => self.script.cancel_modal(),
            Action::ScriptModalChar(c) => {
                use crate::ui::script_composer::ScriptModal;
                match &mut self.script.modal {
                    ScriptModal::PromptEdit { buffer } => buffer.push(c),
                    ScriptModal::FramesEdit { buffer, error }
                    | ScriptModal::SavePath { buffer, error }
                    | ScriptModal::LoadPath { buffer, error } => {
                        buffer.push(c);
                        *error = None;
                    }
                    ScriptModal::Closed => {}
                }
            }
            Action::ScriptModalBackspace => {
                use crate::ui::script_composer::ScriptModal;
                match &mut self.script.modal {
                    ScriptModal::PromptEdit { buffer } => {
                        buffer.pop();
                    }
                    ScriptModal::FramesEdit { buffer, error }
                    | ScriptModal::SavePath { buffer, error }
                    | ScriptModal::LoadPath { buffer, error } => {
                        buffer.pop();
                        *error = None;
                    }
                    ScriptModal::Closed => {}
                }
            }
            Action::ScriptModalNewline => {
                use crate::ui::script_composer::ScriptModal;
                if let ScriptModal::PromptEdit { buffer } = &mut self.script.modal {
                    buffer.push('\n');
                }
            }
            _ => {}
        }
    }

    fn increment_param(&mut self, delta: i32) {
        use crate::ui::create_form::CreateRow;
        if self.active_view != View::Create || self.generate.focus != GenerateFocus::Parameters {
            return;
        }
        let row = match self.generate.rows.get(self.generate.param_index) {
            Some(r) => *r,
            None => return,
        };
        match row {
            CreateRow::Field(field) | CreateRow::SectionField(_, field) => {
                self.adjust_field(field, delta);
            }
            // ◀▶ on the disclosure/section rows mirrors Enter: → opens,
            // ← closes.
            CreateRow::AdvancedHeader => {
                if (delta > 0) != self.generate.advanced.open {
                    self.dispatch_action(Action::ToggleAdvanced);
                }
            }
            CreateRow::Section(sec) => {
                if delta > 0 {
                    self.set_advanced_expanded(Some(sec));
                } else if self.generate.advanced.expanded == Some(sec) {
                    self.set_advanced_expanded(None);
                }
            }
            CreateRow::NegativeEditor | CreateRow::ResetDefaults => {}
        }
    }

    /// Apply a `+`/`-`/`◀▶` adjustment to one parameter field.
    fn adjust_field(&mut self, field: ParamField, delta: i32) {
        // Size presets need the model's default area before we borrow
        // params mutably.
        let default_size = if field == ParamField::Size {
            Some(self.model_default_size())
        } else {
            None
        };
        let p = &mut self.generate.params;
        match field {
            ParamField::Size => {
                // Cycle the aspect presets fitted to the model's default
                // pixel area; a custom size snaps to the first preset.
                let (dw, dh) = default_size.unwrap_or((p.width, p.height));
                let presets = crate::ui::create_form::size_presets(dw, dh, 64);
                if presets.is_empty() {
                    return;
                }
                let len = presets.len() as i32;
                let next = match presets.iter().position(|&s| s == (p.width, p.height)) {
                    Some(i) => (i as i32 + delta).rem_euclid(len) as usize,
                    None => 0,
                };
                (p.width, p.height) = presets[next];
            }
            ParamField::Steps => {
                p.steps = (p.steps as i32 + delta).clamp(1, 200) as u32;
            }
            ParamField::Guidance => {
                p.guidance = (p.guidance + delta as f64 * 0.5).clamp(0.0, 30.0);
            }
            ParamField::Seed => {
                // ◀▶ cycles the seed mode (random → fixed → increment).
                p.seed_mode = if delta >= 0 {
                    p.seed_mode.next()
                } else {
                    p.seed_mode.next().next()
                };
                if p.seed_mode == SeedMode::Fixed && p.seed.is_none() {
                    p.seed = Some(rand::thread_rng().gen_range(0..u64::MAX));
                }
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
            ParamField::Expand => {
                p.expand = !p.expand;
            }
            ParamField::Offload => {
                p.offload = !p.offload;
            }
            ParamField::Upscale => {
                // ◀▶ clears the post-generate upscaler; Enter picks one.
                p.upscale_model = None;
            }
            ParamField::Model
            | ParamField::Scheduler
            | ParamField::Lora
            | ParamField::SourceImage
            | ParamField::MaskImage
            | ParamField::ControlImage
            | ParamField::ControlModel => {}
        }
    }

    /// Load the currently selected gallery entry's image into the preview.
    fn load_gallery_preview(&mut self) {
        if let Some(entry) = self.gallery.entries.get(self.gallery.selected) {
            let origin = entry.primary_origin();
            if let Some(url) = origin.url.clone() {
                // Server-backed: check cache first, then fetch async.
                // Cache paths and clients are scoped to the OWNING host.
                let host_id = origin.host_id.clone();
                let filename = entry.filename();
                let is_video = crate::gallery_scan::is_video_filename(&filename);

                // For video entries, prefer the cached animated GIF preview
                // so the detail pane animates instead of sitting on a frozen
                // first-frame thumbnail. When the preview isn't locally
                // cached we fetch `/api/gallery/preview/:filename` before
                // falling back to the raw MP4.
                if is_video {
                    let preview_cache =
                        crate::gallery_scan::preview_cache_path(&host_id, &filename);
                    if preview_cache.is_file() && self.try_install_gallery_animation(&preview_cache)
                    {
                        return;
                    }
                    let tx = self.bg_tx.clone();
                    let fetch_url = url.clone();
                    let fetch_name = filename.clone();
                    self.tokio_handle.spawn(async move {
                        if let Some(data) = crate::gallery_scan::fetch_and_cache_preview(
                            &fetch_url,
                            &host_id,
                            &fetch_name,
                        )
                        .await
                        {
                            let _ = tx.send(BackgroundEvent::GalleryPreviewReady(data));
                            return;
                        }
                        // No preview GIF on the server (older server or the
                        // video was generated without gif_preview). Fall back
                        // to the PNG thumbnail — the thumbnail endpoint runs
                        // openh264 first-frame extraction for MP4s, so the
                        // image pipeline can decode it. Sending the raw MP4
                        // bytes here would leave the pane blank because
                        // `image::load_from_memory` can't parse them.
                        let api_key = crate::hosts::api_key_for(&host_id);
                        let client = crate::hosts::client_for(&fetch_url, api_key.as_deref());
                        if let Ok(thumb) = client.get_gallery_thumbnail(&fetch_name).await {
                            let _ = tx.send(BackgroundEvent::GalleryPreviewReady(thumb));
                        }
                    });
                    self.gallery.preview_image = None;
                    self.gallery.image_state = None;
                    self.gallery.animation = None;
                    return;
                }

                let cache_path = crate::gallery_scan::cached_image_path(&host_id, &filename);
                if cache_path.is_file() {
                    // Cached locally — load synchronously
                    if self.try_install_gallery_animation(&cache_path) {
                        return;
                    }
                    if let Ok(img) = image::open(&cache_path) {
                        let protocol = self.picker.new_resize_protocol(img.clone());
                        self.gallery.preview_image = Some(img);
                        self.gallery.image_state = Some(protocol);
                        self.gallery.animation = None;
                        return;
                    }
                }
                // Not cached — fetch asynchronously
                let tx = self.bg_tx.clone();
                self.tokio_handle.spawn(async move {
                    if let Some(cached) =
                        crate::gallery_scan::fetch_and_cache_image(&url, &host_id, &filename).await
                    {
                        let data = tokio::fs::read(&cached).await.unwrap_or_default();
                        let _ = tx.send(BackgroundEvent::GalleryPreviewReady(data));
                    }
                });
                self.gallery.preview_image = None;
                self.gallery.image_state = None;
                self.gallery.animation = None;
            } else if entry.path.exists() && entry.path.is_file() {
                // For video files, prefer the cached GIF preview (animated)
                let gif_path = crate::thumbnails::preview_gif_path(&entry.path);
                let load_path = if gif_path.is_file() {
                    gif_path.clone()
                } else {
                    entry.path.clone()
                };
                if self.try_install_gallery_animation(&load_path) {
                    return;
                }
                if let Ok(img) = image::open(&load_path) {
                    let protocol = self.picker.new_resize_protocol(img.clone());
                    self.gallery.preview_image = Some(img);
                    self.gallery.image_state = Some(protocol);
                    self.gallery.animation = None;
                    return;
                }
            }
        }
        self.gallery.preview_image = None;
        self.gallery.image_state = None;
        self.gallery.animation = None;
    }

    /// Try to decode `path` as an animated container and install it as the
    /// active gallery preview. Returns `true` when animation was installed.
    fn try_install_gallery_animation(&mut self, path: &std::path::Path) -> bool {
        let frames = match crate::animation::decode_animation_path(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let state = match crate::animation::AnimationState::new(frames) {
            Some(s) => s,
            None => return false,
        };
        let first = state.current_image().clone();
        let protocol = self.picker.new_resize_protocol(first.clone());
        self.gallery.preview_image = Some(first);
        self.gallery.image_state = Some(protocol);
        self.gallery.animation = Some(state);
        true
    }

    /// Advance any active animations in the gallery/generate previews and
    /// rebuild their image protocols so the next render shows the new
    /// frame. Called once per event-loop tick.
    pub fn tick_animations(&mut self) {
        if let Some(anim) = self.gallery.animation.as_mut() {
            if anim.tick() {
                let img = anim.current_image().clone();
                self.gallery.preview_image = Some(img.clone());
                self.gallery.image_state = Some(self.picker.new_resize_protocol(img));
            }
        }
        if let Some(anim) = self.generate.animation.as_mut() {
            if anim.tick() {
                let img = anim.current_image().clone();
                self.generate.preview_image = Some(img.clone());
                self.generate.image_state = Some(self.picker.new_resize_protocol(img));
            }
        }
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
        self.active_view = View::Create;
        self.generate.focus = GenerateFocus::Prompt;
    }

    /// Replace the gallery with a fresh scan result, preserving the user's
    /// current selection by filename where possible. When the previously-
    /// selected entry is still in the new list, `selected` points at its
    /// new index; otherwise we clamp the old index against the new length
    /// (falling back to 0 only when the list is empty). This keeps the
    /// viewport stable across deletes, reconnects, and any other rescan
    /// trigger — no more "back to the first image" on every refresh.
    pub fn apply_gallery_scan(&mut self, entries: Vec<GalleryEntry>) {
        let previous_selected = self.gallery.selected;
        let previous_filename = self
            .gallery
            .entries
            .get(previous_selected)
            .map(|e| e.filename());

        self.gallery.thumbnail_states = vec![None; entries.len()];
        self.gallery.thumb_dimensions = vec![None; entries.len()];
        self.gallery.thumb_fixed_cache = vec![None; entries.len()];
        self.gallery.details_thumb = None;
        self.gallery.entries = entries;
        self.gallery.scanning = false;
        self.gallery.last_scan = Some(std::time::Instant::now());
        self.gallery.dirty = false;

        self.gallery.selected = if self.gallery.entries.is_empty() {
            0
        } else if let Some(idx) = previous_filename.as_deref().and_then(|name| {
            self.gallery
                .entries
                .iter()
                .position(|e| e.filename() == name)
        }) {
            idx
        } else {
            previous_selected.min(self.gallery.entries.len() - 1)
        };
        self.gallery.refresh_filter();
    }

    /// React to a failed server-side gallery delete: surface the server's
    /// error to the user and — when we still have a live server
    /// connection — kick off a rescan so the local gallery reconverges
    /// with the server's authoritative list. The tile was already
    /// optimistically removed from `self.gallery.entries` by the earlier
    /// `delete_selected_gallery_image()` call, so the rescan puts it back
    /// if the server never actually deleted it.
    pub fn apply_delete_failure(&mut self, err: &str) {
        self.generate.error_message = Some(format!("Delete failed: {err}"));
        if self.server_url.is_some() {
            self.gallery.scanning = true;
            self.spawn_gallery_scan();
        }
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
        let filename = entry.filename();
        let thumb_path = crate::thumbnails::thumbnail_path(&entry.path);
        let _ = std::fs::remove_file(&thumb_path);
        // Also remove the legacy (pre-host-scoping) cached copy.
        let cache_path = crate::gallery_scan::image_cache_dir().join(&filename);
        let _ = std::fs::remove_file(&cache_path);

        // Delete on every machine the print exists on. Remote deletes go
        // through the OWNING host's client (with its saved API key) and
        // propagate errors through the background channel so the UI can
        // surface them and rescan — a silent fire-and-forget would mask
        // transient network errors and leave the deleted tile "gone"
        // locally while a server still holds the file.
        for origin in entry.owning_origins() {
            let Some(url) = origin.url else { continue };
            let _ = std::fs::remove_file(crate::gallery_scan::cached_image_path(
                &origin.host_id,
                &filename,
            ));
            let _ = std::fs::remove_file(crate::gallery_scan::preview_cache_path(
                &origin.host_id,
                &filename,
            ));
            let host_id = origin.host_id;
            let filename = filename.clone();
            let tx = self.bg_tx.clone();
            self.tokio_handle.spawn(async move {
                let api_key = crate::hosts::api_key_for(&host_id);
                let client = crate::hosts::client_for(&url, api_key.as_deref());
                if let Err(e) = client.delete_gallery_image(&filename).await {
                    let _ = tx.send(BackgroundEvent::GalleryDeleteFailed(e.to_string()));
                }
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
        self.gallery.details_thumb = None;

        // Drop the deleted image's preview state first — load_gallery_preview
        // below will repopulate these for the new selection. Doing it in the
        // other order (load → wipe) is the bug that left Detail view blank
        // after a delete: we'd read the new image off disk and then
        // immediately throw it away.
        self.gallery.preview_image = None;
        self.gallery.image_state = None;
        self.gallery.animation = None;

        // Adjust selection — keep the user on the next neighbour, or clamp
        // to the new last entry when they were already at the end.
        if !self.gallery.entries.is_empty() {
            self.gallery.selected = idx.min(self.gallery.entries.len() - 1);
            self.gallery.refresh_filter();
            if self.gallery.view_mode == GalleryViewMode::Detail {
                self.load_gallery_preview();
            }
        } else {
            self.gallery.selected = 0;
            self.gallery.view_mode = GalleryViewMode::Grid;
            self.gallery.refresh_filter();
        }
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
        } else if let Some(url) = entry.primary_origin().url {
            // Server-backed — fetch to the host-scoped cache, then open
            let host_id = entry.primary_origin().host_id;
            let filename = entry.filename();
            self.tokio_handle.spawn(async move {
                if let Some(cached) =
                    crate::gallery_scan::fetch_and_cache_image(&url, &host_id, &filename).await
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

    /// Whether destructive actions must ask before dispatching —
    /// the `tui.confirm_destructive` preference.
    pub fn needs_confirm(&self) -> bool {
        self.prefs.confirm_destructive
    }

    /// Gate a destructive action behind the Confirm popup. Every
    /// `ConfirmAction` site must route through here (never build
    /// `Popup::Confirm` directly) so `tui.confirm_destructive = off`
    /// dispatches immediately — and future actions get the gate for free.
    pub fn request_confirm(&mut self, message: String, on_confirm: ConfirmAction) {
        if self.needs_confirm() {
            self.popup = Some(Popup::Confirm {
                message,
                on_confirm,
            });
        } else {
            self.handle_confirm_action(on_confirm);
        }
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
            ConfirmAction::DeleteScriptStage => {
                self.script.delete_stage();
            }
            ConfirmAction::ForgetHost(id) => {
                self.machines.forget(&id);
                // The registry changed — the next Library visit re-merges
                // galleries without the forgotten host.
                self.gallery.dirty = true;
                if self.target == crate::hosts::GenTarget::Host(id) {
                    self.target = crate::hosts::GenTarget::Auto;
                    self.target.save();
                }
            }
            ConfirmAction::CancelHostJob { host_id, job_id } => {
                if let Some(entry) = self.machines.registry.get(&host_id).cloned() {
                    self.tokio_handle.spawn(crate::hosts::cancel_host_job(
                        entry,
                        job_id,
                        self.bg_tx.clone(),
                    ));
                }
            }
        }
    }

    /// Kick a queue fetch for the newly selected Machines row (no-op for
    /// the local row — its lane is composed from in-process state).
    fn spawn_selected_host_queue_fetch(&self) {
        if let Some(entry) = self.machines.selected_host().cloned() {
            self.spawn_host_queue_fetch(entry);
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

    /// Handle Enter on the currently selected Create-form row.
    fn activate_current_param(&mut self) {
        use crate::ui::create_form::CreateRow;
        let row = match self.generate.rows.get(self.generate.param_index) {
            Some(r) => *r,
            None => return,
        };
        match row {
            CreateRow::Field(field) | CreateRow::SectionField(_, field) => {
                self.activate_field(field)
            }
            CreateRow::AdvancedHeader => self.dispatch_action(Action::ToggleAdvanced),
            CreateRow::Section(sec) => {
                if self.generate.advanced.expanded == Some(sec) {
                    self.set_advanced_expanded(None);
                } else {
                    self.set_advanced_expanded(Some(sec));
                }
            }
            CreateRow::NegativeEditor => {
                self.generate.focus = GenerateFocus::NegativePrompt;
            }
            // Reset all params to model defaults (keep model and prompt)
            CreateRow::ResetDefaults => self.reset_params_to_model_defaults(),
        }
    }

    /// Handle Enter on one parameter field.
    fn activate_field(&mut self, field: ParamField) {
        match field {
            // Open model selector popup
            ParamField::Model => self.open_model_selector(),
            // Free-text WxH entry
            ParamField::Size => {
                let input = format!(
                    "{}x{}",
                    self.generate.params.width, self.generate.params.height
                );
                self.popup = Some(Popup::SizeInput { input });
            }
            // Toggle boolean fields
            ParamField::Expand => self.generate.params.expand = !self.generate.params.expand,
            ParamField::Offload => self.generate.params.offload = !self.generate.params.offload,
            // Cycle format
            ParamField::Format => self.adjust_field(ParamField::Format, 1),
            // Cycle scheduler
            ParamField::Scheduler => {
                self.generate.params.scheduler = match self.generate.params.scheduler {
                    None => Some(Scheduler::Ddim),
                    Some(Scheduler::Ddim) => Some(Scheduler::EulerAncestral),
                    Some(Scheduler::EulerAncestral) => Some(Scheduler::UniPc),
                    Some(Scheduler::UniPc) => None,
                };
            }
            // Enter on the merged Seed row edits the value (◀▶ cycles mode)
            ParamField::Seed => {
                let current = self
                    .generate
                    .params
                    .seed
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                self.popup = Some(Popup::SeedInput { input: current });
            }
            // Pick (or clear) the post-generate upscaler
            ParamField::Upscale => {
                let mut filtered = vec![UPSCALE_OFF_ENTRY.to_string()];
                filtered.extend(self.available_upscaler_models());
                self.popup = Some(Popup::UpscaleModelSelector {
                    filter: String::new(),
                    selected: 0,
                    filtered,
                    purpose: UpscalePickerPurpose::SetGenerateParam,
                });
            }
            // For the remaining fields, Enter does nothing special (use +/-)
            _ => {}
        }
    }

    /// Restore the selected model's defaults (keeps model and prompt).
    fn reset_params_to_model_defaults(&mut self) {
        let model = self.generate.params.model.clone();

        // Use server catalog defaults when connected (and not in local mode),
        // local config otherwise
        if let Some(entry) = if self.should_poll_remote() {
            self.models.catalog.iter().find(|m| m.name == model)
        } else {
            None
        } {
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
        self.generate.params.upscale_model = None;
        self.generate.params.frames = 25;
        self.generate.params.fps = 24;
        self.generate.params.strength = 0.75;
        self.generate.params.source_image_path = None;
        self.generate.params.mask_image_path = None;
        self.generate.params.control_image_path = None;
        self.generate.params.control_model = None;
        self.generate.params.control_scale = 1.0;
    }

    // ── Settings view helpers ─────────────────────────────────────────

    /// Build the flat list of settings rows from current config state.
    #[allow(clippy::vec_init_then_push)]
    pub fn build_settings_rows(&self) -> Vec<SettingsRow> {
        let mut rows = Vec::new();

        // ── Preferences (DB-backed, top of the list per the mock's
        //    APPEARANCE / PREFERENCES stacked sections) ────────────
        rows.push(SettingsRow::SectionHeader {
            name: "Preferences".into(),
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::PrefDefaultFormat,
            label: "Format",
            field_type: SettingsFieldType::Toggle {
                options: vec!["png", "jpeg"],
            },
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::PrefReduceMotion,
            label: "Reduce Motion",
            field_type: SettingsFieldType::Bool,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::PrefShowTimeline,
            label: "Show Timeline",
            field_type: SettingsFieldType::Bool,
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::PrefConfirmDestructive,
            label: "Confirmations",
            field_type: SettingsFieldType::Bool,
        });

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
            // Preferences (DB-backed)
            SettingsKey::PrefDefaultFormat => self.prefs.default_format_slug().to_string(),
            SettingsKey::PrefReduceMotion => if self.prefs.reduce_motion {
                "on"
            } else {
                "off"
            }
            .into(),
            SettingsKey::PrefShowTimeline => if self.prefs.show_timeline {
                "on"
            } else {
                "off"
            }
            .into(),
            SettingsKey::PrefConfirmDestructive => if self.prefs.confirm_destructive {
                "on"
            } else {
                "off"
            }
            .into(),
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
    ///
    /// When focus is on the Appearance card grid, Up/Down move the theme
    /// selection by one visual row (live-applying, like Left/Right's
    /// linear cycle): Down below the bottom row hands focus to the
    /// Configuration list, Up above the top row is a no-op. When focus is
    /// on Configuration and Up is pressed at the first field, focus
    /// returns to Appearance.
    fn settings_navigate(&mut self, delta: i32) {
        if self.settings.focus == SettingsFocus::Appearance {
            use crate::ui::theme::ThemePreset;
            let cols = self.settings.appearance_cols.max(1);
            let idx = ThemePreset::ALL
                .iter()
                .position(|p| *p == self.settings.theme_preset)
                .unwrap_or(0);
            if delta > 0 {
                let below = idx + cols;
                if below < ThemePreset::ALL.len() {
                    self.apply_theme_preset(ThemePreset::ALL[below]);
                } else {
                    // Walked off the bottom of the card grid → enter the
                    // Configuration list.
                    self.settings.focus = SettingsFocus::Configuration;
                }
            } else if idx >= cols {
                self.apply_theme_preset(ThemePreset::ALL[idx - cols]);
            }
            return;
        }

        let rows = self.build_settings_rows();
        if rows.is_empty() {
            return;
        }
        let len = rows.len();
        let mut next = self.settings.row_index;
        loop {
            let candidate = next as i32 + delta;
            if candidate < 0 || candidate >= len as i32 {
                // Walked off the top of the list → hand focus back to Appearance.
                if delta < 0 {
                    self.settings.focus = SettingsFocus::Appearance;
                }
                break;
            }
            next = candidate as usize;
            if rows[next].is_field() {
                self.settings.row_index = next;
                break;
            }
        }
    }

    /// Cycle the active theme preset by `delta` (wraps around).
    fn settings_cycle_theme(&mut self, delta: i32) {
        use crate::ui::theme::ThemePreset;
        let current = self.settings.theme_preset;
        let len = ThemePreset::ALL.len() as i32;
        let current_idx = ThemePreset::ALL
            .iter()
            .position(|p| *p == current)
            .unwrap_or(0) as i32;
        let next_idx = ((current_idx + delta).rem_euclid(len)) as usize;
        self.apply_theme_preset(ThemePreset::ALL[next_idx]);
    }

    /// Adjust the current settings field by delta (+1 or -1).
    fn settings_increment(&mut self, delta: i32) {
        if self.settings.focus == SettingsFocus::Appearance {
            self.settings_cycle_theme(delta);
            return;
        }

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
            SettingsFieldType::Toggle { options } if !options.is_empty() => {
                self.settings_cycle_toggle(key, &options, delta);
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
            SettingsKey::PrefDefaultFormat => {
                // DB-backed preference — persist immediately, no
                // config.toml write.
                self.prefs.default_format = if value == "jpeg" {
                    OutputFormat::Jpeg
                } else {
                    OutputFormat::Png
                };
                self.prefs.save_key(mold_db::settings::TUI_DEFAULT_FORMAT);
                return;
            }
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
        // Preference bools live in the DB, not config.toml — flip and
        // persist immediately (like theme changes), skipping save_config.
        match key {
            SettingsKey::PrefReduceMotion => {
                self.prefs.reduce_motion = !self.prefs.reduce_motion;
                self.prefs.save_key(mold_db::settings::TUI_REDUCE_MOTION);
                return;
            }
            SettingsKey::PrefShowTimeline => {
                self.prefs.show_timeline = !self.prefs.show_timeline;
                self.prefs.save_key(mold_db::settings::TUI_SHOW_TIMELINE);
                return;
            }
            SettingsKey::PrefConfirmDestructive => {
                self.prefs.confirm_destructive = !self.prefs.confirm_destructive;
                self.prefs
                    .save_key(mold_db::settings::TUI_CONFIRM_DESTRUCTIVE);
                return;
            }
            _ => {}
        }
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

        // Route by the sticky Machines generation target. Auto keeps
        // today's exact behavior; Local forces the in-process engine; a
        // Host target pins the run to that registry entry with its API
        // key and no silent fallback. Routing runs before any state
        // mutation so a stale target aborts cleanly.
        use crate::hosts::GenTarget;
        let mut route_mode = None;
        let mut route_host = None;
        let mut route_name = None;
        let mut api_key = None;
        match &self.target {
            GenTarget::Auto => {}
            GenTarget::Local => route_mode = Some(InferenceMode::Local),
            GenTarget::Host(id) => match self.machines.registry.get(id) {
                Some(entry) => {
                    route_mode = Some(InferenceMode::Remote);
                    route_host = Some(entry.url.clone());
                    route_name = Some(entry.display_name());
                    api_key = crate::hosts::api_key_for(id);
                }
                None => {
                    self.generate.error_message = Some(format!(
                        "Machine '{id}' is no longer saved. Pick a target in Machines (4)."
                    ));
                    return;
                }
            },
        }

        self.generate.generating = true;
        self.generate.batch_remaining = self.generate.params.batch;
        self.generate.error_message = None;
        self.generate.progress.clear();
        self.generate.progress.mark_generation_start();
        self.generate.preview_image = None;
        self.generate.image_state = None;
        self.generate.animation = None;

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
        if let Some(mode) = route_mode {
            params.inference_mode = mode;
        }
        if route_host.is_some() {
            params.host = route_host;
        }
        params.target_host_name = route_name;

        let tx = self.bg_tx.clone();
        let server_url = self.server_url.clone();

        self.tokio_handle.spawn(async move {
            crate::backend::run_generation(
                server_url,
                params,
                prompt_text,
                negative_prompt,
                api_key,
                tx,
            )
            .await;
        });
    }

    /// Drain and process all pending background events.
    pub fn process_background_events(&mut self) {
        while let Ok(event) = self.bg_rx.try_recv() {
            match event {
                BackgroundEvent::Progress(sse) => self.handle_progress(sse),
                BackgroundEvent::GenerationComplete {
                    response,
                    from_local,
                } => {
                    self.generate.batch_remaining = self.generate.batch_remaining.saturating_sub(1);
                    if self.generate.batch_remaining == 0 {
                        self.generate.generating = false;
                        // Stop the Overall heartbeat row now that the
                        // pipeline has produced a result.
                        self.generate.progress.generation_started_at = None;
                        self.generate.progress.stage_started_at = None;
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

                    // Resolve output directory. Returns None when output is
                    // explicitly disabled *or* when the TUI is connected to
                    // a remote server — the server already saved the file
                    // to its own output dir, and a TUI-side write would
                    // duplicate it (with a different timestamp suffix) and
                    // surface as two tiles on the next gallery scan.
                    // The `from_local` override handles Auto-mode fallbacks
                    // — we still want to save those locally even though
                    // `server_url` is set.
                    let output_dir = if self.should_persist_response_locally(from_local) {
                        let dir = self.config.effective_output_dir();
                        let _ = std::fs::create_dir_all(&dir);
                        Some(dir)
                    } else {
                        None
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
                                self.generate.animation = None;
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
                            if let Ok(frames) = crate::animation::decode_animation_bytes(
                                &video.gif_preview,
                                Some("gif"),
                            ) {
                                if let Some(state) = crate::animation::AnimationState::new(frames) {
                                    let first = state.current_image().clone();
                                    let protocol = self.picker.new_resize_protocol(first.clone());
                                    self.generate.preview_image = Some(first);
                                    self.generate.image_state = Some(protocol);
                                    self.generate.animation = Some(state);
                                } else if let Ok(img) = image::load_from_memory(&video.gif_preview)
                                {
                                    let protocol = self.picker.new_resize_protocol(img.clone());
                                    self.generate.preview_image = Some(img);
                                    self.generate.image_state = Some(protocol);
                                    self.generate.animation = None;
                                }
                            } else if let Ok(img) = image::load_from_memory(&video.gif_preview) {
                                let protocol = self.picker.new_resize_protocol(img.clone());
                                self.generate.preview_image = Some(img);
                                self.generate.image_state = Some(protocol);
                                self.generate.animation = None;
                            }
                        }
                    }

                    let saved_name = saved_path
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_default();

                    // Feed the activity strip's "done · saved to …" state.
                    self.generate.last_output_path = if saved_name.is_empty() {
                        None
                    } else {
                        Some(saved_path.clone())
                    };

                    // Sweep the caption row in under the finished print —
                    // caption only, never the image cells (graphics
                    // protocols are escape-sequence passthrough).
                    if self.generate.batch_remaining == 0 {
                        let p = self.layout.preview;
                        if p.height > 0 {
                            let caption = ratatui::layout::Rect {
                                y: p.y + p.height - 1,
                                height: 1,
                                ..p
                            };
                            self.motion.trigger_completion_sweep(caption, self.theme.bg);
                        }
                    }

                    self.generate.progress.push_log(ProgressLogEntry {
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
                    mold_db::settings::record_last_model(&actual_model);

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
                    // In remote-server mode we don't write a local copy, so
                    // `saved_path` is empty and there's nothing for the
                    // gallery to point at. Kick off a gallery rescan
                    // instead — the server's own save will surface on the
                    // next poll. In local mode this branch is skipped and
                    // the `insert(0)` below runs as before.
                    // Only kick off a server rescan when this response
                    // actually came from the server. An Auto-mode local
                    // fallback would still pass `should_poll_remote()`
                    // (server_url is set) but there is nothing new on
                    // the server to scan — and the scan would wipe the
                    // local gallery entry we just inserted.
                    if saved_path.as_os_str().is_empty() && self.should_poll_remote() && !from_local
                    {
                        self.gallery.scanning = true;
                        self.spawn_gallery_scan();
                    }

                    if (!response.images.is_empty() || response.video.is_some())
                        && !saved_path.as_os_str().is_empty()
                    {
                        let meta = mold_core::OutputMetadata {
                            prompt: prompt_text,
                            negative_prompt: if neg_text.is_empty() {
                                None
                            } else {
                                Some(neg_text)
                            },
                            original_prompt: None,
                            batch_id: None,
                            batch_index: None,
                            batch_count: None,
                            model: actual_model,
                            seed: response.seed_used,
                            steps: self.generate.params.steps,
                            guidance: self.generate.params.guidance,
                            width: entry_width,
                            height: entry_height,
                            generation_width: Some(entry_width),
                            generation_height: Some(entry_height),
                            strength: if self.generate.params.source_image_path.is_some() {
                                Some(self.generate.params.strength)
                            } else {
                                None
                            },
                            source_image_name: None,
                            source_image_sha256: None,
                            scheduler: self.generate.params.scheduler,
                            output_format: Some(self.generate.params.format),
                            cfg_plus: None,
                            lora: self.generate.params.lora_path.clone(),
                            lora_scale: self
                                .generate
                                .params
                                .lora_path
                                .as_ref()
                                .map(|_| self.generate.params.lora_scale),
                            loras: self.generate.params.lora_path.as_ref().map(|path| {
                                vec![mold_core::LoraWeight {
                                    path: path.clone(),
                                    scale: self.generate.params.lora_scale,
                                }]
                            }),
                            control_model: self.generate.params.control_model.clone(),
                            control_scale: self
                                .generate
                                .params
                                .control_image_path
                                .as_ref()
                                .and(self.generate.params.control_model.as_ref())
                                .map(|_| self.generate.params.control_scale),
                            upscale_model: None,
                            gif_preview: response.video.as_ref().map(|_| true),
                            enable_audio: None,
                            audio_file_path: None,
                            source_video_path: None,
                            pipeline: None,
                            retake_range: None,
                            spatial_upscale: None,
                            temporal_upscale: None,
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
                                origins: vec![GalleryOrigin::local()],
                            },
                        );
                        self.gallery.thumbnail_states.insert(0, None);
                        self.gallery.thumb_dimensions.insert(0, None);
                        self.gallery.thumb_fixed_cache.insert(0, None);
                        self.gallery.refresh_filter();

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
                    self.generate.progress.generation_started_at = None;
                    self.generate.progress.stage_started_at = None;
                }
                BackgroundEvent::GalleryScanComplete(scan) => {
                    self.gallery.offline_hosts = scan.offline_hosts;
                    self.apply_gallery_scan(scan.entries);

                    // Spawn background thumbnail generation. Remote
                    // thumbnails are fetched from the entry's primary
                    // origin with that host's saved API key.
                    let entries_info: Vec<(std::path::PathBuf, Option<(String, String)>)> = self
                        .gallery
                        .entries
                        .iter()
                        .map(|e| {
                            let origin = e.primary_origin();
                            let remote = origin.url.map(|url| (url, origin.host_id));
                            (e.path.clone(), remote)
                        })
                        .collect();
                    let tx = self.bg_tx.clone();
                    self.tokio_handle.spawn(async move {
                        // Spawn all thumbnail fetches concurrently
                        let mut handles = Vec::new();
                        for (path, remote) in entries_info {
                            if crate::thumbnails::thumbnail_exists(&path) {
                                continue;
                            }
                            let handle = tokio::spawn(async move {
                                let filename = path
                                    .file_name()
                                    .map(|f| f.to_string_lossy().to_string())
                                    .unwrap_or_default();
                                if let Some((url, host_id)) = remote {
                                    // Fetch pre-generated thumbnail from server (fast, ~10KB)
                                    let api_key = crate::hosts::api_key_for(&host_id);
                                    let client = crate::hosts::client_for(&url, api_key.as_deref());
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
                                        let cache_path = crate::gallery_scan::cached_image_path(
                                            &host_id, &filename,
                                        );
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
                    let mut installed_animation = false;
                    if crate::animation::is_animated_bytes(&data) {
                        if let Ok(frames) = crate::animation::decode_animation_bytes(&data, None) {
                            if let Some(state) = crate::animation::AnimationState::new(frames) {
                                let first = state.current_image().clone();
                                let protocol = self.picker.new_resize_protocol(first.clone());
                                self.gallery.preview_image = Some(first);
                                self.gallery.image_state = Some(protocol);
                                self.gallery.animation = Some(state);
                                installed_animation = true;
                            }
                        }
                    }
                    if !installed_animation {
                        if let Ok(img) = image::load_from_memory(&data) {
                            let protocol = self.picker.new_resize_protocol(img.clone());
                            self.gallery.preview_image = Some(img);
                            self.gallery.image_state = Some(protocol);
                            self.gallery.animation = None;
                        }
                    }
                }
                BackgroundEvent::ThumbnailsReady => {
                    // Invalidate all thumbnail states so they reload on next render
                    let len = self.gallery.entries.len();
                    self.gallery.thumbnail_states = vec![None; len];
                    self.gallery.thumb_dimensions = vec![None; len];
                    self.gallery.thumb_fixed_cache = vec![None; len];
                    self.gallery.details_thumb = None;
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
                    // Apply model defaults from server catalog
                    self.apply_remote_model_defaults(&models);
                    self.generate.progress.push_log(ProgressLogEntry {
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
                    self.generate.progress.push_log(ProgressLogEntry {
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
                    self.generate.progress.push_log(ProgressLogEntry {
                        message: format!("Pull complete: {model}"),
                        style: ProgressStyle::Done,
                    });
                    // Refresh the catalog (from server when in remote mode, local otherwise).
                    // Don't reuse ServerConnected here — its handler auto-switches Local→Auto.
                    if self.should_poll_remote() {
                        let url = self.server_url.clone().unwrap();
                        let tx = self.bg_tx.clone();
                        self.tokio_handle.spawn(async move {
                            let client = mold_core::MoldClient::new(&url);
                            if let Ok(models) = client.list_models_extended().await {
                                // Update catalog directly without mode-switching side effects
                                let _ = tx.send(BackgroundEvent::CatalogRefreshed(models));
                            }
                        });
                    } else {
                        self.config = Config::load_or_default();
                        self.models.catalog =
                            mold_core::build_model_catalog(&self.config, None, false);
                    }
                }
                BackgroundEvent::ModelRemoveComplete(model) => {
                    self.generate.progress.push_log(ProgressLogEntry {
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
                    self.generate.progress.push_log(ProgressLogEntry {
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
                        self.generate.progress.push_log(ProgressLogEntry {
                            message: format!(
                                "Upscaled {original_width}x{original_height} -> {upscaled_w}x{upscaled_h} ({scale_factor}x, {:.1}s) — output dir disabled",
                                upscale_time_ms as f64 / 1000.0
                            ),
                            style: ProgressStyle::Warning,
                        });
                        return;
                    };

                    self.generate.progress.push_log(ProgressLogEntry {
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
                        batch_id: source_meta.as_ref().and_then(|m| m.batch_id.clone()),
                        batch_index: source_meta.as_ref().and_then(|m| m.batch_index),
                        batch_count: source_meta.as_ref().and_then(|m| m.batch_count),
                        model,
                        seed: source_meta.as_ref().map(|m| m.seed).unwrap_or(0),
                        steps: source_meta.as_ref().map(|m| m.steps).unwrap_or(0),
                        guidance: source_meta.as_ref().map(|m| m.guidance).unwrap_or(0.0),
                        width: upscaled_w,
                        height: upscaled_h,
                        generation_width: Some(
                            source_meta
                                .as_ref()
                                .map(|m| m.generation_width.unwrap_or(m.width))
                                .unwrap_or(original_width),
                        ),
                        generation_height: Some(
                            source_meta
                                .as_ref()
                                .map(|m| m.generation_height.unwrap_or(m.height))
                                .unwrap_or(original_height),
                        ),
                        strength: None,
                        source_image_name: None,
                        source_image_sha256: None,
                        scheduler: source_meta.as_ref().and_then(|m| m.scheduler),
                        output_format: Some(mold_core::OutputFormat::Png),
                        cfg_plus: source_meta.as_ref().and_then(|m| m.cfg_plus),
                        lora: source_meta.as_ref().and_then(|m| m.lora.clone()),
                        lora_scale: source_meta.as_ref().and_then(|m| m.lora_scale),
                        loras: source_meta.as_ref().and_then(|m| m.loras.clone()),
                        control_model: source_meta.as_ref().and_then(|m| m.control_model.clone()),
                        control_scale: source_meta.as_ref().and_then(|m| m.control_scale),
                        upscale_model: source_meta.as_ref().and_then(|m| m.upscale_model.clone()),
                        gif_preview: None,
                        enable_audio: source_meta.as_ref().and_then(|m| m.enable_audio),
                        audio_file_path: source_meta
                            .as_ref()
                            .and_then(|m| m.audio_file_path.clone()),
                        source_video_path: source_meta
                            .as_ref()
                            .and_then(|m| m.source_video_path.clone()),
                        pipeline: source_meta.as_ref().and_then(|m| m.pipeline),
                        retake_range: source_meta.as_ref().and_then(|m| m.retake_range.clone()),
                        spatial_upscale: source_meta.as_ref().and_then(|m| m.spatial_upscale),
                        temporal_upscale: source_meta.as_ref().and_then(|m| m.temporal_upscale),
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
                            origins: vec![GalleryOrigin::local()],
                        },
                    );
                    self.gallery.thumbnail_states.insert(0, None);
                    self.gallery.thumb_dimensions.insert(0, None);
                    self.gallery.thumb_fixed_cache.insert(0, None);
                    self.gallery.selected = 0;
                    self.gallery.refresh_filter();

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
                BackgroundEvent::HostStatusUpdate { host_id, status } => {
                    self.machines.apply_status(host_id, status);
                }
                BackgroundEvent::HostDevicesUpdate { host_id, devices } => {
                    self.machines.apply_devices(host_id, devices);
                }
                BackgroundEvent::HostDeviceMutationApplied { host_id, device } => {
                    self.machines.apply_device_mutation(host_id, *device);
                }
                BackgroundEvent::HostCapabilitiesUpdate {
                    host_id,
                    capabilities,
                } => {
                    self.machines.apply_capabilities(host_id, capabilities);
                }
                BackgroundEvent::HostQueueUpdate { host_id, queue } => {
                    self.machines.apply_queue(host_id, queue);
                }
                BackgroundEvent::MachineConnectTested {
                    url,
                    api_key,
                    result,
                } => {
                    use crate::hosts::{connect_advance, ConnectEffect, ConnectInput, ConnectStep};
                    // Only apply to the in-flight connect form for this URL —
                    // a closed or re-targeted popup drops the stale result.
                    let Some(Popup::MachineConnect { form }) = &mut self.popup else {
                        continue;
                    };
                    if form.step != ConnectStep::Testing || form.url != url {
                        continue;
                    }
                    match result {
                        Ok(status) => {
                            let effect = connect_advance(
                                form,
                                ConnectInput::TestOk {
                                    hostname: status.hostname.as_deref(),
                                },
                            );
                            if effect == ConnectEffect::Save {
                                match self.machines.complete_connect(
                                    &url,
                                    api_key.as_deref(),
                                    status,
                                ) {
                                    Ok(id) => {
                                        self.close_popup();
                                        // The registry changed — the next
                                        // Library visit re-merges galleries.
                                        self.gallery.dirty = true;
                                        if let Some(entry) =
                                            self.machines.registry.get(&id).cloned()
                                        {
                                            self.spawn_host_queue_fetch(entry);
                                        }
                                    }
                                    Err(crate::hosts::AddHostError::AlreadyKnown { name }) => {
                                        self.popup = Some(Popup::Info {
                                            message: format!("Already connected as {name}."),
                                        });
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            connect_advance(form, ConnectInput::TestErr(&e));
                        }
                    }
                }
                BackgroundEvent::CatalogRefreshed(models) => {
                    self.models.catalog = models;
                    if !self.models.catalog.is_empty()
                        && self.models.selected >= self.models.catalog.len()
                    {
                        self.models.selected = self.models.catalog.len() - 1;
                    }
                }
                BackgroundEvent::ChainProgress(event) => {
                    use mold_core::ChainProgressEvent;
                    let msg = match &event {
                        ChainProgressEvent::ChainStart {
                            stage_count,
                            estimated_total_frames,
                        } => {
                            format!("Chain: {stage_count} stages, ~{estimated_total_frames} frames")
                        }
                        ChainProgressEvent::StageStart { stage_idx } => {
                            format!(
                                "Stage {}/{} started",
                                stage_idx + 1,
                                self.script.script.stages.len()
                            )
                        }
                        ChainProgressEvent::DenoiseStep {
                            stage_idx,
                            step,
                            total,
                        } => {
                            format!("Stage {} step {}/{}", stage_idx + 1, step, total)
                        }
                        ChainProgressEvent::StageDone {
                            stage_idx,
                            frames_emitted,
                        } => {
                            format!("Stage {} done ({} frames)", stage_idx + 1, frames_emitted)
                        }
                        ChainProgressEvent::Stitching { total_frames } => {
                            format!("Stitching {total_frames} frames...")
                        }
                    };
                    self.generate.progress.push_log(ProgressLogEntry {
                        message: msg,
                        style: ProgressStyle::Info,
                    });
                }
                BackgroundEvent::ChainComplete { response } => {
                    self.generate.generating = false;
                    self.generate.progress.generation_started_at = None;
                    self.generate.progress.stage_started_at = None;
                    self.generate.progress.push_log(ProgressLogEntry {
                        message: format!(
                            "Chain complete: {} stages, GPU {}",
                            response.stage_count,
                            response
                                .gpu
                                .map(|g| g.to_string())
                                .unwrap_or_else(|| "unknown".into()),
                        ),
                        style: ProgressStyle::Done,
                    });
                }
                BackgroundEvent::ChainError(msg) => {
                    self.generate.generating = false;
                    self.generate.progress.generation_started_at = None;
                    self.generate.progress.stage_started_at = None;
                    self.generate.error_message = Some(msg);
                }
                BackgroundEvent::GalleryDeleteFailed(msg) => {
                    self.apply_delete_failure(&msg);
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
        // Latent previews are for canvas clients (web SPA / desktop app);
        // the TUI progress bar already tracks DenoiseStep.
        SseProgressEvent::Preview { .. } => {}
        SseProgressEvent::DependencyWait { dependency, reason } => {
            progress.current_stage = Some(format!("Waiting for {dependency}: {reason}"));
            progress.stage_started_at = None;
            progress.clear_download();
            progress.clear_weight();
        }
        SseProgressEvent::StageStart { name } => {
            progress.current_stage = Some(name);
            // Each StageStart counts as a new pipeline step; tracking the
            // index gives the Timeline an at-a-glance "you are on step N"
            // indicator without needing an estimated total.
            progress.stage_index = progress.stage_index.saturating_add(1);
            progress.stage_started_at = Some(std::time::Instant::now());
            // Reset transient bars when the stream moves into a new phase.
            progress.clear_download();
            progress.clear_weight();
        }
        SseProgressEvent::StageDone { name, elapsed_ms } => {
            progress.current_stage = None;
            progress.stage_started_at = None;
            progress.push_log(ProgressLogEntry {
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
                progress.push_log(ProgressLogEntry {
                    message,
                    style: ProgressStyle::Info,
                });
            } else {
                progress.push_log(ProgressLogEntry {
                    message,
                    style: ProgressStyle::Info,
                });
            }
        }
        SseProgressEvent::CacheHit { resource } => {
            progress.push_log(ProgressLogEntry {
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
            progress.push_log(ProgressLogEntry {
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
            progress.push_log(ProgressLogEntry {
                message: format!("Pull complete: {model}"),
                style: ProgressStyle::Done,
            });
            return true;
        }
        SseProgressEvent::Queued { position, .. } => {
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

    // NOTE: the old `ParamField::visible_fields` capability tests moved to
    // `ui::create_form` (`visible_rows_*`, `scheduler_row_gated_on_capability`,
    // `video_section_only_when_caps_support_video`, …). The Mode/Host rows —
    // and their display tests — are gone: routing comes from the Machines
    // generation target.

    #[test]
    fn generate_params_display_size_merges_width_height() {
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.width = 1024;
        params.height = 768;
        assert_eq!(params.display_value(&ParamField::Size), "1024 \u{00d7} 768");
    }

    #[test]
    fn parse_size_input_accepts_wxh_forms() {
        assert_eq!(parse_size_input("1024x768"), Some((1024, 768)));
        assert_eq!(parse_size_input("1024 X 768"), Some((1024, 768)));
        assert_eq!(parse_size_input("1024 \u{00d7} 768"), Some((1024, 768)));
        // Clamped into the same range the old Width/Height rows enforced.
        assert_eq!(parse_size_input("64x9999"), Some((256, 4096)));
        assert_eq!(parse_size_input("banana"), None);
        assert_eq!(parse_size_input("1024"), None);
        assert_eq!(parse_size_input("1x2x3"), None);
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

    // (`every_field_row_has_a_nonempty_label` in `ui::create_form` covers
    // the label contract for every row the form can produce.)

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
        state.push_log(ProgressLogEntry {
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
    fn dependency_wait_is_visible_without_counting_as_a_started_stage() {
        let mut state = ProgressState {
            stage_index: 3,
            stage_started_at: Some(std::time::Instant::now()),
            download_filename: "stale.safetensors".to_string(),
            weight_component: "stale".to_string(),
            ..Default::default()
        };

        reduce_progress_state(
            &mut state,
            SseProgressEvent::DependencyWait {
                dependency: "ltx2.3-vae".to_string(),
                reason: "download in progress".to_string(),
            },
        );

        assert_eq!(
            state.current_stage.as_deref(),
            Some("Waiting for ltx2.3-vae: download in progress")
        );
        assert_eq!(state.stage_index, 3);
        assert!(state.stage_started_at.is_none());
        assert!(state.download_filename.is_empty());
        assert!(state.weight_component.is_empty());
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
    fn seed_display_merges_mode_and_value() {
        // The Seed essentials row absorbed the old SeedValue row.
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        assert_eq!(params.display_value(&ParamField::Seed), "random");
        params.seed_mode = SeedMode::Fixed;
        params.seed = Some(12345);
        assert_eq!(
            params.display_value(&ParamField::Seed),
            "fixed \u{00b7} 12345"
        );
        params.seed_mode = SeedMode::Increment;
        params.seed = Some(11275518943372801901);
        assert_eq!(
            params.display_value(&ParamField::Seed),
            "increment \u{00b7} 11275518943372801901"
        );
        // Pinned mode without a value yet still communicates "random".
        params.seed_mode = SeedMode::Fixed;
        params.seed = None;
        assert!(params.display_value(&ParamField::Seed).contains("random"));
    }

    // ── Regression tests for Codex review findings ────────

    #[test]
    fn history_nav_only_from_prompt_focus() {
        // History navigation should only work from Prompt focus,
        // not NegativePrompt — prevents clobbering the main prompt.
        let mut history = crate::history::PromptHistory::empty();
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
    fn create_rows_end_with_reset_defaults_and_never_offer_unload() {
        // UnloadModel left Create for good — Models owns `u`. The reset
        // action row closes the form in every accordion state.
        use crate::ui::create_form::{visible_rows, AdvancedState, CreateRow};
        let caps = crate::model_info::capabilities_for_family("flux");
        for adv in [
            AdvancedState::default(),
            AdvancedState {
                open: true,
                expanded: None,
            },
        ] {
            let rows = visible_rows(&caps, &adv);
            assert_eq!(*rows.last().unwrap(), CreateRow::ResetDefaults);
        }
    }

    #[test]
    fn upscale_display_value_off_by_default() {
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        assert_eq!(params.display_value(&ParamField::Upscale), "off");
        params.upscale_model = Some("real-esrgan-x2:fp16".to_string());
        assert_eq!(
            params.display_value(&ParamField::Upscale),
            "real-esrgan-x2:fp16"
        );
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
                batch_id: None,
                batch_index: None,
                batch_count: None,
                model: "flux:q8".to_string(),
                seed: 42,
                steps: 20,
                guidance: 7.5,
                width: 1024,
                height: 1024,
                generation_width: Some(1024),
                generation_height: Some(1024),
                strength: None,
                source_image_name: None,
                source_image_sha256: None,
                scheduler: None,
                output_format: Some(mold_core::OutputFormat::Png),
                cfg_plus: None,
                lora: None,
                lora_scale: None,
                loras: None,
                control_model: None,
                control_scale: None,
                upscale_model: None,
                gif_preview: None,
                enable_audio: None,
                audio_file_path: None,
                source_video_path: None,
                pipeline: None,
                retake_range: None,
                spatial_upscale: None,
                temporal_upscale: None,
                version: "0.3.1".to_string(),
                frames: None,
                fps: None,
            },
            generation_time_ms: Some(5000),
            timestamp: 1234,
            server_url: None,
            origins: Vec::new(),
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
                batch_id: None,
                batch_index: None,
                batch_count: None,
                model: "test".to_string(),
                seed: 0,
                steps: 1,
                guidance: 0.0,
                width: 512,
                height: 512,
                generation_width: Some(512),
                generation_height: Some(512),
                strength: None,
                source_image_name: None,
                source_image_sha256: None,
                scheduler: None,
                output_format: Some(mold_core::OutputFormat::Png),
                cfg_plus: None,
                lora: None,
                lora_scale: None,
                loras: None,
                control_model: None,
                control_scale: None,
                upscale_model: None,
                gif_preview: None,
                enable_audio: None,
                audio_file_path: None,
                source_video_path: None,
                pipeline: None,
                retake_range: None,
                spatial_upscale: None,
                temporal_upscale: None,
                version: "0.0.0".to_string(),
                frames: None,
                fps: None,
            },
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            origins: Vec::new(),
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
            batch_id: None,
            batch_index: None,
            batch_count: None,
            model: "flux:q8".to_string(),
            seed: 42,
            steps: 20,
            guidance: 7.5,
            width: 1024,
            height: 1024,
            generation_width: Some(1024),
            generation_height: Some(1024),
            strength: Some(0.75),
            source_image_name: None,
            source_image_sha256: None,
            scheduler: None,
            output_format: Some(mold_core::OutputFormat::Png),
            cfg_plus: None,
            lora: Some("/path/to/adapter.safetensors".to_string()),
            lora_scale: Some(0.8),
            loras: Some(vec![mold_core::LoraWeight {
                path: "/path/to/adapter.safetensors".to_string(),
                scale: 0.8,
            }]),
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            pipeline: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
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
            origins: Vec::new(),
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
        let state = GalleryState::default();
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
        // Make sure this test process never touches a real DB unless a
        // specific test body opts in via `test_env::with_isolated_env`.
        crate::test_env::disable_db_for_non_isolated_tests();
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
        let advanced = crate::ui::create_form::AdvancedState::default();
        let rows = crate::ui::create_form::visible_rows(&caps, &advanced);
        let (bg_tx, bg_rx) = mpsc::unbounded_channel();

        App {
            active_view: View::Settings,
            create_mode: CreateMode::default(),
            generate: GenerateState {
                prompt: TextArea::default(),
                negative_prompt: TextArea::default(),
                params,
                focus: GenerateFocus::Navigation,
                param_index: 0,
                rows,
                advanced,
                param_scroll: 0,
                capabilities: caps,
                progress: ProgressState::default(),
                preview_image: None,
                image_state: None,
                animation: None,
                generating: false,
                batch_remaining: 0,
                last_seed: None,
                last_generation_time_ms: None,
                error_message: None,
                model_description: String::new(),
                last_output_path: None,
            },
            gallery: GalleryState::default(),
            models: ModelsState {
                catalog: Vec::new(),
                selected: 0,
                filter: String::new(),
                filtering: false,
            },
            machines: crate::hosts::MachinesState::default(),
            target: crate::hosts::GenTarget::default(),
            settings: SettingsState {
                selected_model: Some("test-model:q8".to_string()),
                row_index: 1,
                skip_save: true,
                ..Default::default()
            },
            prefs: crate::prefs::TuiPrefs::default(),
            script: crate::ui::script_composer::ScriptComposerState::default(),
            config,
            server_url: None,
            motion: crate::motion::MotionState::new(false),
            picker,
            theme: crate::ui::theme::Theme::default(),
            popup: None,
            should_quit: false,
            bg_tx,
            bg_rx,
            tokio_handle: tokio::runtime::Handle::current(),
            resource_info: crate::ui::info::ResourceInfo::default(),
            // Start test apps with an empty in-memory history — avoids
            // reaching into whatever MOLD_DB_PATH currently points at,
            // which was the source of flakes in parallel runs.
            history: crate::history::PromptHistory::empty(),
            layout: LayoutAreas::default(),
            server_process: None,
            upscale_in_progress: false,
            upscale_task: None,
            upscale_tile_progress: None,
            upscale_progress: ProgressState::default(),
            connecting: false,
            show_timeline: true,
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
    fn view_labels_and_indices() {
        assert_eq!(View::Create.label(), "Create");
        assert_eq!(View::Library.label(), "Library");
        assert_eq!(View::Models.label(), "Models");
        assert_eq!(View::Machines.label(), "Machines");
        assert_eq!(View::Settings.label(), "Settings");
        // Machines sits at index 3 between Models and Settings.
        assert_eq!(View::Machines.index(), 3);
        assert_eq!(View::Settings.index(), 4);
        assert_eq!(View::ALL.len(), 5);
        assert_eq!(View::ALL[3], View::Machines);
        assert_eq!(View::ALL[4], View::Settings);
    }

    #[tokio::test]
    async fn chain_is_a_create_submode_not_a_tab() {
        // The chain composer must never reappear as a sixth tab — it nests
        // under Create like the desktop's /create/chain route.
        assert_eq!(View::ALL.len(), 5);
        let mut app = make_settings_test_app();
        app.active_view = View::Settings;
        app.dispatch_action(Action::ChainEnter);
        assert_eq!(app.active_view, View::Create);
        assert_eq!(app.create_mode, CreateMode::Chain);
        // Leaving Create and coming back keeps the chain in progress.
        app.dispatch_action(Action::SwitchView(View::Library));
        app.dispatch_action(Action::SwitchView(View::Create));
        assert_eq!(app.create_mode, CreateMode::Chain);
        // ChainExit is the only way back to compose.
        app.dispatch_action(Action::ChainExit);
        assert_eq!(app.create_mode, CreateMode::Compose);
    }

    #[tokio::test]
    async fn ctrl_k_opens_palette_from_every_view_and_focus() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        for view in View::ALL {
            let mut app = make_settings_test_app();
            app.active_view = view;
            app.handle_crossterm_event(Event::Key(KeyEvent::new(
                KeyCode::Char('k'),
                KeyModifiers::CONTROL,
            )));
            assert!(
                matches!(app.popup, Some(Popup::CommandPalette { .. })),
                "^K must open the palette from {view:?}"
            );
        }
        // Even while the prompt textarea has focus.
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Prompt;
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Char('k'),
            KeyModifiers::CONTROL,
        )));
        assert!(matches!(app.popup, Some(Popup::CommandPalette { .. })));
    }

    #[tokio::test]
    async fn palette_filters_dispatches_and_closes() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let mut app = make_settings_test_app();
        app.active_view = View::Settings;
        app.dispatch_action(Action::OpenPalette);

        // Typing filters instead of triggering view actions — `q` while
        // the palette is open must not quit.
        for c in "quit".chars() {
            app.handle_crossterm_event(Event::Key(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
        assert!(!app.should_quit, "typing in the palette must not quit");
        let Some(Popup::CommandPalette { filter, .. }) = &app.popup else {
            panic!("palette should still be open");
        };
        assert_eq!(filter, "quit");

        // Backspace all, filter to Library, Enter navigates and closes.
        for _ in 0..4 {
            app.handle_crossterm_event(Event::Key(KeyEvent::new(
                KeyCode::Backspace,
                KeyModifiers::NONE,
            )));
        }
        for c in "library".chars() {
            app.handle_crossterm_event(Event::Key(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.active_view, View::Library);
        assert!(app.popup.is_none(), "palette closes after dispatch");
    }

    #[tokio::test]
    async fn palette_esc_closes_without_action() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let mut app = make_settings_test_app();
        let before = app.active_view;
        app.dispatch_action(Action::OpenPalette);
        app.handle_crossterm_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)));
        assert!(app.popup.is_none());
        assert_eq!(app.active_view, before);
        assert!(!app.should_quit);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn palette_set_theme_applies_and_persists_slug() {
        crate::test_env::with_isolated_env(|_home| {
            use crate::ui::theme::ThemePreset;
            let mut app = make_settings_test_app();
            app.dispatch_action(Action::SetTheme(ThemePreset::SafelightDark));
            assert_eq!(app.settings.theme_preset, ThemePreset::SafelightDark);
            assert_eq!(app.theme.bg, ThemePreset::SafelightDark.build().bg);
            let loaded = crate::session::TuiSession::load();
            assert_eq!(loaded.theme.as_deref(), Some("safelight-dark"));
        });
    }

    #[tokio::test]
    async fn palette_renders_on_narrow_terminal() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        app.dispatch_action(Action::OpenPalette);
        let backend = TestBackend::new(30, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| crate::ui::render(f, &mut app)).unwrap();
    }

    #[tokio::test]
    async fn activity_line_states() {
        use crate::ui::chrome::{activity_line, ActivityKind};
        let mut app = make_settings_test_app();
        app.generate.params.model = "cv:1759168".into();
        let mut catalog_model =
            make_test_catalog_entry("cv:1759168", 25, 7.0, 1024, 1024, "legacy title");
        catalog_model.display_name = Some("Juggernaut XL - Ragnarok".into());
        app.models.catalog = vec![catalog_model];

        // Idle: names the model and host; never fakes a queue depth.
        let (kind, text) = activity_line(&app);
        assert_eq!(kind, ActivityKind::Idle);
        let expected = "idle · Juggernaut XL - Ragnarok";
        assert!(text.starts_with(expected), "{text}");
        assert!(!text.contains("cv:1759168"), "{text}");
        assert!(
            !text.contains("queue"),
            "queue must be omitted when unknown"
        );

        // Generating with denoise progress: step/total + it/s.
        app.generate.generating = true;
        app.generate.progress.denoise_step = 12;
        app.generate.progress.denoise_total = 28;
        app.generate.progress.denoise_elapsed_ms = 6000;
        let (kind, text) = activity_line(&app);
        assert_eq!(kind, ActivityKind::Generating);
        assert!(
            text.contains("developing · Juggernaut XL - Ragnarok"),
            "{text}"
        );
        assert!(text.contains("12/28"), "{text}");
        assert!(text.contains("2.0 it/s"), "{text}");

        // it/s is omitted (not NaN/inf) when elapsed is zero.
        app.generate.progress.denoise_elapsed_ms = 0;
        let (_, text) = activity_line(&app);
        assert!(!text.contains("it/s"), "{text}");

        // Done: needs a duration; names the save dir only when known.
        app.generate.generating = false;
        app.generate.last_generation_time_ms = Some(4000);
        app.generate.last_output_path = None;
        let (kind, text) = activity_line(&app);
        assert_eq!(kind, ActivityKind::Done);
        assert!(text.contains("done · 4.0s"), "{text}");
        assert!(!text.contains("saved to"), "{text}");
        app.generate.last_output_path = Some(std::path::PathBuf::from("/tmp/out/print.png"));
        let (_, text) = activity_line(&app);
        assert!(text.contains("saved to /tmp/out"), "{text}");

        // Error wins over done and shows only the first line.
        app.generate.error_message = Some("boom\nsecond line".into());
        let (kind, text) = activity_line(&app);
        assert_eq!(kind, ActivityKind::Error);
        assert!(text.contains("error · boom"), "{text}");
        assert!(!text.contains("second"), "{text}");
    }

    #[tokio::test]
    async fn host_chip_states() {
        use crate::ui::chrome::{host_chip, ChipState};
        let mut app = make_settings_test_app();

        // Local mode: ready chip naming this machine + compiled backend.
        app.generate.params.inference_mode = InferenceMode::Local;
        let (state, text) = host_chip(&app);
        assert_eq!(state, ChipState::Ready);
        assert!(text.starts_with("This "), "{text}");

        // Remote with a status: hostname shown.
        app.generate.params.inference_mode = InferenceMode::Auto;
        app.resource_info.server_status = Some(mold_core::ServerStatus {
            version: "0.0.0".into(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: false,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 0,
            hostname: Some("studio".into()),
            memory_status: None,
            gpus: None,
            queue_depth: Some(2),
            queue_capacity: None,
            queue_paused: None,
            instance_id: None,
            models_disk: None,
        });
        let (state, text) = host_chip(&app);
        assert_eq!(state, ChipState::Ready);
        assert!(text.starts_with("studio"), "{text}");
        // A reported queue depth surfaces once a job state exists (the
        // idle line deliberately omits it, matching the mockup).
        app.generate.last_generation_time_ms = Some(1000);
        let (_, line) = crate::ui::chrome::activity_line(&app);
        assert!(line.contains("queue 2"), "{line}");
        app.generate.last_generation_time_ms = None;

        // No status + connecting flag → connecting chip.
        app.resource_info.server_status = None;
        app.connecting = true;
        let (state, _) = host_chip(&app);
        assert_eq!(state, ChipState::Connecting);

        // No status, not connecting → offline chip.
        app.connecting = false;
        let (state, _) = host_chip(&app);
        assert_eq!(state, ChipState::Offline);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn toggle_advanced_flips_disclosure_and_rows() {
        use crate::ui::create_form::CreateRow;
        let mut app = make_settings_test_app();
        assert!(!app.generate.advanced.open);
        let collapsed_rows = app.generate.rows.len();
        app.dispatch_action(Action::ToggleAdvanced);
        assert!(app.generate.advanced.open);
        assert!(
            app.generate
                .rows
                .iter()
                .any(|r| matches!(r, CreateRow::Section(_))),
            "opening the disclosure must surface the section rows"
        );
        app.dispatch_action(Action::ToggleAdvanced);
        assert!(!app.generate.advanced.open);
        assert_eq!(app.generate.rows.len(), collapsed_rows);
    }

    // ── Settings E2E: display values ──────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
    async fn settings_display_all_logging_defaults() {
        let app = make_settings_test_app();
        assert_eq!(app.settings_display_value(&SettingsKey::LogLevel), "info");
        assert_eq!(app.settings_display_value(&SettingsKey::LogFile), "off");
        assert_eq!(app.settings_display_value(&SettingsKey::LogMaxDays), "7");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_server_port() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ServerPort, 1.0, 1024.0, 65535.0);
        assert_eq!(app.config.server_port, 7681);
        app.settings_adjust_number(SettingsKey::ServerPort, -2.0, 1024.0, 65535.0);
        assert_eq!(app.config.server_port, 7679);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_default_width() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::DefaultWidth, 64.0, 64.0, 4096.0);
        assert_eq!(app.config.default_width, 832);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_default_height() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::DefaultHeight, -64.0, 64.0, 4096.0);
        assert_eq!(app.config.default_height, 704);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_default_steps() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::DefaultSteps, 1.0, 1.0, 200.0);
        assert_eq!(app.config.default_steps, 5);
    }

    // ── User story: change themes while a generation is running ───
    // Reported: "as a user I should be able to change themes while the
    // app is generating an image". Three regression tests covering the
    // full keyboard path: escape the prompt → switch view → cycle theme.

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn alt_5_while_generating_from_prompt_focus_switches_to_settings() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Prompt;
        app.generate.generating = true;
        app.generate.progress.mark_generation_start();

        // `Alt+5` must escape the prompt textarea and switch views,
        // even with a generation in flight — otherwise users have to
        // wait for the run to finish before they can reach Settings.
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Char('5'),
            KeyModifiers::ALT,
        )));
        assert_eq!(app.active_view, View::Settings);
        assert!(app.generate.generating, "generation must not be aborted");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn theme_cycle_applies_immediately_while_generating() {
        use crate::ui::theme::ThemePreset;
        let mut app = make_settings_test_app();
        app.active_view = View::Settings;
        app.settings.focus = SettingsFocus::Appearance;
        // In-flight generation on the Settings tab.
        app.generate.generating = true;
        app.generate.progress.mark_generation_start();

        let before = app.settings.theme_preset;
        assert_eq!(before, ThemePreset::StudioDark);

        // Right arrow on Appearance cycles the preset. The new palette
        // must apply to `app.theme` immediately so the next render
        // paints the running Timeline bars in the chosen theme —
        // generating must not veto the palette change.
        app.dispatch_action(Action::Increment);

        assert_ne!(app.settings.theme_preset, before);
        assert_eq!(app.theme.bg, app.settings.theme_preset.build().bg);
        assert!(app.generate.generating, "generation must keep running");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn esc_then_5_reaches_settings_while_generating() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Prompt;
        app.generate.generating = true;
        app.generate.progress.mark_generation_start();

        // The discoverable path: Esc unfocuses the textarea, then `5`
        // switches to Settings. Both must keep working while a
        // generation is active.
        app.handle_crossterm_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)));
        assert_eq!(app.generate.focus, GenerateFocus::Navigation);
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Char('5'),
            KeyModifiers::NONE,
        )));
        assert_eq!(app.active_view, View::Settings);
    }

    // ── Enter on the Appearance pane must not trigger settings_confirm ──

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn enter_on_appearance_pane_does_not_open_model_dialog() {
        let mut app = make_settings_test_app();
        app.active_view = View::Settings;
        app.settings.focus = SettingsFocus::Appearance;
        // Before the fix: Confirm on Settings view unconditionally calls
        // `settings_confirm()`, which follows row_index=1 (the first
        // editable field — `Model`) and opens its text-entry popup, even
        // though focus visibly belongs to the Appearance swatch row.
        app.dispatch_action(Action::Confirm);
        assert!(
            app.popup.is_none(),
            "Enter on the Appearance pane must stay on the swatch grid \
             and must not open the Model popup"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn enter_on_configuration_still_opens_popup() {
        // Regression guard for the happy path.
        let mut app = make_settings_test_app();
        app.active_view = View::Settings;
        app.settings.focus = SettingsFocus::Configuration;
        app.settings.row_index = find_settings_row(&app, SettingsKey::DefaultModel);
        app.dispatch_action(Action::Confirm);
        assert!(
            app.popup.is_some(),
            "Enter on a Configuration Text row must still open the popup"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_cycle_theme_wraps_both_directions() {
        // Writes to MOLD_DB_PATH via `apply_theme_preset`, so it must
        // serialize with the other theme tests to avoid clobbering
        // their DB state.
        use crate::ui::theme::ThemePreset;
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            // Starts on the default (Studio Dark).
            assert_eq!(app.settings.theme_preset, ThemePreset::StudioDark);
            // Forward cycles to Studio Light and also rebuilds `app.theme`.
            app.settings_cycle_theme(1);
            assert_eq!(app.settings.theme_preset, ThemePreset::StudioLight);
            // `app.theme` should now match the Studio Light palette.
            assert_eq!(app.theme.bg, ThemePreset::StudioLight.build().bg);
            // Backward from Studio Dark (index 0) wraps to Dracula (last).
            app.apply_theme_preset(ThemePreset::StudioDark);
            app.settings_cycle_theme(-1);
            assert_eq!(app.settings.theme_preset, ThemePreset::Dracula);
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_navigate_up_from_top_focuses_appearance() {
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Configuration;
        // Jump to the first settings field and press Up past the top.
        app.settings.row_index = 1;
        app.settings_navigate(-1);
        app.settings_navigate(-1);
        assert_eq!(app.settings.focus, SettingsFocus::Appearance);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_navigate_down_from_bottom_card_row_enters_configuration() {
        use crate::ui::theme::ThemePreset;
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Appearance;
        app.settings.appearance_cols = 4;
        // Studio Dark (index 0) → Down walks the grid one row at a time:
        // 0 → 4 (Mocha) → 8 (Tokyo) → off the bottom → Configuration.
        app.settings_navigate(1);
        assert_eq!(app.settings.theme_preset, ThemePreset::Mocha);
        assert_eq!(app.settings.focus, SettingsFocus::Appearance);
        app.settings_navigate(1);
        assert_eq!(app.settings.theme_preset, ThemePreset::Tokyo);
        app.settings_navigate(1);
        assert_eq!(app.settings.focus, SettingsFocus::Configuration);
        // The selection stays where it was when focus moved on.
        assert_eq!(app.settings.theme_preset, ThemePreset::Tokyo);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_navigate_up_moves_selection_a_row_and_stops_at_top() {
        use crate::ui::theme::ThemePreset;
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Appearance;
        app.settings.appearance_cols = 4;
        app.apply_theme_preset(ThemePreset::Tokyo); // index 8, row 2
        app.settings_navigate(-1);
        assert_eq!(app.settings.theme_preset, ThemePreset::Mocha); // index 4
        app.settings_navigate(-1);
        assert_eq!(app.settings.theme_preset, ThemePreset::StudioDark); // index 0
                                                                        // Top row: Up is a no-op — focus and selection both hold.
        app.settings_navigate(-1);
        assert_eq!(app.settings.theme_preset, ThemePreset::StudioDark);
        assert_eq!(app.settings.focus, SettingsFocus::Appearance);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_navigate_appearance_live_applies_the_row_moves() {
        use crate::ui::theme::ThemePreset;
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Appearance;
        app.settings.appearance_cols = 4;
        app.settings_navigate(1);
        // Row movement is a selection change → the running theme follows
        // immediately, exactly like Left/Right's linear cycle.
        let expected = ThemePreset::Mocha.build();
        assert_eq!(app.theme.bg, expected.bg);
        assert_eq!(app.theme.accent, expected.accent);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_tab_flips_focus_without_moving_the_theme_selection() {
        use crate::ui::theme::ThemePreset;
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Appearance;
        app.settings.appearance_cols = 4;
        let before = app.settings.theme_preset;
        app.dispatch_action(Action::FocusNext);
        assert_eq!(app.settings.focus, SettingsFocus::Configuration);
        app.dispatch_action(Action::FocusNext);
        assert_eq!(app.settings.focus, SettingsFocus::Appearance);
        app.dispatch_action(Action::FocusPrev);
        assert_eq!(app.settings.focus, SettingsFocus::Configuration);
        // Unlike ↓, Tab never touches (or live-applies) the selection.
        assert_eq!(app.settings.theme_preset, before);
        assert_eq!(app.settings.theme_preset, ThemePreset::StudioDark);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_navigate_appearance_tolerates_unset_cols() {
        use crate::ui::theme::ThemePreset;
        // Before the first render `appearance_cols` is 0 — navigation
        // must clamp to 1 column rather than panic or freeze.
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Appearance;
        app.settings.appearance_cols = 0;
        app.settings_navigate(1);
        assert_eq!(app.settings.theme_preset, ThemePreset::StudioLight);
    }

    // ── Codex P2: Alt-key bypass from prompt textarea ─────────────

    fn alt_key_event(code: crossterm::event::KeyCode) -> crossterm::event::Event {
        use crossterm::event::{Event, KeyEvent, KeyModifiers};
        Event::Key(KeyEvent::new(code, KeyModifiers::ALT))
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn alt_5_while_typing_prompt_switches_to_settings() {
        use crossterm::event::KeyCode;
        let mut app = make_settings_test_app();
        app.generate.focus = GenerateFocus::Prompt;
        app.active_view = View::Create;
        // While focused on the Prompt textarea, Alt+5 must reach the action
        // mapper and switch the active view. Before the bypass fix this
        // event would be consumed by `TextArea::input` and the view would
        // stay on Generate.
        app.handle_crossterm_event(alt_key_event(KeyCode::Char('5')));
        assert_eq!(app.active_view, View::Settings);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn alt_4_while_typing_prompt_switches_to_queue() {
        use crossterm::event::KeyCode;
        let mut app = make_settings_test_app();
        app.generate.focus = GenerateFocus::Prompt;
        app.active_view = View::Create;
        // Alt+4 was re-pointed to Queue in phase 4. Regression guard so the
        // old Alt+4 → Settings mapping can't sneak back in.
        app.handle_crossterm_event(alt_key_event(KeyCode::Char('4')));
        assert_eq!(app.active_view, View::Machines);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn alt_n_focuses_inline_negative_editor() {
        use crate::ui::create_form::{AdvSection, CreateRow};
        use crossterm::event::KeyCode;
        let mut app = make_settings_test_app();
        app.generate.capabilities.supports_negative_prompt = true;
        app.refresh_create_rows();
        app.generate.focus = GenerateFocus::Prompt;
        app.active_view = View::Create;
        // Alt+N (even while typing in the prompt) opens the accordion,
        // expands Negative, and drops focus into the inline editor.
        app.handle_crossterm_event(alt_key_event(KeyCode::Char('n')));
        assert!(app.generate.advanced.open);
        assert_eq!(app.generate.advanced.expanded, Some(AdvSection::Negative));
        assert_eq!(app.generate.focus, GenerateFocus::NegativePrompt);
        assert_eq!(
            app.generate.rows.get(app.generate.param_index),
            Some(&CreateRow::NegativeEditor),
            "selection must land on the inline editor row"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn alt_n_is_a_noop_without_negative_capability() {
        use crossterm::event::KeyCode;
        let mut app = make_settings_test_app();
        app.generate.capabilities.supports_negative_prompt = false;
        app.refresh_create_rows();
        app.generate.focus = GenerateFocus::Prompt;
        app.active_view = View::Create;
        app.handle_crossterm_event(alt_key_event(KeyCode::Char('n')));
        assert!(!app.generate.advanced.open);
        assert_eq!(app.generate.focus, GenerateFocus::Prompt);
    }

    // ── Focus must skip the Negative editor when it isn't drawn ──────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn tab_from_prompt_skips_negative_when_accordion_closed() {
        let mut app = make_settings_test_app();
        // Model supports negative prompts, but the inline editor only
        // renders while Advanced → Negative is expanded.
        app.generate.capabilities.supports_negative_prompt = true;
        app.refresh_create_rows();
        app.generate.focus = GenerateFocus::Prompt;
        app.active_view = View::Create;

        app.dispatch_action(Action::FocusNext);
        assert_eq!(app.generate.focus, GenerateFocus::Parameters);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn shift_tab_from_parameters_skips_negative_when_accordion_closed() {
        let mut app = make_settings_test_app();
        app.generate.capabilities.supports_negative_prompt = true;
        app.refresh_create_rows();
        app.generate.focus = GenerateFocus::Parameters;
        app.active_view = View::Create;

        app.dispatch_action(Action::FocusPrev);
        assert_eq!(app.generate.focus, GenerateFocus::Prompt);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn tab_still_visits_negative_when_editor_expanded() {
        // Happy path: with Advanced → Negative expanded the Tab cycle
        // includes the inline editor.
        use crate::ui::create_form::AdvSection;
        let mut app = make_settings_test_app();
        app.generate.capabilities.supports_negative_prompt = true;
        app.generate.advanced.open = true;
        app.generate.advanced.expanded = Some(AdvSection::Negative);
        app.refresh_create_rows();
        app.generate.focus = GenerateFocus::Prompt;
        app.active_view = View::Create;

        app.dispatch_action(Action::FocusNext);
        assert_eq!(app.generate.focus, GenerateFocus::NegativePrompt);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn mouse_click_on_gallery_tile_row_2_selects_correct_tile() {
        // Regression for "click boxes are finicky in general": the mouse
        // handler was using `cell_h = 14u16` after the grid was shrunk
        // to `CELL_H = 12` in ui::gallery. Each row of tiles drifted the
        // hit-test by 2 rows — clicking on row 2 would select the row-1
        // tile (or nothing).
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
        let mut app = make_settings_test_app();
        app.active_view = View::Library;
        app.gallery.view_mode = GalleryViewMode::Grid;
        // 3 columns, 3 rows worth of tiles = 9 entries.
        for i in 0..9 {
            app.gallery.entries.push(GalleryEntry {
                path: std::path::PathBuf::from(format!("tile-{i}.png")),
                metadata: make_test_metadata(),
                generation_time_ms: None,
                timestamp: 0,
                server_url: None,
                origins: Vec::new(),
            });
            app.gallery.thumbnail_states.push(None);
            app.gallery.thumb_dimensions.push(None);
            app.gallery.thumb_fixed_cache.push(None);
        }
        app.gallery.grid_cols = 3;
        app.gallery.grid_scroll = 0;
        app.gallery.refresh_filter();
        // Gallery grid inner area in a representative layout.
        app.layout.gallery_grid = ratatui::layout::Rect::new(0, 3, 72, 40);

        // Click dead-center of the tile at grid (col=1, row=2).
        // With CELL_W=24 and CELL_H=12, that tile occupies
        // cols 24..=47 and rows (3 + 24)..=(3 + 35). Midpoint col ≈ 36,
        // row ≈ 30. With the old `cell_h=14` the click would have been
        // interpreted as row 1 (tile index 4) instead of row 2 (index 7).
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 36,
            row: 30,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        let expected_index = 2 * 3 + 1; // row 2 * 3 cols + col 1
        assert_eq!(
            app.gallery.selected, expected_index,
            "click on tile (col=1, row=2) at (col=36,row=30) should select index {expected_index} — \
             mouse hit-test must track the real CELL_H, not the stale 14"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn mouse_click_on_each_tab_switches_to_that_view() {
        // Click every tab at its start/middle/end columns and assert the
        // active_view lands on the expected tab. Regression reproducer
        // for "clicking on Queue with mouse doesn't always work" — the
        // existing hit-test math did +2 (border + padding) even though
        // the block has no left border. Anchoring at each real column
        // exposes the off-by-one that matters on real-world tab widths.
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};

        // Tab bar is 3 rows tall; tabs render on row 1 (under the title
        // row, above the bottom border). 120-col terminal (wide enough for
        // all 6 tabs).
        let tab_bar = ratatui::layout::Rect::new(0, 0, 120, 3);

        // Actual rendered layout (verified via TestBackend probe) —
        // ratatui's Tabs widget adds its own pad_left(" ") and
        // pad_right(" ") *around* each title, on top of the " N Label "
        // content we pass in. Stride per tab = label.len() + 6 (pad +
        // title + pad), plus a 1-col divider between tabs.
        //
        //   col 0        → block horizontal padding
        //   col 1        → Generate pad_left
        //   col 2..=13   → " 1 Generate " title (12 chars, "1" at col 3)
        //   col 14       → Generate pad_right
        //   col 15       → divider
        //   col 16       → Gallery pad_left
        //   col 17..=27  → " 2 Gallery " title (11 chars, "2" at col 18)
        //   col 28       → Gallery pad_right
        //   col 29       → divider
        //   col 30       → Models pad_left
        //   col 31..=40  → " 3 Models " title (10 chars, "3" at col 32)
        //   col 41       → Models pad_right
        //   col 42       → divider
        //   col 43       → Queue pad_left
        //   col 44..=52  → " 4 Queue " title (9 chars, "4" at col 45)
        //   col 53       → Queue pad_right
        //   col 54       → divider
        //   col 55       → Settings pad_left
        //   col 56..=67  → " 5 Settings " title (12 chars, "5" at col 57)
        //   col 68       → Settings pad_right
        //
        // Trailing dividers fold into the preceding tab's click zone so
        // there's no dead pixel. Cols 0..=13 → Create, 14..=27 → Library,
        // 28..=40 → Models, 41..=55 → Machines, 56..=69 → Settings.
        let cases: &[(u16, View, &str)] = &[
            (0, View::Create, "block padding"),
            (3, View::Create, "Create '1'"),
            (9, View::Create, "Create 'e'"),
            (13, View::Create, "Create trailing divider"),
            (14, View::Library, "Library pad_left"),
            (16, View::Library, "Library '2'"),
            (22, View::Library, "Library 'r'"),
            (27, View::Library, "Library trailing divider"),
            (28, View::Models, "Models pad_left"),
            (30, View::Models, "Models '3'"),
            (36, View::Models, "Models 's'"),
            (40, View::Models, "Models trailing divider"),
            (41, View::Machines, "Machines pad_left"),
            (43, View::Machines, "Machines '4'"),
            (48, View::Machines, "Machines body of label"),
            (53, View::Machines, "Machines end of title"),
            (55, View::Machines, "Machines trailing divider"),
            (56, View::Settings, "Settings pad_left"),
            (58, View::Settings, "Settings '5'"),
            (63, View::Settings, "Settings 'i'"),
            (69, View::Settings, "Settings pad_right"),
        ];

        for (col, expected, name) in cases {
            let mut app = make_settings_test_app();
            app.layout.tab_bar = tab_bar;
            app.active_view = View::Create; // deterministic starting view
            app.handle_mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: *col,
                row: 1,
                modifiers: crossterm::event::KeyModifiers::NONE,
            });
            assert_eq!(
                app.active_view, *expected,
                "clicking col {col} ({name}) should land on {expected:?}, got {:?}",
                app.active_view
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn mouse_click_past_last_tab_does_not_select_settings() {
        // The old behaviour mapped any click past the last rendered tab
        // (e.g. on the right-aligned version text or empty space) to
        // View::Settings. That made the tab bar feel "finicky" — clicks
        // on the host/version indicator silently switched views.
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
        let mut app = make_settings_test_app();
        app.layout.tab_bar = ratatui::layout::Rect::new(0, 0, 120, 3);
        app.active_view = View::Create;

        // Col 90 is well past Script (which ends around col 82) — it sits
        // under the right-aligned "mold 0.9.0" version indicator.
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 90,
            row: 1,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });

        assert_eq!(
            app.active_view,
            View::Create,
            "clicks past the last rendered tab must be a no-op, not a stealth jump to Settings"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn mouse_hit_test_matches_real_tab_bar_rendering() {
        // End-to-end guard: render the real UI, scan each column of the
        // tab bar row looking for the digit "1".."5", and assert that a
        // click on that column lands on the matching view. If anything
        // (padding, divider, label order) changes upstream, this test
        // surfaces the drift before users report a flaky tab bar.
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| crate::ui::render(f, &mut app)).unwrap();

        let tab_bar = app.layout.tab_bar;
        assert!(tab_bar.height >= 2, "tab bar should have room to render");
        let tab_row = tab_bar.y; // row 0 = tabs, row 1 = accent underline

        // Find the column of each digit "1".."5" in the rendered row.
        // The Tabs widget prefixes each label with " N " — those digits
        // anchor the hit-test and are the visually obvious click target.
        let buf = terminal.backend().buffer();
        let digit_to_view: &[(&str, View)] = &[
            ("1", View::Create),
            ("2", View::Library),
            ("3", View::Models),
            ("4", View::Machines),
            ("5", View::Settings),
        ];
        for (digit, expected) in digit_to_view {
            let col = (0..tab_bar.width)
                .find(|&x| buf[(tab_bar.x + x, tab_row)].symbol() == *digit)
                .unwrap_or_else(|| panic!("digit {digit} not found in rendered tab bar"));
            let click_col = tab_bar.x + col;

            let mut click_app = make_settings_test_app();
            click_app.layout.tab_bar = tab_bar;
            click_app.active_view = View::Create;
            click_app.handle_mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: click_col,
                row: tab_row,
                modifiers: crossterm::event::KeyModifiers::NONE,
            });
            assert_eq!(
                click_app.active_view, *expected,
                "clicking on digit '{digit}' at col {click_col} should land on {expected:?}, got {:?}",
                click_app.active_view
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn mouse_click_on_stale_negative_rect_does_not_focus_hidden_editor() {
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
        let mut app = make_settings_test_app();
        app.generate.capabilities.supports_negative_prompt = true;
        // Accordion closed → the inline editor is not rendered, so a click
        // on a stale stored rect must not focus the hidden textarea.
        app.refresh_create_rows();
        app.generate.focus = GenerateFocus::Prompt;
        app.active_view = View::Create;
        app.layout.negative_prompt = ratatui::layout::Rect::new(0, 5, 80, 1);
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 10,
            row: 5,
            modifiers: crossterm::event::KeyModifiers::NONE,
        });
        assert_ne!(app.generate.focus, GenerateFocus::NegativePrompt);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn closing_advanced_while_editing_negative_moves_focus_out() {
        use crate::ui::create_form::AdvSection;
        let mut app = make_settings_test_app();
        app.generate.capabilities.supports_negative_prompt = true;
        app.generate.advanced.open = true;
        app.generate.advanced.expanded = Some(AdvSection::Negative);
        app.refresh_create_rows();
        app.generate.focus = GenerateFocus::NegativePrompt;
        // Closing the disclosure hides the editor; focus must escape so
        // the user isn't stuck typing into a hidden textarea.
        app.dispatch_action(Action::ToggleAdvanced);
        assert!(!app.generate.advanced.open);
        assert_eq!(app.generate.focus, GenerateFocus::Parameters);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_increment_on_appearance_cycles_theme() {
        use crate::ui::theme::ThemePreset;
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Appearance;
        let before = app.settings.theme_preset;
        app.settings_increment(1);
        assert_ne!(app.settings.theme_preset, before);
        assert_eq!(app.settings.theme_preset, ThemePreset::StudioLight);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_expand_temperature() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ExpandTemperature, 0.1, 0.0, 2.0);
        assert!((app.config.expand.temperature - 0.8).abs() < 0.001);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_expand_top_p() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ExpandTopP, -0.05, 0.0, 1.0);
        assert!((app.config.expand.top_p - 0.85).abs() < 0.001);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_expand_max_tokens() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ExpandMaxTokens, 64.0, 64.0, 4096.0);
        assert_eq!(app.config.expand.max_tokens, 364);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_log_max_days() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::LogMaxDays, 1.0, 1.0, 365.0);
        assert_eq!(app.config.logging.max_days, 8);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_model_steps() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ModelSteps, 1.0, 1.0, 200.0);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert_eq!(mc.default_steps, Some(21));
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_model_guidance() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ModelGuidance, 0.5, 0.0, 30.0);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert!((mc.default_guidance.unwrap() - 4.0).abs() < 0.001);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_model_width() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ModelWidth, 64.0, 64.0, 4096.0);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert_eq!(mc.default_width, Some(1088));
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_model_height() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ModelHeight, -64.0, 64.0, 4096.0);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert_eq!(mc.default_height, Some(960));
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_adjust_model_lora_scale() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::ModelLoraScale, 0.1, 0.0, 2.0);
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert!((mc.lora_scale.unwrap() - 0.9).abs() < 0.001);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_numeric_clamps_at_min() {
        let mut app = make_settings_test_app();
        // Steps = 4, try decrementing by 100
        app.settings_adjust_number(SettingsKey::DefaultSteps, -100.0, 1.0, 200.0);
        assert_eq!(app.config.default_steps, 1);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_numeric_clamps_at_max() {
        let mut app = make_settings_test_app();
        app.settings_adjust_number(SettingsKey::DefaultSteps, 500.0, 1.0, 200.0);
        assert_eq!(app.config.default_steps, 200);
    }

    // ── Settings E2E: boolean toggles ─────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_toggle_embed_metadata() {
        let mut app = make_settings_test_app();
        assert!(app.config.embed_metadata);
        app.settings_toggle_bool(SettingsKey::EmbedMetadata);
        assert!(!app.config.embed_metadata);
        app.settings_toggle_bool(SettingsKey::EmbedMetadata);
        assert!(app.config.embed_metadata);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_toggle_expand_enabled() {
        let mut app = make_settings_test_app();
        assert!(!app.config.expand.enabled);
        app.settings_toggle_bool(SettingsKey::ExpandEnabled);
        assert!(app.config.expand.enabled);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_toggle_expand_thinking() {
        let mut app = make_settings_test_app();
        assert!(!app.config.expand.thinking);
        app.settings_toggle_bool(SettingsKey::ExpandThinking);
        assert!(app.config.expand.thinking);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_toggle_log_file() {
        let mut app = make_settings_test_app();
        assert!(!app.config.logging.file);
        app.settings_toggle_bool(SettingsKey::LogFile);
        assert!(app.config.logging.file);
    }

    // ── Settings E2E: toggle cycles ───────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
    async fn settings_cycle_qwen3_variant() {
        let mut app = make_settings_test_app();
        let opts = &["auto", "bf16", "q8", "q6", "iq4", "q3"];
        app.settings_cycle_toggle(SettingsKey::Qwen3Variant, opts, 1);
        assert_eq!(app.config.qwen3_variant, Some("bf16".into()));
        app.settings_cycle_toggle(SettingsKey::Qwen3Variant, opts, -1);
        assert!(app.config.qwen3_variant.is_none());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
    async fn settings_apply_default_model() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::DefaultModel, "sd15:fp16".into());
        assert_eq!(app.config.default_model, "sd15:fp16");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_apply_models_dir() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::ModelsDir, "/tmp/models".into());
        assert_eq!(app.config.models_dir, "/tmp/models");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_apply_output_dir() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::OutputDir, "/tmp/output".into());
        assert_eq!(app.config.output_dir, Some("/tmp/output".into()));
        // Empty clears
        app.settings_apply_input(SettingsKey::OutputDir, String::new());
        assert!(app.config.output_dir.is_none());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
    async fn settings_apply_expand_backend() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::ExpandBackend, "http://localhost:11434".into());
        assert_eq!(app.config.expand.backend, "http://localhost:11434");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_apply_expand_model() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::ExpandModel, "qwen3-expand:q4".into());
        assert_eq!(app.config.expand.model, "qwen3-expand:q4");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_apply_expand_api_model() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::ExpandApiModel, "gpt-4o".into());
        assert_eq!(app.config.expand.api_model, "gpt-4o");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_apply_log_dir() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::LogDir, "/tmp/logs".into());
        assert_eq!(app.config.logging.dir, Some("/tmp/logs".into()));
        app.settings_apply_input(SettingsKey::LogDir, String::new());
        assert!(app.config.logging.dir.is_none());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_apply_model_negative_prompt() {
        let mut app = make_settings_test_app();
        app.settings_apply_input(SettingsKey::ModelNegativePrompt, "watermark".into());
        let mc = app.config.models.get("test-model:q8").unwrap();
        assert_eq!(mc.negative_prompt, Some("watermark".into()));
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
    async fn settings_navigate_skips_headers() {
        let mut app = make_settings_test_app();
        // Start at index 0, which is a section header
        app.settings.row_index = 0;
        app.settings_navigate(1); // should skip header, land on first field
        let rows = app.build_settings_rows();
        assert!(rows[app.settings.row_index].is_field());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_navigate_clamps_at_boundaries() {
        let mut app = make_settings_test_app();
        app.settings.row_index = 0;
        app.settings_navigate(-1); // can't go above 0
        assert_eq!(app.settings.row_index, 0);
    }

    // ── Settings E2E: build_settings_rows structure ───────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
    async fn settings_increment_via_row_index_adjusts_width() {
        let mut app = make_settings_test_app();
        let idx = find_settings_row(&app, SettingsKey::DefaultWidth);
        app.settings.row_index = idx;
        app.settings_increment(1);
        assert_eq!(app.config.default_width, 832); // 768 + 64
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_increment_via_row_index_toggles_bool() {
        let mut app = make_settings_test_app();
        let idx = find_settings_row(&app, SettingsKey::EmbedMetadata);
        app.settings.row_index = idx;
        assert!(app.config.embed_metadata);
        app.settings_increment(1);
        assert!(!app.config.embed_metadata);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
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
    #[serial_test::serial(mold_env)]
    async fn settings_confirm_cycles_toggle_field() {
        let mut app = make_settings_test_app();
        let idx = find_settings_row(&app, SettingsKey::T5Variant);
        app.settings.row_index = idx;
        app.settings_confirm();
        assert_eq!(app.config.t5_variant, Some("fp16".into()));
        assert!(app.popup.is_none());
    }

    // ── Preferences rows (Settings redesign) ───────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_rows_start_with_the_preferences_section() {
        let app = make_settings_test_app();
        let rows = app.build_settings_rows();
        assert!(
            matches!(&rows[0], SettingsRow::SectionHeader { name } if name == "Preferences"),
            "first row must be the Preferences header"
        );
        // The four preference fields follow in mock order, before General.
        let keys: Vec<SettingsKey> = rows
            .iter()
            .take(5)
            .filter_map(|r| match r {
                SettingsRow::Field { key, .. } => Some(*key),
                _ => None,
            })
            .collect();
        assert_eq!(
            keys,
            vec![
                SettingsKey::PrefDefaultFormat,
                SettingsKey::PrefReduceMotion,
                SettingsKey::PrefShowTimeline,
                SettingsKey::PrefConfirmDestructive,
            ]
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_pref_display_values_reflect_prefs() {
        let mut app = make_settings_test_app();
        assert_eq!(
            app.settings_display_value(&SettingsKey::PrefDefaultFormat),
            "png"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::PrefReduceMotion),
            "off"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::PrefShowTimeline),
            "on"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::PrefConfirmDestructive),
            "on"
        );
        app.prefs.default_format = OutputFormat::Jpeg;
        app.prefs.reduce_motion = true;
        assert_eq!(
            app.settings_display_value(&SettingsKey::PrefDefaultFormat),
            "jpeg"
        );
        assert_eq!(
            app.settings_display_value(&SettingsKey::PrefReduceMotion),
            "on"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_increment_toggles_pref_bools_without_touching_config() {
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Configuration;
        app.settings.row_index = find_settings_row(&app, SettingsKey::PrefReduceMotion);
        app.settings_increment(1);
        assert!(app.prefs.reduce_motion);
        app.settings_increment(1);
        assert!(!app.prefs.reduce_motion);
        // DB-backed prefs must never trip the config.toml save-error path.
        assert!(app.settings.save_error.is_none());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_increment_cycles_default_format_pref() {
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Configuration;
        app.settings.row_index = find_settings_row(&app, SettingsKey::PrefDefaultFormat);
        app.settings_increment(1);
        assert_eq!(app.prefs.default_format, OutputFormat::Jpeg);
        app.settings_increment(1);
        assert_eq!(app.prefs.default_format, OutputFormat::Png);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_confirm_toggles_pref_bool() {
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Configuration;
        app.settings.row_index = find_settings_row(&app, SettingsKey::PrefShowTimeline);
        app.settings_confirm();
        assert!(!app.prefs.show_timeline);
        assert!(app.popup.is_none());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn pref_toggle_persists_to_db_immediately() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.settings.focus = SettingsFocus::Configuration;
            app.settings.row_index = find_settings_row(&app, SettingsKey::PrefReduceMotion);
            app.settings_increment(1);
            // Live-apply: the flip is on disk before quit, like theme
            // changes.
            let loaded = crate::prefs::TuiPrefs::load();
            assert!(loaded.reduce_motion);
        });
    }

    // ── Confirm gate (`tui.confirm_destructive`) ────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn confirm_on_opens_popup_without_dispatching() {
        let mut app = make_settings_test_app();
        app.script.add_stage_after();
        assert_eq!(app.script.script.stages.len(), 2);
        assert!(app.needs_confirm(), "confirmations default to on");
        app.dispatch_action(Action::ScriptDelete);
        assert!(
            matches!(app.popup, Some(Popup::Confirm { .. })),
            "confirm popup must open when the preference is on"
        );
        assert_eq!(app.script.script.stages.len(), 2, "nothing dispatched yet");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn confirm_off_skips_popup_and_dispatches() {
        let mut app = make_settings_test_app();
        app.script.add_stage_after();
        assert_eq!(app.script.script.stages.len(), 2);
        app.prefs.confirm_destructive = false;
        app.dispatch_action(Action::ScriptDelete);
        assert!(
            app.popup.is_none(),
            "confirm-off must not open the Confirm popup"
        );
        assert_eq!(
            app.script.script.stages.len(),
            1,
            "the destructive action must dispatch immediately"
        );
    }

    // ── `tui.default_format` seeding contract ───────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn pref_seeds_format_when_session_has_none() {
        // Boot order: config defaults → pref seed → session overlay.
        // With no saved session format, the pref's choice sticks.
        let prefs = crate::prefs::TuiPrefs {
            default_format: OutputFormat::Jpeg,
            ..Default::default()
        };
        let mut params = GenerateParams::from_config(&Config::default());
        params.format = prefs.default_format;
        let session = crate::session::TuiSession::default();
        session.apply_non_model_params(&mut params);
        assert_eq!(params.format, OutputFormat::Jpeg);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn session_format_still_wins_over_pref_seed() {
        // A saved session / per-model format overlays the pref seed —
        // the preference only decides *fresh* sessions.
        let prefs = crate::prefs::TuiPrefs {
            default_format: OutputFormat::Jpeg,
            ..Default::default()
        };
        let mut params = GenerateParams::from_config(&Config::default());
        params.format = prefs.default_format;
        let session = crate::session::TuiSession {
            format: Some("png".into()),
            ..Default::default()
        };
        session.apply_non_model_params(&mut params);
        assert_eq!(params.format, OutputFormat::Png);
    }

    // ── Settings render landmarks + theme hint (UAT contract) ───────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_render_keeps_panel_titles_and_theme_hint() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = make_settings_test_app();
        let backend = TestBackend::new(100, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                let area = frame.area();
                crate::ui::settings::render(frame, &mut app, area);
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
        // `scripts/tui-uat.sh` landmarks: the two panel titles…
        assert!(out.contains(" Appearance "), "{out}");
        assert!(out.contains(" Configuration "), "{out}");
        // …and the theme hint `theme · <slug>` its theme-set flow greps.
        assert!(out.contains("theme · studio-dark"), "{out}");
        // The Preferences section header renders at the top of the list.
        assert!(out.contains("Preferences"), "{out}");
        // The render recorded the card-grid columns for 2-D navigation:
        // 98 inner cells → 5 columns of 18-wide cards.
        assert_eq!(app.settings.appearance_cols, 5);
    }

    // ── Regression: metadata uses response model, not UI state (#161) ────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn generation_complete_metadata_uses_response_model() {
        // Simulates: user starts generation with model A, then switches UI
        // to model B before generation completes. Metadata must record
        // model A (from the response), not model B (current UI state).
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
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
            gpu: None,
        };
        app.bg_tx
            .send(BackgroundEvent::GenerationComplete {
                response: Box::new(response),
                from_local: false,
            })
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

    // ── Create redesign: layout, accordion behavior, timeline contract ──

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn saved_file_line_appears_in_timeline_after_completion() {
        // The Recent strip is retired — its role is absorbed by the
        // Timeline's `✓ Saved <file>` entries pushed on completion.
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.active_view = View::Create;
            app.generate.generating = true;
            app.generate.batch_remaining = 1;
            app.generate.prompt = TextArea::from(["a timeline test"]);

            let response = GenerateResponse {
                images: vec![mold_core::ImageData {
                    data: vec![0u8; 4],
                    format: OutputFormat::Png,
                    width: 64,
                    height: 64,
                    index: 0,
                }],
                generation_time_ms: 300,
                model: "flux-schnell:q8".to_string(),
                seed_used: 7,
                video: None,
                gpu: None,
            };
            app.bg_tx
                .send(BackgroundEvent::GenerationComplete {
                    response: Box::new(response),
                    from_local: false,
                })
                .unwrap();
            app.process_background_events();

            assert!(
                app.generate
                    .progress
                    .log
                    .iter()
                    .any(|e| e.message.starts_with("Saved ") && e.style == ProgressStyle::Done),
                "completion must push a ✓ Saved <file> timeline entry: {:?}",
                app.generate
                    .progress
                    .log
                    .iter()
                    .map(|e| &e.message)
                    .collect::<Vec<_>>()
            );
        });
    }

    #[tokio::test]
    async fn create_layout_shows_preview_timeline_and_advanced() {
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        let text = render_view_to_string(&mut app, 110, 40);
        assert!(text.contains("\u{250c} Prompt"), "Prompt panel:\n{text}");
        assert!(
            text.contains("\u{250c} Parameters"),
            "Parameters panel:\n{text}"
        );
        assert!(text.contains("\u{250c} Preview"), "Preview panel:\n{text}");
        assert!(
            text.contains("\u{250c} Timeline"),
            "Timeline panel:\n{text}"
        );
        assert!(text.contains("Advanced"), "Advanced header row:\n{text}");
        // The Info sub-panel and Recent strip are retired from Create.
        assert!(
            !text.contains("\u{250c} Info"),
            "Info must be gone:\n{text}"
        );
        assert!(
            !text.contains("\u{250c} Recent"),
            "Recent must be gone:\n{text}"
        );
    }

    #[tokio::test]
    async fn create_layout_hides_timeline_when_pref_off() {
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.show_timeline = false;
        let text = render_view_to_string(&mut app, 110, 40);
        assert!(
            !text.contains("\u{250c} Timeline"),
            "tui.show_timeline=false must hide the panel:\n{text}"
        );
        assert!(text.contains("\u{250c} Preview"), "Preview keeps the row");
    }

    #[tokio::test]
    async fn create_layout_collapses_timeline_below_min_height() {
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        // 22 rows total − 4 chrome rows = 18 content rows, below the
        // 4 (prompt) + 8 (min preview) + 7 (timeline) = 19 budget.
        let text = render_view_to_string(&mut app, 110, 22);
        assert!(
            !text.contains("\u{250c} Timeline"),
            "Timeline must collapse (not squeeze) when short:\n{text}"
        );
    }

    #[tokio::test]
    async fn timeline_empty_state_renders_idle_copy() {
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        let text = render_view_to_string(&mut app, 110, 40);
        assert!(
            text.contains("idle. no runs this session."),
            "fresh session shows the timeline empty state:\n{text}"
        );
    }

    #[tokio::test]
    async fn idle_state_renders_press_enter_hint() {
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        let text = render_view_to_string(&mut app, 110, 40);
        assert!(
            text.contains("Press Enter to generate"),
            "idle preview shows the hint:\n{text}"
        );
        assert!(text.contains("\u{25c7}"), "idle glyph:\n{text}");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn enter_on_section_expands_and_collapses_others() {
        use crate::ui::create_form::{AdvSection, CreateRow};
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        app.dispatch_action(Action::ToggleAdvanced);
        assert!(app.generate.advanced.open);

        let sampling_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == CreateRow::Section(AdvSection::Sampling))
            .unwrap();
        app.generate.param_index = sampling_idx;
        app.dispatch_action(Action::Confirm);
        assert_eq!(app.generate.advanced.expanded, Some(AdvSection::Sampling));
        assert!(app.generate.rows.contains(&CreateRow::SectionField(
            AdvSection::Sampling,
            ParamField::Expand
        )));

        // Expanding another section collapses Sampling.
        let output_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == CreateRow::Section(AdvSection::Output))
            .unwrap();
        app.generate.param_index = output_idx;
        app.dispatch_action(Action::Confirm);
        assert_eq!(app.generate.advanced.expanded, Some(AdvSection::Output));
        assert!(!app
            .generate
            .rows
            .iter()
            .any(|r| matches!(r, CreateRow::SectionField(AdvSection::Sampling, _))));

        // Enter again on the expanded section collapses it.
        let output_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == CreateRow::Section(AdvSection::Output))
            .unwrap();
        app.generate.param_index = output_idx;
        app.dispatch_action(Action::Confirm);
        assert_eq!(app.generate.advanced.expanded, None);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn accordion_dispatch_persists_state_to_db() {
        use crate::ui::create_form::{AdvSection, AdvancedState, CreateRow};
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.active_view = View::Create;
            app.generate.focus = GenerateFocus::Parameters;
            app.dispatch_action(Action::ToggleAdvanced);
            let lora_idx = app
                .generate
                .rows
                .iter()
                .position(|r| *r == CreateRow::Section(AdvSection::Lora))
                .expect("flux2 supports LoRA");
            app.generate.param_index = lora_idx;
            app.dispatch_action(Action::Confirm);

            let loaded = AdvancedState::load();
            assert!(loaded.open, "tui.advanced_open must persist");
            assert_eq!(
                loaded.expanded,
                Some(AdvSection::Lora),
                "tui.advanced_section must persist the slug"
            );
        });
    }

    #[tokio::test]
    async fn size_row_cycles_aspect_presets() {
        use crate::ui::create_form::{size_presets, CreateRow};
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        let size_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == CreateRow::Field(ParamField::Size))
            .unwrap();
        app.generate.param_index = size_idx;

        let (dw, dh) = (app.generate.params.width, app.generate.params.height);
        let presets = size_presets(dw, dh, 64);
        assert_eq!((dw, dh), presets[0], "defaults are the 1:1 preset");

        app.increment_param(1);
        assert_eq!(
            (app.generate.params.width, app.generate.params.height),
            presets[1],
            "◀▶ steps to the 3:2 preset"
        );
        app.increment_param(-1);
        assert_eq!(
            (app.generate.params.width, app.generate.params.height),
            presets[0]
        );
        // A custom size snaps back onto the preset ring.
        app.generate.params.width = 1000;
        app.generate.params.height = 999;
        app.increment_param(1);
        assert_eq!(
            (app.generate.params.width, app.generate.params.height),
            presets[0]
        );
    }

    #[tokio::test]
    async fn seed_row_cycles_mode_with_arrows_and_enter_edits_value() {
        use crate::ui::create_form::CreateRow;
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        let seed_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == CreateRow::Field(ParamField::Seed))
            .unwrap();
        app.generate.param_index = seed_idx;

        assert_eq!(app.generate.params.seed_mode, SeedMode::Random);
        app.increment_param(1);
        assert_eq!(app.generate.params.seed_mode, SeedMode::Fixed);
        assert!(
            app.generate.params.seed.is_some(),
            "entering Fixed pins a seed"
        );
        app.increment_param(-1);
        assert_eq!(app.generate.params.seed_mode, SeedMode::Random);

        // Enter opens the SeedInput popup (the absorbed SeedValue row).
        app.dispatch_action(Action::Confirm);
        assert!(matches!(app.popup, Some(Popup::SeedInput { .. })));
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn upscale_row_picker_sets_and_clears_generate_param() {
        use crate::ui::create_form::{AdvSection, CreateRow};
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        app.dispatch_action(Action::ToggleAdvanced);
        let sec_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == CreateRow::Section(AdvSection::Upscale))
            .unwrap();
        app.generate.param_index = sec_idx;
        app.dispatch_action(Action::Confirm);
        let row_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == CreateRow::SectionField(AdvSection::Upscale, ParamField::Upscale))
            .unwrap();
        app.generate.param_index = row_idx;
        app.dispatch_action(Action::Confirm);

        let Some(Popup::UpscaleModelSelector {
            filtered, purpose, ..
        }) = &app.popup
        else {
            panic!("Enter on the Upscale row must open the picker");
        };
        assert_eq!(*purpose, UpscalePickerPurpose::SetGenerateParam);
        assert_eq!(filtered[0], UPSCALE_OFF_ENTRY);
        let picked = filtered[1].clone();

        // Down + Enter picks the first real model.
        app.handle_crossterm_event(Event::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)));
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        assert_eq!(
            app.generate.params.upscale_model.as_deref(),
            Some(picked.as_str())
        );

        // Re-open and select "(off)" to clear.
        app.generate.param_index = row_idx;
        app.dispatch_action(Action::Confirm);
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.generate.params.upscale_model, None);
    }

    #[tokio::test]
    async fn size_input_popup_applies_wxh() {
        use crate::ui::create_form::CreateRow;
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        let size_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == CreateRow::Field(ParamField::Size))
            .unwrap();
        app.generate.param_index = size_idx;
        app.dispatch_action(Action::Confirm);
        assert!(matches!(app.popup, Some(Popup::SizeInput { .. })));
        // Clear the prefilled WxH and type a new one.
        for _ in 0..12 {
            app.handle_crossterm_event(Event::Key(KeyEvent::new(
                KeyCode::Backspace,
                KeyModifiers::NONE,
            )));
        }
        for c in "1152x832".chars() {
            app.handle_crossterm_event(Event::Key(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.generate.params.width, 1152);
        assert_eq!(app.generate.params.height, 832);
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
            rows: vec![],
            advanced: crate::ui::create_form::AdvancedState::default(),
            param_scroll: 0,
            capabilities: capabilities_for_family("flux"),
            progress: ProgressState::default(),
            preview_image: None,
            image_state: None,
            animation: None,
            generating: true,
            batch_remaining: 3,
            last_seed: None,
            last_generation_time_ms: None,
            error_message: None,
            model_description: String::new(),
            last_output_path: None,
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
            rows: vec![],
            advanced: crate::ui::create_form::AdvancedState::default(),
            param_scroll: 0,
            capabilities: capabilities_for_family("flux"),
            progress: ProgressState::default(),
            preview_image: None,
            image_state: None,
            animation: None,
            generating: true,
            batch_remaining: 4,
            last_seed: None,
            last_generation_time_ms: None,
            error_message: None,
            model_description: String::new(),
            last_output_path: None,
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
            rows: vec![],
            advanced: crate::ui::create_form::AdvancedState::default(),
            param_scroll: 0,
            capabilities: capabilities_for_family("flux"),
            progress: ProgressState::default(),
            preview_image: None,
            image_state: None,
            animation: None,
            generating: false,
            batch_remaining: 0,
            last_seed: None,
            last_generation_time_ms: None,
            error_message: None,
            model_description: String::new(),
            last_output_path: None,
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
    #[serial_test::serial(mold_env)]
    async fn batch_increment_no_upper_cap() {
        let mut app = make_settings_test_app();
        // Switch to Generate view with Parameters focus
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        // Point param_index at the Batch row
        let batch_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == crate::ui::create_form::CreateRow::Field(ParamField::Batch))
            .expect("Batch row should be in the Create form");
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
            purpose: UpscalePickerPurpose::RunNow,
        };
        // Verify the variant can be pattern-matched and fields accessed
        if let Popup::UpscaleModelSelector {
            filter,
            selected,
            filtered,
            purpose,
        } = &popup
        {
            assert!(filter.is_empty());
            assert_eq!(*selected, 0);
            assert_eq!(filtered.len(), 2);
            assert_eq!(*purpose, UpscalePickerPurpose::RunNow);
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
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
        }
    }

    #[test]
    fn remote_catalog_defaults_applied_to_matching_model() {
        // Simulates apply_remote_model_defaults logic
        let mut params = GenerateParams::from_config(&Config::load_or_default());
        params.model = "flux-dev:q4".to_string();
        params.steps = 1;

        let catalog = [make_test_catalog_entry(
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

        let catalog = [make_test_catalog_entry(
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
                backend: None,
            }),
            uptime_secs: 3600,
            hostname: Some("hal9000".to_string()),
            memory_status: Some("VRAM: 16.0 GB free".to_string()),
            gpus: None,
            queue_depth: None,
            queue_capacity: None,
            queue_paused: None,
            instance_id: None,
            models_disk: None,
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
        let mut ri = crate::ui::info::ResourceInfo {
            server_status: Some(mold_core::ServerStatus {
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
                gpus: None,
                queue_depth: None,
                queue_capacity: None,
                queue_paused: None,
                instance_id: None,
                models_disk: None,
            }),
            ..Default::default()
        };
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
            gpus: None,
            queue_depth: None,
            queue_capacity: None,
            queue_paused: None,
            instance_id: None,
            models_disk: None,
        };
        let _event = BackgroundEvent::ServerStatusUpdate(Some(Box::new(status)));
        // None variant for server-unreachable
        let _event_none = BackgroundEvent::ServerStatusUpdate(None);
    }

    // ── should_poll_remote() tests ────────────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn should_poll_remote_true_when_server_and_auto() {
        let mut app = make_settings_test_app();
        app.server_url = Some("http://hal9000:7680".to_string());
        app.generate.params.inference_mode = InferenceMode::Auto;
        assert!(app.should_poll_remote());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn should_poll_remote_true_when_server_and_remote() {
        let mut app = make_settings_test_app();
        app.server_url = Some("http://hal9000:7680".to_string());
        app.generate.params.inference_mode = InferenceMode::Remote;
        assert!(app.should_poll_remote());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn should_poll_remote_false_when_server_but_local_mode() {
        let mut app = make_settings_test_app();
        app.server_url = Some("http://hal9000:7680".to_string());
        app.generate.params.inference_mode = InferenceMode::Local;
        assert!(
            !app.should_poll_remote(),
            "local mode must not poll remote even with server_url set"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn should_poll_remote_false_when_no_server() {
        let mut app = make_settings_test_app();
        app.server_url = None;
        app.generate.params.inference_mode = InferenceMode::Auto;
        assert!(!app.should_poll_remote());
    }

    // ── update_model() remote vs local branching ──────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn update_model_uses_server_catalog_when_connected() {
        // Isolated env — `update_model` now consults `model_prefs` in
        // the metadata DB, so a test that doesn't scope MOLD_HOME could
        // be poisoned by the developer's real `~/.mold/mold.db`.
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.server_url = Some("http://hal9000:7680".to_string());
            app.models.catalog = vec![make_test_catalog_entry(
                "flux-dev:q4",
                28,
                4.0,
                768,
                768,
                "Server FLUX Dev Q4",
            )];

            app.update_model("flux-dev:q4");

            assert_eq!(app.generate.params.steps, 28);
            assert!((app.generate.params.guidance - 4.0).abs() < f64::EPSILON);
            assert_eq!(app.generate.params.width, 768);
            assert_eq!(app.generate.params.height, 768);
            assert_eq!(app.generate.model_description, "Server FLUX Dev Q4");
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn update_model_falls_back_to_local_when_model_not_in_catalog() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.server_url = Some("http://hal9000:7680".to_string());
            // Catalog has a different model with an absurd step count no real model uses
            app.models.catalog = vec![make_test_catalog_entry(
                "flux-schnell:q8",
                199,
                99.9,
                256,
                256,
                "Schnell",
            )];

            // Update to a model NOT in the catalog — should use local config
            let model = app.config.resolved_default_model();
            app.update_model(&model);
            // Should not have used the catalog entry's absurd values
            assert_ne!(app.generate.params.steps, 199);
            assert_ne!(app.generate.params.width, 256);
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn should_save_output_locally_false_when_connected_to_remote_server() {
        // When the TUI is connected to a server in non-Local mode, the
        // server has already saved the output to its own `~/.mold/output/`.
        // A TUI-side write creates a second file with a later timestamp
        // suffix, which surfaces as a duplicate tile on the next gallery
        // rescan (bug reproducer for feat/tui-updates). The predicate must
        // return false so the generation-complete handler skips the write.
        let mut app = make_settings_test_app();
        app.server_url = Some("http://remote.example:7680".to_string());
        app.generate.params.inference_mode = InferenceMode::Remote;
        assert!(!app.should_save_output_locally());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn should_save_output_locally_true_when_no_server() {
        let mut app = make_settings_test_app();
        app.server_url = None;
        app.generate.params.inference_mode = InferenceMode::Local;
        assert!(app.should_save_output_locally());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn should_save_output_locally_true_when_forced_local_even_with_server_url() {
        // User pressed `mold run --local` or toggled the Local mode in the
        // UI. The server exists but we're not using it — TUI owns the save.
        let mut app = make_settings_test_app();
        app.server_url = Some("http://remote.example:7680".to_string());
        app.generate.params.inference_mode = InferenceMode::Local;
        assert!(app.should_save_output_locally());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn should_persist_response_locally_true_for_auto_mode_local_fallback() {
        // Codex finding: in Auto mode, the backend silently falls back to
        // local inference when the connected server becomes unreachable.
        // `server_url` stays set, so `should_save_output_locally()` would
        // return false and the locally-generated image would be dropped.
        // The per-response predicate must honour the `from_local` flag
        // the backend attaches to the completion event.
        let mut app = make_settings_test_app();
        app.server_url = Some("http://remote.example:7680".to_string());
        app.generate.params.inference_mode = InferenceMode::Auto;

        assert!(
            !app.should_save_output_locally(),
            "precondition: in Auto+connected mode the generic predicate treats this as remote"
        );
        assert!(
            app.should_persist_response_locally(true),
            "Auto-mode fallback response must still be saved locally"
        );
        assert!(
            !app.should_persist_response_locally(false),
            "genuine remote success must still skip the local write to avoid duplicates"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn should_persist_response_locally_respects_output_disabled() {
        let mut app = make_settings_test_app();
        app.server_url = None;
        app.generate.params.inference_mode = InferenceMode::Local;
        app.config.output_dir = Some(String::new()); // empty string = disabled
        assert!(app.config.is_output_disabled());

        assert!(
            !app.should_persist_response_locally(true),
            "output disabled wins over from_local — user explicitly opted out of saving"
        );
    }

    /// Gallery-delete test helper: create a real file on disk inside a
    /// per-test subdirectory of the system tempdir and return a
    /// `GalleryEntry` whose `path` points at it. Callers pass a unique
    /// name prefix so parallel tests don't collide.
    fn add_temp_gallery_entry(app: &mut App, name_prefix: &str) -> std::path::PathBuf {
        let tmp = std::env::temp_dir().join(format!("mold-delete-test-{name_prefix}"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join(format!("{name_prefix}.png"));
        // Write a tiny valid PNG header — contents don't matter to delete.
        std::fs::write(&path, b"fake-png-bytes-for-test").unwrap();
        app.gallery.entries.push(GalleryEntry {
            path: path.clone(),
            metadata: make_test_metadata(),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            origins: Vec::new(),
        });
        app.gallery.thumbnail_states.push(None);
        app.gallery.thumb_dimensions.push(None);
        app.gallery.thumb_fixed_cache.push(None);
        app.gallery.selected = app.gallery.entries.len() - 1;
        path
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn delete_selected_gallery_image_empty_gallery_is_noop() {
        let mut app = make_settings_test_app();
        app.gallery.entries.clear();
        // Should not panic, should not touch state.
        app.delete_selected_gallery_image();
        assert!(app.gallery.entries.is_empty());
        assert_eq!(app.gallery.selected, 0);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn delete_selected_gallery_image_out_of_bounds_index_is_noop() {
        let mut app = make_settings_test_app();
        add_temp_gallery_entry(&mut app, "oob");
        // Point selected past the end — must not panic or mutate state.
        app.gallery.selected = 999;
        app.delete_selected_gallery_image();
        assert_eq!(app.gallery.entries.len(), 1);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn delete_selected_gallery_image_removes_local_file_from_disk() {
        // Primary user guarantee: pressing Delete must actually remove the
        // file from disk, not just from the in-memory gallery state.
        let mut app = make_settings_test_app();
        let path = add_temp_gallery_entry(&mut app, "local-file");
        assert!(path.exists(), "precondition: file exists before delete");

        app.delete_selected_gallery_image();

        assert!(!path.exists(), "file should be deleted from disk");
        assert!(app.gallery.entries.is_empty());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn delete_selected_gallery_image_removes_thumbnail_from_disk() {
        let mut app = make_settings_test_app();
        let path = add_temp_gallery_entry(&mut app, "thumb");
        let thumb_path = crate::thumbnails::thumbnail_path(&path);
        if let Some(parent) = thumb_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&thumb_path, b"thumb-bytes").unwrap();
        assert!(thumb_path.exists());

        app.delete_selected_gallery_image();

        assert!(
            !thumb_path.exists(),
            "thumbnail should be deleted from disk"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn delete_in_detail_view_advances_to_next_image_with_preview_loaded() {
        // When a user deletes from Detail (full-screen) view, they expect
        // to land on the next image with the preview pane showing it —
        // not on a blank screen with the deleted file's filename.
        // Prior bug: delete_selected_gallery_image cleared preview_image
        // *after* calling load_gallery_preview, wiping the just-loaded
        // image. Reproducer below decodes a real PNG into the preview
        // and asserts it survives.
        use image::ImageEncoder;

        // Build two real PNGs on disk so load_gallery_preview can decode
        // the surviving entry's image.
        let tmp = std::env::temp_dir().join(format!(
            "mold-detail-delete-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        fn write_real_png(path: &std::path::Path, color: [u8; 3]) {
            let pixels: Vec<u8> = (0..16 * 16).flat_map(|_| color.iter().copied()).collect();
            let f = std::fs::File::create(path).unwrap();
            let encoder = image::codecs::png::PngEncoder::new(f);
            encoder
                .write_image(&pixels, 16, 16, image::ExtendedColorType::Rgb8)
                .unwrap();
        }

        let a_path = tmp.join("a.png");
        let b_path = tmp.join("b.png");
        write_real_png(&a_path, [255, 0, 0]);
        write_real_png(&b_path, [0, 255, 0]);

        let mut app = make_settings_test_app();
        for path in [&a_path, &b_path] {
            app.gallery.entries.push(GalleryEntry {
                path: path.clone(),
                metadata: make_test_metadata(),
                generation_time_ms: None,
                timestamp: 0,
                server_url: None,
                origins: Vec::new(),
            });
            app.gallery.thumbnail_states.push(None);
            app.gallery.thumb_dimensions.push(None);
            app.gallery.thumb_fixed_cache.push(None);
        }
        app.gallery.selected = 0;
        app.gallery.view_mode = GalleryViewMode::Detail;

        app.delete_selected_gallery_image();

        assert_eq!(
            app.gallery.entries.len(),
            1,
            "one entry should remain after delete"
        );
        assert_eq!(
            app.gallery.view_mode,
            GalleryViewMode::Detail,
            "Detail view should persist when there is still an image to show"
        );
        assert!(
            app.gallery.preview_image.is_some(),
            "preview_image must be loaded for the new selection — \
             previously the code cleared it right after load_gallery_preview"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn delete_last_entry_in_detail_view_returns_to_grid() {
        // Deleting the only image in Detail view should drop back to the
        // Grid (where the empty-state banner lives) — not leave the user
        // staring at an empty Detail pane.
        let mut app = make_settings_test_app();
        let _path = add_temp_gallery_entry(&mut app, "lone-entry");
        app.gallery.view_mode = GalleryViewMode::Detail;

        app.delete_selected_gallery_image();

        assert!(app.gallery.entries.is_empty());
        assert_eq!(app.gallery.view_mode, GalleryViewMode::Grid);
        assert!(app.gallery.preview_image.is_none());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn delete_selected_gallery_image_shrinks_parallel_arrays_in_lockstep() {
        // The gallery maintains three parallel vectors alongside `entries`
        // (thumbnail_states, thumb_dimensions, thumb_fixed_cache) — a
        // delete that drops from only `entries` would misalign subsequent
        // thumbnail lookups by one. Two entries → delete selected → all
        // four vectors must end at len 1.
        let mut app = make_settings_test_app();
        add_temp_gallery_entry(&mut app, "lockstep-a");
        add_temp_gallery_entry(&mut app, "lockstep-b");
        app.gallery.selected = 0;

        app.delete_selected_gallery_image();

        assert_eq!(app.gallery.entries.len(), 1);
        assert_eq!(app.gallery.thumbnail_states.len(), 1);
        assert_eq!(app.gallery.thumb_dimensions.len(), 1);
        assert_eq!(app.gallery.thumb_fixed_cache.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    // Serialized: the failure path spawns a gallery rescan whose
    // `db.reconcile` write can outlive the test and race a parallel
    // isolated-env round-trip (the intermittent session/theme/registry
    // CI failures).
    #[serial_test::serial(mold_env)]
    async fn delete_selected_gallery_image_server_entry_emits_failure_on_api_error() {
        // When a gallery entry has `server_url: Some(...)`, the delete
        // must contact the server via `DELETE /api/gallery/image/:name`
        // AND propagate failure back through the background channel so
        // the UI can surface the error (and rescan to re-sync with the
        // server's authoritative list). Previously the spawn was
        // fire-and-forget — silent failures masked 403 / network errors.
        //
        // We point at 127.0.0.1:1 (reserved port) so the connect fails
        // deterministically.
        let mut app = make_settings_test_app();
        let server = "http://127.0.0.1:1".to_string();
        app.server_url = Some(server.clone());

        // Entry mimics what `scan_images_from_server` produces: bare
        // filename path (not absolute), server_url populated.
        app.gallery.entries.push(GalleryEntry {
            path: std::path::PathBuf::from("mold-server-entry.png"),
            metadata: make_test_metadata(),
            generation_time_ms: None,
            timestamp: 0,
            server_url: Some(server),
            origins: Vec::new(),
        });
        app.gallery.thumbnail_states.push(None);
        app.gallery.thumb_dimensions.push(None);
        app.gallery.thumb_fixed_cache.push(None);
        app.gallery.selected = 0;

        app.delete_selected_gallery_image();

        // Drain the bg channel; a GalleryDeleteFailed event must arrive.
        let ev = tokio::time::timeout(std::time::Duration::from_secs(5), app.bg_rx.recv())
            .await
            .expect("delete should emit a background event within 5s")
            .expect("channel was closed");
        assert!(
            matches!(ev, BackgroundEvent::GalleryDeleteFailed(_)),
            "expected GalleryDeleteFailed; the API delete must not be fire-and-forget"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_delete_failure_surfaces_error_and_rescans() {
        // After a server-side delete fails, the UI has already optimistically
        // removed the tile — we need to (a) surface the error so the user
        // knows, and (b) kick off a gallery rescan so the local list
        // re-converges with the server's authoritative state (the entry
        // may still be there).
        let mut app = make_settings_test_app();
        app.server_url = Some("http://server.example:7680".to_string());
        app.generate.error_message = None;
        app.gallery.scanning = false;

        app.apply_delete_failure("forbidden");

        let msg = app.generate.error_message.clone().unwrap_or_default();
        assert!(
            msg.to_lowercase().contains("delete") && msg.to_lowercase().contains("forbidden"),
            "error_message should mention delete + the server's reason, got: {msg:?}"
        );
        assert!(
            app.gallery.scanning,
            "delete failure should trigger a gallery rescan"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_gallery_scan_preserves_selection_by_filename() {
        // Rescans (e.g. after delete failure or reconnect) must not jump the
        // user back to the first image — if the currently-selected entry
        // still exists in the fresh list, its new index wins. Prior
        // behaviour: `selected = 0` unconditionally.
        let mut app = make_settings_test_app();
        app.gallery.entries = vec![
            make_test_entry_with_name("a.png"),
            make_test_entry_with_name("b.png"),
            make_test_entry_with_name("c.png"),
        ];
        app.gallery.thumbnail_states = vec![None; 3];
        app.gallery.thumb_dimensions = vec![None; 3];
        app.gallery.thumb_fixed_cache = vec![None; 3];
        app.gallery.selected = 1; // b.png

        // Fresh scan returns the same filenames in a different order.
        let new_entries = vec![
            make_test_entry_with_name("a.png"),
            make_test_entry_with_name("c.png"),
            make_test_entry_with_name("b.png"), // b moved to index 2
        ];
        app.apply_gallery_scan(new_entries);

        assert_eq!(
            app.gallery.selected, 2,
            "selected should follow b.png to its new index, not reset to 0"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_gallery_scan_clamps_when_previous_filename_is_gone() {
        // The entry we had selected no longer exists (e.g. a successful
        // delete followed by a rescan). Fall back to clamping the old
        // index so the viewport barely shifts — not back to 0.
        let mut app = make_settings_test_app();
        app.gallery.entries = vec![
            make_test_entry_with_name("a.png"),
            make_test_entry_with_name("b.png"),
            make_test_entry_with_name("c.png"),
        ];
        app.gallery.thumbnail_states = vec![None; 3];
        app.gallery.thumb_dimensions = vec![None; 3];
        app.gallery.thumb_fixed_cache = vec![None; 3];
        app.gallery.selected = 1; // b.png

        // Fresh scan returned without b.png — it was really deleted.
        let new_entries = vec![
            make_test_entry_with_name("a.png"),
            make_test_entry_with_name("c.png"),
        ];
        app.apply_gallery_scan(new_entries);

        // Old index 1 clamped to new len-1 = 1, which is c.png — neighbour
        // selection, not a jump back to a.png.
        assert_eq!(app.gallery.selected, 1);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_gallery_scan_empty_list_resets_selected() {
        let mut app = make_settings_test_app();
        app.gallery.entries = vec![make_test_entry_with_name("a.png")];
        app.gallery.thumbnail_states = vec![None];
        app.gallery.thumb_dimensions = vec![None];
        app.gallery.thumb_fixed_cache = vec![None];
        app.gallery.selected = 0;

        app.apply_gallery_scan(Vec::new());

        assert_eq!(app.gallery.selected, 0);
        assert!(app.gallery.entries.is_empty());
    }

    fn make_test_entry_with_name(filename: &str) -> GalleryEntry {
        GalleryEntry {
            path: std::path::PathBuf::from(filename),
            metadata: make_test_metadata(),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            origins: Vec::new(),
        }
    }

    // ── Multi-host Library: filter, rescan triggers, delete routing ──

    #[test]
    fn filter_preserves_thumb_cache_indices() {
        // The thumb caches are parallel vectors keyed by the UNDERLYING
        // entry index. Applying/clearing a filter must never touch them —
        // only the filtered index list changes.
        let mut entry_a = make_test_entry_with_name("sunset.png");
        entry_a.metadata.prompt = "golden sunset".into();
        let mut entry_b = make_test_entry_with_name("cat.png");
        entry_b.metadata.prompt = "a tabby cat".into();
        let mut gallery = GalleryState {
            entries: vec![entry_a, entry_b],
            thumbnail_states: vec![None, None],
            thumb_dimensions: vec![Some((100, 100)), Some((200, 200))],
            thumb_fixed_cache: vec![None, None],
            ..Default::default()
        };
        gallery.refresh_filter();
        assert_eq!(gallery.filtered, vec![0, 1]);

        gallery.filter = "cat".into();
        gallery.refresh_filter();
        assert_eq!(gallery.filtered, vec![1], "only cat.png matches");
        assert_eq!(
            gallery.thumb_dimensions,
            vec![Some((100, 100)), Some((200, 200))],
            "filtering must not invalidate index-keyed thumb caches"
        );
        assert_eq!(gallery.thumbnail_states.len(), 2);
        assert_eq!(gallery.thumb_fixed_cache.len(), 2);
        assert_eq!(
            gallery.selected, 1,
            "selection snaps to the first matching entry"
        );

        gallery.filter.clear();
        gallery.refresh_filter();
        assert_eq!(gallery.filtered, vec![0, 1]);
        assert_eq!(gallery.thumb_dimensions.len(), 2);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn grid_navigation_moves_through_filtered_indices() {
        // With a filter hiding the middle entry, Right must jump from the
        // first match to the next match — not to the hidden neighbour.
        let mut app = make_settings_test_app();
        app.active_view = View::Library;
        let mut a = make_test_entry_with_name("cat-a.png");
        a.metadata.prompt = "cat".into();
        let mut b = make_test_entry_with_name("dog.png");
        b.metadata.prompt = "dog".into();
        let mut c = make_test_entry_with_name("cat-c.png");
        c.metadata.prompt = "cat".into();
        app.gallery.entries = vec![a, b, c];
        app.gallery.thumbnail_states = vec![None; 3];
        app.gallery.thumb_dimensions = vec![None; 3];
        app.gallery.thumb_fixed_cache = vec![None; 3];
        app.gallery.filter = "cat".into();
        app.gallery.refresh_filter();
        assert_eq!(app.gallery.filtered, vec![0, 2]);
        app.gallery.selected = 0;

        app.dispatch_action(Action::GridRight);
        assert_eq!(app.gallery.selected, 2, "Right skips the filtered-out dog");
        app.dispatch_action(Action::GridRight);
        assert_eq!(app.gallery.selected, 2, "clamps at the last match");
        app.dispatch_action(Action::GridLeft);
        assert_eq!(app.gallery.selected, 0);
    }

    #[test]
    fn handle_filter_key_editing_contract() {
        let mut gallery = GalleryState {
            entries: vec![make_test_entry_with_name("cat.png")],
            ..Default::default()
        };
        gallery.refresh_filter();
        gallery.filtering = true;

        assert!(gallery.handle_filter_key(KeyCode::Char('c'), KeyModifiers::NONE));
        assert!(gallery.handle_filter_key(KeyCode::Char('a'), KeyModifiers::NONE));
        assert_eq!(gallery.filter, "ca");
        assert!(gallery.handle_filter_key(KeyCode::Backspace, KeyModifiers::NONE));
        assert_eq!(gallery.filter, "c");

        // Enter confirms — the query stays applied, editing stops.
        assert!(gallery.handle_filter_key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!gallery.filtering);
        assert_eq!(gallery.filter, "c");

        // Esc clears everything.
        gallery.filtering = true;
        assert!(gallery.handle_filter_key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!gallery.filtering);
        assert!(gallery.filter.is_empty());
        assert_eq!(gallery.filtered, vec![0]);

        // Ctrl-modified chars are NOT consumed (Ctrl+C must still quit).
        gallery.filtering = true;
        assert!(!gallery.handle_filter_key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(gallery.filter.is_empty());
    }

    #[test]
    fn rescan_due_contract() {
        let mut gallery = GalleryState::default();
        assert!(gallery.rescan_due(), "never scanned → due");

        gallery.last_scan = Some(std::time::Instant::now());
        assert!(!gallery.rescan_due(), "fresh scan → not due");

        gallery.dirty = true;
        assert!(gallery.rescan_due(), "host registry changed → due");
        gallery.dirty = false;

        if let Some(stale) = std::time::Instant::now()
            .checked_sub(crate::app::LIBRARY_RESCAN_STALE + std::time::Duration::from_secs(1))
        {
            gallery.last_scan = Some(stale);
            assert!(gallery.rescan_due(), "stale scan (>30s) → due");
        }

        gallery.scanning = true;
        assert!(!gallery.rescan_due(), "a running scan is never doubled");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn switching_to_library_rescans_only_when_due() {
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.gallery.last_scan = Some(std::time::Instant::now());
        app.gallery.dirty = false;

        app.dispatch_action(Action::SwitchView(View::Library));
        assert!(
            !app.gallery.scanning,
            "fresh scan → entering Library does not rescan"
        );

        // Host registry changed (connect/forget marks dirty) → rescan.
        app.dispatch_action(Action::SwitchView(View::Create));
        app.gallery.dirty = true;
        app.dispatch_action(Action::SwitchView(View::Library));
        assert!(app.gallery.scanning, "dirty registry → merged rescan");
    }

    #[test]
    fn delete_confirm_message_names_machine_count() {
        assert_eq!(delete_confirm_message("a.png", 1), "Delete a.png?");
        assert_eq!(
            delete_confirm_message("a.png", 2),
            "Delete a.png? It exists on 2 machines. This can't be undone."
        );
    }

    #[test]
    fn owning_origins_synthesizes_legacy_entries() {
        // Legacy entries (empty origins) resolve through server_url: a
        // remote URL becomes a slug-identified remote origin, local stays
        // "this machine".
        let mut entry = make_test_entry_with_name("legacy.png");
        let origins = entry.owning_origins();
        assert_eq!(origins.len(), 1);
        assert!(origins[0].is_local());

        entry.server_url = Some("http://hal9000:7680".into());
        let origins = entry.owning_origins();
        assert_eq!(origins.len(), 1);
        assert_eq!(origins[0].host_id, "hal9000-7680");
        assert_eq!(origins[0].url.as_deref(), Some("http://hal9000:7680"));
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn delete_removes_host_scoped_caches_for_every_owning_host() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            let entry = GalleryEntry {
                path: std::path::PathBuf::from("everywhere.png"),
                metadata: make_test_metadata(),
                generation_time_ms: None,
                timestamp: 0,
                server_url: Some("http://127.0.0.1:1".into()),
                origins: vec![
                    GalleryOrigin {
                        host_id: "host-a".into(),
                        url: Some("http://127.0.0.1:1".into()),
                        name: "host-a".into(),
                    },
                    GalleryOrigin {
                        host_id: "host-b".into(),
                        url: Some("http://127.0.0.1:1".into()),
                        name: "host-b".into(),
                    },
                ],
            };
            let cache_a = crate::gallery_scan::cached_image_path("host-a", "everywhere.png");
            let cache_b = crate::gallery_scan::cached_image_path("host-b", "everywhere.png");
            std::fs::create_dir_all(cache_a.parent().unwrap()).unwrap();
            std::fs::write(&cache_a, b"a").unwrap();
            std::fs::write(&cache_b, b"b").unwrap();

            app.gallery.entries = vec![entry];
            app.gallery.thumbnail_states = vec![None];
            app.gallery.thumb_dimensions = vec![None];
            app.gallery.thumb_fixed_cache = vec![None];
            app.gallery.refresh_filter();
            app.gallery.selected = 0;

            app.delete_selected_gallery_image();

            assert!(app.gallery.entries.is_empty());
            assert!(
                !cache_a.exists() && !cache_b.exists(),
                "delete must clear the host-scoped cache for EVERY owning host"
            );
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn gallery_scan_complete_records_offline_hosts() {
        let mut app = make_settings_test_app();
        app.bg_tx
            .send(BackgroundEvent::GalleryScanComplete(
                crate::gallery_scan::MergedScan {
                    entries: vec![make_test_entry_with_name("a.png")],
                    offline_hosts: 2,
                },
            ))
            .unwrap();
        app.process_background_events();
        assert_eq!(app.gallery.offline_hosts, 2);
        assert_eq!(app.gallery.entries.len(), 1);
        assert_eq!(app.gallery.filtered, vec![0], "scan populates the filter");
        assert!(app.gallery.last_scan.is_some(), "scan stamps freshness");
    }

    #[test]
    fn background_server_command_passes_serve_args() {
        let mut cmd = std::process::Command::new("mold");
        super::configure_background_server_command(&mut cmd, 7680);

        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(
            args.contains(&"serve".to_string()) && args.contains(&"7680".to_string()),
            "serve subcommand and port must still be passed: {args:?}"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_theme_preset_persists_immediately() {
        // `apply_theme_preset` must flush state to the DB right away so a
        // crash/force-quit right after a theme change still restores the
        // latest choice on next launch.
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.apply_theme_preset(crate::ui::theme::ThemePreset::Dracula);

            // Re-read via the public load path — confirms the change
            // actually made it into persistent storage.
            let loaded = crate::session::TuiSession::load();
            assert_eq!(
                loaded.theme.as_deref(),
                Some("dracula"),
                "apply_theme_preset should have persisted the theme immediately"
            );
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_theme_preset_supports_every_preset() {
        // `apply_theme_preset` persists through the process-global DB env.
        // Serialize this in-memory coverage test so it cannot write into an
        // isolated persistence test's temporary database.
        for preset in crate::ui::theme::ThemePreset::ALL {
            let mut app = make_settings_test_app();
            app.apply_theme_preset(preset);
            assert_eq!(app.settings.theme_preset, preset);
            assert_eq!(app.theme.accent, preset.build().accent);
        }
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn update_model_restores_per_model_saved_params() {
        // Marquee behavior: FLUX remembers its own (width, steps, guidance);
        // SDXL remembers its own. Switching away and back restores each
        // model's saved settings, not fresh defaults.
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();

            // Set up two models with distinct manifest-resolved config
            // entries so resolved_model_config returns meaningful defaults.
            app.config.models.insert(
                "flux-dev:q4".to_string(),
                mold_core::config::ModelConfig {
                    default_steps: Some(20),
                    default_guidance: Some(3.5),
                    default_width: Some(1024),
                    default_height: Some(1024),
                    ..Default::default()
                },
            );
            app.config.models.insert(
                "sdxl:fp16".to_string(),
                mold_core::config::ModelConfig {
                    default_steps: Some(30),
                    default_guidance: Some(7.5),
                    default_width: Some(768),
                    default_height: Some(768),
                    ..Default::default()
                },
            );

            // Start on FLUX and bump width to a non-default value.
            app.update_model("flux-dev:q4");
            app.generate.params.width = 1024;
            app.generate.params.height = 1024;
            app.generate.params.steps = 20;
            app.generate.params.guidance = 3.5;

            // Switch to SDXL — FLUX's current params must get snapshotted
            // now, before we clobber them.
            app.update_model("sdxl:fp16");
            // SDXL should get SDXL's config defaults, not FLUX's values.
            assert_eq!(app.generate.params.width, 768);
            assert_eq!(app.generate.params.steps, 30);
            assert_eq!(app.generate.params.guidance, 7.5);

            // Tweak SDXL to a weird value so we can tell its snapshot apart.
            app.generate.params.width = 512;
            app.generate.params.steps = 15;

            // Switch back to FLUX — must come back at the FLUX values, not
            // the config defaults (which are also 1024x1024 here, so use
            // steps=20 / guidance=3.5 as the real signal).
            app.update_model("flux-dev:q4");
            assert_eq!(app.generate.params.width, 1024);
            assert_eq!(app.generate.params.steps, 20);
            assert!((app.generate.params.guidance - 3.5).abs() < 1e-9);

            // And one more flip: SDXL must restore 512x512 / steps=15.
            app.update_model("sdxl:fp16");
            assert_eq!(app.generate.params.width, 512);
            assert_eq!(app.generate.params.steps, 15);
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn update_model_to_same_model_is_noop() {
        // Defensive — same-model calls shouldn't overwrite live params with
        // a stale DB snapshot.
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.config.models.insert(
                "flux-dev:q4".to_string(),
                mold_core::config::ModelConfig::default(),
            );
            app.update_model("flux-dev:q4");
            app.generate.params.width = 777;
            app.update_model("flux-dev:q4");
            assert_eq!(app.generate.params.width, 777);
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn theme_default_is_studio_dark_when_session_is_missing() {
        // The default theme must always be Studio Dark — the other presets
        // should only appear when explicitly selected.
        crate::test_env::with_isolated_env(|_home| {
            let loaded = crate::session::TuiSession::load();
            let resolved = loaded
                .theme
                .as_deref()
                .map(crate::ui::theme::ThemePreset::from_slug)
                .unwrap_or_default();
            assert_eq!(
                resolved,
                crate::ui::theme::ThemePreset::StudioDark,
                "missing session must resolve to Studio Dark"
            );
            assert!(
                loaded.theme.is_none(),
                "no theme key should be present in a fresh session"
            );
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn theme_default_is_studio_dark_when_slug_is_unknown_or_empty() {
        // An old session file or a hand-edited config could carry a
        // garbage slug — it must fall back to the Studio Dark default.
        assert_eq!(
            crate::ui::theme::ThemePreset::from_slug(""),
            crate::ui::theme::ThemePreset::StudioDark
        );
        assert_eq!(
            crate::ui::theme::ThemePreset::from_slug("not-a-real-theme"),
            crate::ui::theme::ThemePreset::StudioDark
        );
        assert_eq!(
            crate::ui::theme::ThemePreset::default(),
            crate::ui::theme::ThemePreset::StudioDark
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_theme_preset_persists_across_multiple_changes() {
        // Rapidly cycling themes should always leave the *latest* choice
        // on disk — an earlier save must not be skipped because
        // something thinks the preset hasn't changed.
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.apply_theme_preset(crate::ui::theme::ThemePreset::Dracula);
            app.apply_theme_preset(crate::ui::theme::ThemePreset::Nord);
            app.apply_theme_preset(crate::ui::theme::ThemePreset::Gruvbox);

            let loaded = crate::session::TuiSession::load();
            assert_eq!(
                loaded.theme.as_deref(),
                Some("gruvbox"),
                "latest theme (gruvbox) should be the persisted slug"
            );
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_delete_failure_no_rescan_when_not_connected_to_server() {
        // In pure-local mode there's no server to re-scan against — just
        // surface the error without firing a rescan that would fail
        // anyway.
        let mut app = make_settings_test_app();
        app.server_url = None;
        app.apply_delete_failure("permission denied");

        assert!(app.generate.error_message.is_some());
        assert!(
            !app.gallery.scanning,
            "no rescan should be kicked off when there is no server to rescan from"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn update_model_uses_local_config_when_no_server() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.server_url = None;
            let model = app.config.resolved_default_model();
            app.update_model(&model);
            // Should succeed without panic and use local config defaults
            assert!(app.generate.params.steps > 0);
            assert!(app.generate.params.width > 0);
        });
    }

    // ── apply_remote_model_defaults() ─────────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_remote_model_defaults_updates_all_fields() {
        let mut app = make_settings_test_app();
        app.generate.params.model = "flux-dev:q4".to_string();
        app.generate.params.steps = 1;
        app.generate.params.guidance = 0.0;
        app.generate.params.width = 64;
        app.generate.params.height = 64;

        let catalog = vec![make_test_catalog_entry(
            "flux-dev:q4",
            20,
            3.5,
            1024,
            1024,
            "FLUX Dev Q4",
        )];
        app.apply_remote_model_defaults(&catalog);

        assert_eq!(app.generate.params.steps, 20);
        assert!((app.generate.params.guidance - 3.5).abs() < f64::EPSILON);
        assert_eq!(app.generate.params.width, 1024);
        assert_eq!(app.generate.params.height, 1024);
        assert_eq!(app.generate.model_description, "FLUX Dev Q4");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_remote_model_defaults_skips_empty_description() {
        let mut app = make_settings_test_app();
        app.generate.params.model = "flux-dev:q4".to_string();
        app.generate.model_description = "Original description".to_string();

        let catalog = vec![make_test_catalog_entry(
            "flux-dev:q4",
            20,
            3.5,
            1024,
            1024,
            "", // empty description should not overwrite
        )];
        app.apply_remote_model_defaults(&catalog);

        assert_eq!(app.generate.model_description, "Original description");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn apply_remote_model_defaults_no_match_leaves_params_unchanged() {
        let mut app = make_settings_test_app();
        app.generate.params.model = "nonexistent:q4".to_string();
        app.generate.params.steps = 42;

        let catalog = vec![make_test_catalog_entry(
            "flux-dev:q4",
            20,
            3.5,
            1024,
            1024,
            "FLUX",
        )];
        app.apply_remote_model_defaults(&catalog);

        assert_eq!(
            app.generate.params.steps, 42,
            "should not change for non-matching model"
        );
    }

    // ── ResetDefaults branching ───────────────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn reset_defaults_uses_server_catalog_when_connected() {
        let mut app = make_settings_test_app();
        app.server_url = Some("http://hal9000:7680".to_string());
        app.generate.params.model = "flux-dev:q4".to_string();
        app.models.catalog = vec![make_test_catalog_entry(
            "flux-dev:q4",
            30,
            7.0,
            512,
            512,
            "Server Flux",
        )];

        // Mutate params away from defaults
        app.generate.params.steps = 1;
        app.generate.params.width = 9999;
        app.generate.params.batch = 5;
        app.generate.params.format = OutputFormat::Jpeg;

        // Focus on parameters, select ResetDefaults, and trigger it
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        let reset_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == crate::ui::create_form::CreateRow::ResetDefaults)
            .unwrap();
        app.generate.param_index = reset_idx;
        app.activate_current_param();

        // Server catalog defaults should be applied
        assert_eq!(app.generate.params.steps, 30);
        assert!((app.generate.params.guidance - 7.0).abs() < f64::EPSILON);
        assert_eq!(app.generate.params.width, 512);
        assert_eq!(app.generate.params.height, 512);
        // Non-default fields should be reset to generic defaults
        assert_eq!(app.generate.params.batch, 1);
        assert_eq!(app.generate.params.format, OutputFormat::Png);
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn reset_defaults_uses_local_config_when_no_server() {
        let mut app = make_settings_test_app();
        app.server_url = None;

        // Mutate params
        app.generate.params.steps = 999;
        app.generate.params.batch = 10;

        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        let reset_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == crate::ui::create_form::CreateRow::ResetDefaults)
            .unwrap();
        app.generate.param_index = reset_idx;
        app.activate_current_param();

        // Should use local config defaults (steps won't be 999)
        assert_ne!(app.generate.params.steps, 999);
        assert_eq!(app.generate.params.batch, 1);
    }

    // ── sync_resource_info_mode() ─────────────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn sync_resource_info_mode_local_clears_server_status() {
        let mut app = make_settings_test_app();
        app.generate.params.inference_mode = InferenceMode::Local;
        // Simulate having stale server status
        app.resource_info.server_status = Some(mold_core::ServerStatus {
            version: "0.6.3".to_string(),
            git_sha: None,
            build_date: None,
            models_loaded: vec![],
            busy: false,
            current_generation: None,
            gpu_info: None,
            uptime_secs: 0,
            hostname: Some("stale-host".to_string()),
            memory_status: None,
            gpus: None,
            queue_depth: None,
            queue_capacity: None,
            queue_paused: None,
            instance_id: None,
            models_disk: None,
        });

        app.sync_resource_info_mode();

        assert!(
            app.resource_info.server_status.is_none(),
            "local mode should clear server_status"
        );
    }

    // ── ServerConnected handler ───────────────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn server_connected_applies_model_defaults_and_clears_connecting() {
        let mut app = make_settings_test_app();
        app.connecting = true;
        app.generate.params.model = "flux-dev:q4".to_string();
        app.generate.params.steps = 1;

        let models = vec![make_test_catalog_entry(
            "flux-dev:q4",
            20,
            3.5,
            1024,
            1024,
            "Server FLUX",
        )];

        // Simulate receiving ServerConnected
        let _ = app.bg_tx.send(BackgroundEvent::ServerConnected {
            url: "http://hal9000:7680".to_string(),
            models,
        });
        app.process_background_events();

        assert!(!app.connecting);
        assert_eq!(app.server_url.as_deref(), Some("http://hal9000:7680"));
        assert_eq!(app.generate.params.steps, 20);
    }

    // ── ServerStatusUpdate handlers ───────────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn server_status_update_some_populates_resource_info() {
        let mut app = make_settings_test_app();
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
                backend: None,
            }),
            uptime_secs: 3600,
            hostname: Some("hal9000".to_string()),
            memory_status: Some("VRAM: 16.0 GB free".to_string()),
            gpus: None,
            queue_depth: None,
            queue_capacity: None,
            queue_paused: None,
            instance_id: None,
            models_disk: None,
        };

        let _ = app
            .bg_tx
            .send(BackgroundEvent::ServerStatusUpdate(Some(Box::new(status))));
        app.process_background_events();

        let ri = &app.resource_info;
        assert!(ri.server_status.is_some());
        assert_eq!(
            ri.server_status.as_ref().unwrap().hostname.as_deref(),
            Some("hal9000")
        );
        assert_eq!(ri.memory_line.as_deref(), Some("VRAM: 16.0 GB free"));
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn server_status_update_none_clears_stale_status() {
        let mut app = make_settings_test_app();
        // Pre-populate with server status
        app.resource_info
            .update_from_server_status(mold_core::ServerStatus {
                version: "0.6.3".to_string(),
                git_sha: None,
                build_date: None,
                models_loaded: vec![],
                busy: false,
                current_generation: None,
                gpu_info: None,
                uptime_secs: 0,
                hostname: Some("stale-host".to_string()),
                memory_status: Some("VRAM: 16.0 GB free".to_string()),
                gpus: None,
                queue_depth: None,
                queue_capacity: None,
                queue_paused: None,
                instance_id: None,
                models_disk: None,
            });
        assert!(app.resource_info.server_status.is_some());

        // Server went down — receive None
        let _ = app.bg_tx.send(BackgroundEvent::ServerStatusUpdate(None));
        app.process_background_events();

        assert!(
            app.resource_info.server_status.is_none(),
            "stale server status should be cleared on fetch failure"
        );
    }

    // ── ServerUnreachable handler ─────────────────────────────

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn server_unreachable_clears_connecting_and_reverts_host() {
        let mut app = make_settings_test_app();
        app.connecting = true;
        app.server_url = Some("http://original:7680".to_string());
        app.generate.params.host = Some("http://new-host:7680".to_string());

        let _ = app
            .bg_tx
            .send(BackgroundEvent::ServerUnreachable("timeout".to_string()));
        app.process_background_events();

        assert!(!app.connecting);
        // host should revert to server_url
        assert_eq!(
            app.generate.params.host.as_deref(),
            Some("http://original:7680")
        );
        assert!(app.resource_info.server_status.is_none());
    }

    // ── Settings view render coverage ────────────────────────
    //
    // These tests exercise `crate::ui::settings::render` against real
    // `App` fixtures so the per-row branch matrix in `build_settings_line`
    // and the focus / scroll / save_error branches in `render_configuration`
    // are exercised. Coverage attribution lands on `ui/settings.rs` even
    // though the `#[test]` lives here — we need the App fixture from this
    // module's existing helper.

    #[tokio::test]
    async fn settings_render_appearance_focus() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Appearance;
        let backend = TestBackend::new(80, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| crate::ui::settings::render(f, &mut app, f.area()))
            .unwrap();
    }

    #[tokio::test]
    async fn settings_render_configuration_focus_with_selection() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Configuration;
        // Land on a known field so the selected-line + suffix glyphs render.
        app.settings.row_index = find_settings_row(&app, SettingsKey::DefaultWidth);
        let backend = TestBackend::new(80, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| crate::ui::settings::render(f, &mut app, f.area()))
            .unwrap();
        // Selected row stays on-screen.
        assert!(app.settings.scroll_offset <= app.settings.row_index);
    }

    #[tokio::test]
    async fn settings_render_text_field_selected_emits_dropdown_glyph() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Configuration;
        // DefaultModel is a Text field — the suffix is the down-triangle.
        app.settings.row_index = find_settings_row(&app, SettingsKey::DefaultModel);
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| crate::ui::settings::render(f, &mut app, f.area()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let mut found_glyph = false;
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                if buf[(x, y)].symbol() == "\u{25bc}" {
                    found_glyph = true;
                    break;
                }
            }
        }
        assert!(found_glyph, "expected ▼ on the selected Text field's row",);
    }

    #[tokio::test]
    async fn settings_render_with_save_error_drives_error_branch() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Configuration;
        app.settings.save_error = Some("disk full".to_string());
        // Make the area generous enough to fit the entire settings list
        // plus the save_error line so the error styling branch runs end
        // to end (the error line is the last entry in `lines` and
        // wouldn't render in a small viewport).
        let backend = TestBackend::new(100, 80);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| crate::ui::settings::render(f, &mut app, f.area()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let mut full_text = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                full_text.push_str(buf[(x, y)].symbol());
            }
            full_text.push('\n');
        }
        assert!(
            full_text.contains("disk full"),
            "save_error message must appear in rendered output",
        );
    }

    #[tokio::test]
    async fn settings_render_zero_height_inner_returns_early() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        // Tiny area: APPEARANCE_HEIGHT (5) plus Min(5) → inner of
        // Configuration collapses to zero. The render path early-returns
        // without painting.
        let backend = TestBackend::new(40, 4);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| crate::ui::settings::render(f, &mut app, f.area()))
            .unwrap();
    }

    #[tokio::test]
    async fn settings_render_tall_list_overflows_scrollbar() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        app.settings.focus = SettingsFocus::Configuration;
        // Inner height < total rows → scrollbar branch fires.
        // Use a small window so any non-trivial settings list overflows.
        let backend = TestBackend::new(80, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| crate::ui::settings::render(f, &mut app, f.area()))
            .unwrap();
        // Drive the scroll-offset code path by selecting a far-down field.
        let last_row_index = app.build_settings_rows().len().saturating_sub(1);
        app.settings.row_index = last_row_index;
        terminal
            .draw(|f| crate::ui::settings::render(f, &mut app, f.area()))
            .unwrap();
        assert!(
            app.settings.scroll_offset > 0,
            "scrolling far down must shift the visible window",
        );
    }

    // ── Models view render coverage ──────────────────────────
    //
    // Same idea as the settings tests above — we drive every branch of
    // `crate::ui::models::render` (empty-catalog, populated-catalog,
    // installed/available split, loaded/ready/empty status markers,
    // zero-height inspector early returns) so that `ui/models.rs` lights
    // up under llvm-cov. The fixture comes from `make_settings_test_app()`
    // because every test helper here already pays the cost of building it.

    fn synth_model(
        name: &str,
        family: &str,
        downloaded: bool,
        is_loaded: bool,
    ) -> mold_core::ModelInfoExtended {
        use mold_core::types::{ModelDefaults, ModelInfo, ModelInfoExtended};
        ModelInfoExtended {
            info: ModelInfo {
                name: name.to_string(),
                family: family.to_string(),
                size_gb: 4.2,
                is_loaded,
                last_used: None,
                hf_repo: format!("test-org/{name}"),
            },
            defaults: ModelDefaults {
                default_steps: 4,
                default_guidance: 3.5,
                default_width: 1024,
                default_height: 1024,
                description: format!("synthetic {name} fixture"),
            },
            downloaded,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
        }
    }

    #[tokio::test]
    async fn models_render_empty_catalog_shows_no_matches_in_details() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        app.active_view = View::Models;
        app.models.catalog.clear();
        let backend = TestBackend::new(120, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| crate::ui::models::render(f, &mut app, f.area()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let mut text = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                text.push_str(buf[(x, y)].symbol());
            }
            text.push(' ');
        }
        assert!(
            text.contains("no matches"),
            "empty catalog must surface the 'no matches' empty-state",
        );
    }

    #[tokio::test]
    async fn models_render_installed_section_active_with_loaded_model() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        app.active_view = View::Models;
        // Two installed models — first is currently loaded.
        app.models.catalog = vec![
            synth_model("flux-dev:q8", "flux", true, true),
            synth_model("sdxl-base:fp16", "sdxl", true, false),
        ];
        app.models.selected = 0;
        let backend = TestBackend::new(140, 32);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| crate::ui::models::render(f, &mut app, f.area()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let mut text = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                text.push_str(buf[(x, y)].symbol());
            }
            text.push(' ');
        }
        assert!(text.contains("loaded"), "loaded status must render");
        assert!(text.contains("FLUX"), "family is uppercased in the row");
        assert!(
            text.contains("flux-dev:q8"),
            "installed model name must appear",
        );
        // Inspector renders the selected model's HF repo.
        assert!(
            text.contains("test-org/flux-dev:q8"),
            "details panel must surface the selected model's HF repo",
        );
    }

    #[tokio::test]
    async fn models_render_available_section_active_with_undownloaded_model() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        app.active_view = View::Models;
        // One installed, one not installed — selecting index 1 (the first
        // entry past the installed length) flips the active section to
        // "Available" and exercises the wrapping_sub() path.
        app.models.catalog = vec![
            synth_model("flux-dev:q8", "flux", true, false),
            synth_model("z-image:fp16", "z-image", false, false),
        ];
        app.models.selected = 1;
        let backend = TestBackend::new(140, 32);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| crate::ui::models::render(f, &mut app, f.area()))
            .unwrap();
        let buf = terminal.backend().buffer();
        let mut text = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                text.push_str(buf[(x, y)].symbol());
            }
            text.push(' ');
        }
        assert!(
            text.contains("ready"),
            "downloaded-but-not-loaded → 'ready'"
        );
        assert!(
            text.contains("z-image:fp16"),
            "the non-installed model must list under Available",
        );
    }

    #[tokio::test]
    async fn models_render_zero_width_collapses_panels_without_panic() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = make_settings_test_app();
        app.active_view = View::Models;
        app.models.catalog = vec![synth_model("flux-dev:q8", "flux", true, false)];
        // Pathological but reachable: a 1-wide window collapses the Details
        // panel inner to zero, which must early-return rather than panic.
        let backend = TestBackend::new(1, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| crate::ui::models::render(f, &mut app, f.area()))
            .unwrap();
    }

    // ── Machines workspace ────────────────────────────────────────

    fn machines_test_host(id: &str) -> crate::hosts::HostEntry {
        crate::hosts::HostEntry {
            id: id.to_string(),
            url: format!("http://{id}:7680"),
            name: Some(id.to_string()),
            instance_id: None,
        }
    }

    fn render_view_to_string(app: &mut App, width: u16, height: u16) -> String {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| crate::ui::render(f, app)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[tokio::test]
    async fn machines_render_shows_machines_landmark() {
        // `scripts/tui-uat.sh view machines` greps for "┌ Machines" — this
        // is the landmark regression test for that contract.
        let mut app = make_settings_test_app();
        app.active_view = View::Machines;
        app.machines
            .registry
            .add(machines_test_host("hal9000"))
            .unwrap();
        let text = render_view_to_string(&mut app, 100, 30);
        assert!(text.contains("┌ Machines"), "landmark missing:\n{text}");
        assert!(text.contains("hal9000"), "host row missing:\n{text}");
        assert!(
            text.contains("+ Connect a machine"),
            "connect affordance missing:\n{text}"
        );
        // The local machine is always the first row.
        let local = crate::hosts::local_display_name();
        let local_pos = text.find(local).expect("local row rendered");
        let host_pos = text.find("hal9000").unwrap();
        assert!(local_pos < host_pos, "local row must render first");
    }

    #[tokio::test]
    async fn machines_status_shortcuts_advertise_the_key_map() {
        let mut app = make_settings_test_app();
        app.active_view = View::Machines;
        let hints = crate::ui::status_shortcuts(&app);
        let keys: Vec<&str> = hints.iter().map(|(k, _)| k.as_str()).collect();
        for expected in ["^K", "j/k", "Enter", "c", "d", "r", "Esc"] {
            assert!(keys.contains(&expected), "missing {expected} in {keys:?}");
        }
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn machines_set_target_is_sticky_and_toggles_back_to_auto() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.active_view = View::Machines;
            app.machines
                .registry
                .add(machines_test_host("hal9000"))
                .unwrap();
            app.machines.select_next(); // select the host row
            app.dispatch_action(Action::MachinesSetTarget);
            assert_eq!(app.target, crate::hosts::GenTarget::Host("hal9000".into()));
            assert_eq!(
                crate::hosts::GenTarget::load(),
                crate::hosts::GenTarget::Host("hal9000".into()),
                "target must persist"
            );
            // Enter on the same row reverts to Auto (escapable sticky pick).
            app.dispatch_action(Action::MachinesSetTarget);
            assert_eq!(app.target, crate::hosts::GenTarget::Auto);
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn machines_forget_confirm_names_key_deletion_and_resets_target() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.active_view = View::Machines;
            app.machines
                .registry
                .add(machines_test_host("hal9000"))
                .unwrap();
            app.machines.select_next();
            app.target = crate::hosts::GenTarget::Host("hal9000".into());

            app.dispatch_action(Action::MachinesForget);
            let Some(Popup::Confirm {
                message,
                on_confirm,
            }) = &app.popup
            else {
                panic!("expected the Forget confirm popup");
            };
            assert!(
                message.contains("hal9000") && message.contains("API key is deleted"),
                "confirm copy must name the host and the key deletion: {message}"
            );
            let action = on_confirm.clone();
            app.popup = None;
            app.handle_confirm_action(action);
            assert!(app.machines.registry.hosts.is_empty());
            assert_eq!(
                app.target,
                crate::hosts::GenTarget::Auto,
                "a forgotten target host must fall back to Auto"
            );
        });
    }

    #[tokio::test]
    async fn machines_forget_on_local_row_is_a_noop() {
        let mut app = make_settings_test_app();
        app.active_view = View::Machines;
        assert_eq!(app.machines.selected, 0);
        app.dispatch_action(Action::MachinesForget);
        assert!(app.popup.is_none(), "the local machine cannot be forgotten");
    }

    #[tokio::test]
    async fn machines_connect_action_opens_stepped_popup() {
        let mut app = make_settings_test_app();
        // Reachable from any view via the ^K palette entry.
        app.dispatch_action(Action::MachinesConnect);
        assert_eq!(app.active_view, View::Machines);
        assert!(matches!(app.popup, Some(Popup::MachineConnect { .. })));

        // Typing routes into the form; Enter normalizes and advances.
        use crossterm::event::{Event, KeyEvent};
        for c in "hal9000".chars() {
            app.handle_crossterm_event(Event::Key(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        let Some(Popup::MachineConnect { form }) = &app.popup else {
            panic!("popup must stay open through the ApiKey step");
        };
        assert_eq!(form.step, crate::hosts::ConnectStep::ApiKey);
        assert_eq!(form.url, "http://hal9000:7680");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn machines_connect_tested_success_registers_host() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.dispatch_action(Action::MachinesConnect);
            if let Some(Popup::MachineConnect { form }) = &mut app.popup {
                form.step = crate::hosts::ConnectStep::Testing;
                form.url = "http://hal9000:7680".to_string();
            }
            let status = ServerStatus {
                version: "0.20.2".into(),
                git_sha: None,
                build_date: None,
                models_loaded: vec![],
                busy: false,
                current_generation: None,
                gpu_info: None,
                uptime_secs: 1,
                hostname: Some("hal9000".into()),
                memory_status: None,
                gpus: None,
                queue_depth: None,
                queue_capacity: None,
                queue_paused: None,
                instance_id: Some("uuid-a".into()),
                models_disk: None,
            };
            let _ = app.bg_tx.send(BackgroundEvent::MachineConnectTested {
                url: "http://hal9000:7680".into(),
                api_key: Some("sekrit".into()),
                result: Ok(Box::new(status)),
            });
            app.process_background_events();

            assert!(app.popup.is_none(), "popup closes on success");
            let entry = app
                .machines
                .registry
                .get("hal9000-7680")
                .expect("registered");
            assert_eq!(entry.name.as_deref(), Some("hal9000"));
            assert_eq!(
                crate::hosts::api_key_for("hal9000-7680").as_deref(),
                Some("sekrit")
            );
        });
    }

    #[tokio::test]
    async fn machines_connect_tested_failure_shows_retryable_error() {
        let mut app = make_settings_test_app();
        app.dispatch_action(Action::MachinesConnect);
        if let Some(Popup::MachineConnect { form }) = &mut app.popup {
            form.step = crate::hosts::ConnectStep::Testing;
            form.url = "http://down:7680".to_string();
        }
        let _ = app.bg_tx.send(BackgroundEvent::MachineConnectTested {
            url: "http://down:7680".into(),
            api_key: None,
            result: Err("connection refused".into()),
        });
        app.process_background_events();
        let Some(Popup::MachineConnect { form }) = &app.popup else {
            panic!("popup stays open in the Failed step");
        };
        assert_eq!(form.step, crate::hosts::ConnectStep::Failed);
        assert_eq!(form.error.as_deref(), Some("connection refused"));
    }

    #[tokio::test]
    async fn host_status_and_queue_events_feed_machines_state() {
        let mut app = make_settings_test_app();
        app.machines
            .registry
            .add(machines_test_host("hal9000"))
            .unwrap();
        app.machines.select_next();

        let _ = app.bg_tx.send(BackgroundEvent::HostStatusUpdate {
            host_id: "hal9000".into(),
            status: None,
        });
        let _ = app.bg_tx.send(BackgroundEvent::HostQueueUpdate {
            host_id: "hal9000".into(),
            queue: Some(mold_core::QueueListingWire::default()),
        });
        app.process_background_events();

        assert!(matches!(
            app.machines.statuses.get("hal9000").unwrap().health,
            crate::hosts::HostHealth::Offline(_)
        ));
        assert!(app.machines.queue.is_some());
    }

    #[tokio::test]
    async fn start_generation_with_stale_host_target_errors_with_fix() {
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.prompt = TextArea::new(vec!["a cat".to_string()]);
        app.target = crate::hosts::GenTarget::Host("gone-host".into());
        app.dispatch_action(Action::Generate);
        assert!(
            !app.generate.generating,
            "stale target must not start a run"
        );
        let err = app.generate.error_message.as_deref().unwrap_or_default();
        assert!(
            err.contains("gone-host") && err.contains("Machines"),
            "error must name the host and point at Machines: {err}"
        );
    }

    #[tokio::test]
    async fn machines_cancel_job_gated_on_detail_focus_and_queued_state() {
        let mut app = make_settings_test_app();
        app.active_view = View::Machines;
        app.machines
            .registry
            .add(machines_test_host("hal9000"))
            .unwrap();
        app.machines.select_next();
        app.machines.apply_queue(
            "hal9000".into(),
            Some(mold_core::QueueListingWire {
                entries: vec![
                    mold_core::QueueJobEntryWire {
                        id: "job-run".into(),
                        model: "flux2-klein:q8".into(),
                        state: "running".into(),
                        started_at_unix_ms: 0,
                        position: 0,
                        gpu: Some(0),
                        target_gpu: None,
                        seed_pinned: None,
                        metadata: None,
                    },
                    mold_core::QueueJobEntryWire {
                        id: "job-q".into(),
                        model: "sdxl:fp16".into(),
                        state: "queued".into(),
                        started_at_unix_ms: 0,
                        position: 1,
                        gpu: None,
                        target_gpu: None,
                        seed_pinned: None,
                        metadata: None,
                    },
                ],
            }),
        );

        // HostList focus: `x` is a no-op.
        app.dispatch_action(Action::MachinesCancelJob);
        assert!(app.popup.is_none());

        // Detail focus on the running job: still a no-op (not cancelable).
        app.machines.focus = crate::hosts::MachinesFocus::Detail;
        app.machines.queue_selected = 0;
        app.dispatch_action(Action::MachinesCancelJob);
        assert!(app.popup.is_none(), "running jobs are not cancelable");

        // Detail focus on the queued job: confirm gate opens.
        app.machines.queue_selected = 1;
        app.dispatch_action(Action::MachinesCancelJob);
        let Some(Popup::Confirm { on_confirm, .. }) = &app.popup else {
            panic!("expected the cancel confirm popup");
        };
        assert!(matches!(
            on_confirm,
            ConfirmAction::CancelHostJob { host_id, job_id }
                if host_id == "hal9000" && job_id == "job-q"
        ));
    }

    #[tokio::test]
    async fn machines_tab_toggles_focus_and_jk_routes_by_focus() {
        let mut app = make_settings_test_app();
        app.active_view = View::Machines;
        app.machines
            .registry
            .add(machines_test_host("hal9000"))
            .unwrap();
        assert_eq!(app.machines.focus, crate::hosts::MachinesFocus::HostList);
        app.dispatch_action(Action::FocusNext);
        assert_eq!(app.machines.focus, crate::hosts::MachinesFocus::Detail);

        // Detail focus: Down moves the queue selection, not the host row.
        app.machines.apply_queue("hal9000".into(), None);
        let before = app.machines.selected;
        app.dispatch_action(Action::Down);
        assert_eq!(app.machines.selected, before);
        app.dispatch_action(Action::FocusPrev);
        assert_eq!(app.machines.focus, crate::hosts::MachinesFocus::HostList);
    }
}
