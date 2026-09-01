use anyhow::{Context, Result};
use crossterm::event::{Event as CrosstermEvent, KeyCode, KeyModifiers};
use mold_core::{
    validation::{MAX_STG_BLOCKS, MAX_STG_BLOCK_INDEX},
    Config, GenerateResponse, Ltx2GuidanceOverrides, Ltx2PipelineMode, Ltx2SpatialUpscale,
    Ltx2TemporalUpscale, ModelInfoExtended, OutputFormat, PromptTransformOperation, RemixResponse,
    Scheduler, ServerStatus, SseProgressEvent,
};
use rand::Rng;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::{Protocol, StatefulProtocol};
use std::collections::VecDeque;
use tokio::sync::mpsc;
use tui_textarea::TextArea;

use crate::action::{Action, View};
use crate::event::map_event;
#[cfg(test)]
use crate::model_info::capabilities_for_family;
use crate::model_info::{capabilities_for_model, family_for_model, ModelCapabilities};
use crate::ui::theme::Theme;

/// Immutable semantic and routing authority captured when a prompt transform
/// starts. Async results and prepared batches must never be reinterpreted from
/// the live form after this point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PromptTransformSnapshot {
    pub operation: PromptTransformOperation,
    pub model: String,
    pub target: crate::hosts::GenTarget,
    pub task: mold_core::ExpandTask,
    /// Ordered reference identity without retaining any filesystem path.
    pub reference_fingerprint: String,
    pub source_prompt: String,
    pub current_prompt: String,
    pub root_prompt: Option<String>,
    pub source_kind: mold_core::RemixSourceKind,
}

/// One registry-derived bundle whose exact terms must be accepted before the
/// TUI can download it. Presentation and key handling are model-agnostic.
#[derive(Debug, Clone)]
pub struct LicenseDownloadRequirement {
    pub install_model: String,
    pub licenses: Vec<mold_core::LicenseRefusal>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableGenerationChildOutcome {
    pub index: u32,
    pub job_id: String,
    /// The admission authority this child belongs to. `POST
    /// /api/queue/{id}/retry` fences on the whole of it, so only the client
    /// that admitted the batch can retry a held child — which is why the
    /// retry lives here and not in the Machines queue lanes, whose rows carry
    /// no batch identity.
    pub authority: mold_core::GenerationBatchAuthority,
    pub filename: Option<String>,
    pub original_filename: Option<String>,
    pub error: Option<String>,
    pub retryable: bool,
    /// Terminal facts the child settled with. Absent for a child that did
    /// not complete, and for one settled by a server that predates them.
    pub seed: Option<u64>,
    pub generation_time_ms: Option<u64>,
    /// Bytes of the print this pane shows. Only the last completed child
    /// carries them — the pane displays one image, so hydrating every
    /// sibling would download a batch to throw it away.
    pub preview_bytes: Option<Vec<u8>>,
}

/// Events sent from background tasks to the main TUI loop.
pub enum BackgroundEvent {
    /// Progress update from generation or model pull.
    Progress(SseProgressEvent),
    /// Generation completed successfully. `from_local` is true for any
    /// response produced by the in-process inference engine — including
    /// Auto-mode fallbacks after the remote server goes unreachable —
    /// so the completion handler can still write the file locally even
    /// when `server_url` remains set. `metadata_snapshot` is the exact
    /// submitted request state so later form edits cannot corrupt provenance.
    GenerationComplete {
        response: Box<GenerateResponse>,
        from_local: bool,
        metadata_snapshot: Box<GenerationMetadataSnapshot>,
    },
    /// A canonical remote Batch N settled through the reconnectable queue API.
    /// Results are gallery identities, not fabricated response bytes.
    DurableGenerationBatchComplete {
        outcomes: Vec<DurableGenerationChildOutcome>,
        /// The prompt, negative and model this batch was submitted with —
        /// the form may have moved on while it rendered, and prompt history
        /// records what was actually developed.
        prompt: String,
        negative_prompt: Option<String>,
        model: String,
        /// The host that admitted the batch. A held child can only be retried
        /// there — the retry fence is that instance's authority — whatever
        /// machine the form points at by the time the user presses `^T`.
        host: HeldHost,
    },
    /// Generation or background task failed.
    Error(String),
    /// A background placement/download check paused for explicit consent.
    /// A host answered `GET /api/licenses` for the review popup.
    LicenseListingLoaded {
        host_label: String,
        licenses: Vec<mold_core::types::ThirdPartyLicenseStatus>,
    },
    /// That host could not be asked. Never answered with local state: the
    /// question is always "accepted on WHICH machine?".
    LicenseListingFailed {
        host_label: String,
        message: String,
    },
    LicenseRequired {
        host_label: String,
        requirements: Vec<LicenseDownloadRequirement>,
        response: tokio::sync::oneshot::Sender<bool>,
    },
    /// A reviewable prompt transform completed. The token fences late results
    /// after the user edits or starts another request.
    PromptTransformComplete {
        token: u64,
        operation: PromptTransformOperation,
        snapshot: PromptTransformSnapshot,
        response: RemixResponse,
    },
    PromptTransformFailed {
        token: u64,
        message: String,
    },
    /// Merged all-hosts gallery scan completed.
    GalleryScanComplete(crate::gallery_scan::MergedScan),
    /// Model pull completed.
    PullComplete(String),
    /// Background thumbnail generation finished.
    ThumbnailsReady,
    /// One visible grid thumbnail finished decoding off the render path.
    GalleryThumbnailReady {
        path: std::path::PathBuf,
        image: Option<image::DynamicImage>,
    },
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
    UpscaleProgress {
        tile: usize,
        total: usize,
    },
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
    /// Durable Framewise video-upscale state from the owning host.
    FramewiseUpscaleStatus(mold_core::VideoUpscaleJob),
    /// Upscale failed.
    UpscaleFailed(String),
    /// A Library mesh export finished; carries the file it wrote.
    MeshExportComplete(std::path::PathBuf),
    /// A Library mesh export failed, with the host's or the writer's own
    /// sentence.
    MeshExportFailed(String),
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
    /// The durable sequence job settled successfully. Carries only what the
    /// view renders: the compatibility endpoint that returned a whole
    /// `ChainResponse` is gone, and the stitched print now lands in the
    /// host's gallery rather than in this event.
    ChainComplete {
        stage_count: u32,
        request_warnings: Vec<String>,
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
        // Boxed like the device payload above: `ServerCapabilities` grew past
        // the point where inlining it set the size of every variant in this
        // enum, which is sent for every poll of every host.
        capabilities: Option<Box<mold_core::ServerCapabilities>>,
    },
    /// Per-host queue snapshot for the selected Machines row.
    HostQueueUpdate {
        host_id: String,
        queue: Option<mold_core::QueueListingWire>,
    },
    /// User-requested continuation page, fenced by the cursor it followed.
    HostQueuePageLoaded {
        host_id: String,
        cursor: String,
        queue: Option<mold_core::QueueListingWire>,
    },
    /// Releases the per-host periodic polling lease after success or timeout.
    HostPollFinished {
        host_id: String,
    },
    /// Releases the connected-server resource polling lease.
    ServerStatusPollFinished,
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
/// The chain composer is nested under Create (mirroring the graphical
/// surfaces, where Sequence is an Output setting of Create) rather than
/// being a tab of its own. Switching workspaces does not reset the mode —
/// a chain in progress survives a round-trip through Library and back;
/// only [`Action::ChainExit`] returns to Compose.
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
    PredictDuration,
    Duration,
    // Advanced — Sampling
    Scheduler,
    Expand,
    Offload,
    // Advanced — Source
    SourceImage,
    References,
    Strength,
    MaskImage,
    ControlImage,
    ControlModel,
    ControlScale,
    // Advanced — 3-D mesh (`GenerateRequest.mesh`; shown only while the
    // recipe's profile carries a `mesh` block)
    /// Query-grid resolution, cycled through the profile's allowlist.
    Octree,
    /// Iso-level the surface is extracted at.
    MeshThreshold,
    /// Decimation target; off keeps the raw surface.
    TargetFaces,
    // Advanced — Identity (PuLID)
    IdentityImage,
    IdentityWeight,
    IdentityStartStep,
    // Advanced — LoRA / Upscale / Output
    Lora,
    Upscale,
    Format,
    // Advanced — Video
    Frames,
    Fps,
    Pipeline,
    Audio,
    SpatialUpscale,
    TemporalUpscale,
    StgScale,
    StgBlocks,
    RescaleScale,
    ModalityScale,
    GuidanceSkip,
    /// Wan flow shift (#782). Absent-until-touched, like the LTX-2 override
    /// rows above.
    SampleShift,
    // Advanced — File under (creation-time library organization)
    /// The print's name. Rides every request as `GenerateRequest.title` and
    /// is the source of the optional auto tag.
    Title,
    /// Comma-separated creation-time tags.
    Tags,
    /// Collection to file the print under, by display name.
    Collection,
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
            Self::PredictDuration => "Predict duration",
            Self::Duration => "Duration",
            Self::Format => "Format",
            Self::Scheduler => "Scheduler",
            Self::Lora => "LoRA",
            Self::Upscale => "Upscale",
            Self::Expand => "Expand prompt",
            Self::Offload => "Offload",
            Self::SourceImage => "Source",
            Self::References => "References",
            Self::Strength => "Strength",
            Self::MaskImage => "Mask",
            Self::ControlImage => "Control",
            Self::ControlModel => "CNet Mdl",
            Self::Frames => "Frames",
            Self::Fps => "FPS",
            Self::Pipeline => "Pipeline",
            Self::Audio => "Audio",
            Self::SpatialUpscale => "Spatial",
            Self::TemporalUpscale => "Temporal",
            Self::StgScale => "STG scale",
            Self::StgBlocks => "STG blocks",
            Self::RescaleScale => "CFG rescale",
            Self::ModalityScale => "Modality",
            Self::GuidanceSkip => "Guide skip",
            Self::SampleShift => "Flow shift",
            Self::Title => "Title",
            Self::Tags => "Tags",
            Self::Collection => "Collection",
            Self::ControlScale => "Scale",
            Self::Octree => "Octree",
            Self::MeshThreshold => "Iso threshold",
            Self::TargetFaces => "Target faces",
            // These three live inside the "Identity photo" section, so they
            // are named for their role there — `LABEL_W` is 16 columns and a
            // repeated "Identity " prefix would not fit any of them.
            Self::IdentityImage => "Photo",
            Self::IdentityWeight => "Strength",
            Self::IdentityStartStep => "Start step",
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
    /// Earliest known user idea retained across Expand/Remix.
    pub original_prompt: Option<String>,
    /// Structured provenance for the currently visible transformed prompt.
    pub prompt_transform: Option<mold_core::PromptTransformProvenance>,
    /// Frozen semantic authority for one applied variant. Its original host
    /// is intentionally releasable after review.
    pub quick_transform_snapshot: Option<PromptTransformSnapshot>,
    /// Reviewed Remix siblings. Empty keeps the ordinary batch behavior.
    pub prepared_prompts: Vec<String>,
    pub prepared_prompt_transforms: Vec<mold_core::PromptTransformProvenance>,
    /// Frozen route and semantic authority for reviewed Batch N work.
    pub prepared_transform_snapshot: Option<PromptTransformSnapshot>,
    /// Shared sibling identity and one-based position, populated at dispatch.
    pub batch_id: Option<String>,
    pub batch_index: Option<u32>,
    pub batch_count: Option<u32>,
    pub offload: bool,
    /// Upscaler to run after generation (wired to the existing
    /// `GenerateRequest.upscale_model` field). `None` = off.
    pub upscale_model: Option<String>,
    // img2img
    pub source_image_path: Option<String>,
    /// Ordered H3 reference paths. This transient state is deliberately not
    /// serialized; only basename + digest provenance crosses the wire.
    pub reference_paths: Vec<crate::h3_references::ReferencePath>,
    pub strength: f64,
    pub mask_image_path: Option<String>,
    // Identity (PuLID). The path is transient TUI state: only the
    // basename and the bytes cross the wire, exactly as the source image
    // does. Validated once at entry (`crate::identity::load_identity_image`)
    // so an unreadable or out-of-bounds photo never reaches a queue slot.
    pub identity_image_path: Option<String>,
    /// Identity strength in `0.0..=mold_core::identity::ID_WEIGHT_MAX`.
    /// Shipped explicitly whenever a photo is attached, so the saved
    /// provenance records the value the user actually saw.
    pub id_weight: f64,
    /// First identity-conditioned denoise step; always `< steps`.
    pub id_start_step: u32,
    // Video
    /// User opt-in to LTX-2.5's duration head. It only takes effect while the
    /// selected server positively advertises a complete runtime pack.
    pub predict_duration: bool,
    pub duration_prediction_supported: bool,
    pub frames: u32,
    pub fps: u32,
    /// Explicit source-free LTX-2 video recipe. `None` lets the server select
    /// the checkpoint default. Conditioning-dependent and audio-only recipes
    /// stay out of the TUI until their required media paths are authorable.
    pub pipeline: Option<Ltx2PipelineMode>,
    /// `None` preserves the pipeline default; explicit values mirror the
    /// CLI's `--audio` and `--no-audio` controls.
    pub enable_audio: Option<bool>,
    /// `None` keeps the checkpoint's native spatial resolution.
    pub spatial_upscale: Option<Ltx2SpatialUpscale>,
    /// `None` keeps the checkpoint's native frame rate.
    pub temporal_upscale: Option<Ltx2TemporalUpscale>,
    /// Optional LTX-2 guider tuning. An empty value must stay absent from the
    /// request so the selected pipeline retains its own constants.
    pub guidance_overrides: Ltx2GuidanceOverrides,
    /// Wan flow shift (#782). `None` stays absent from the request so the
    /// per-tier pipeline defaults remain authoritative.
    pub sample_shift: Option<f64>,
    // ControlNet
    pub control_image_path: Option<String>,
    pub control_model: Option<String>,
    pub control_scale: f64,
    // File under — creation-time library organization. All three are
    // absent-until-touched: an untouched form sends no `title`, no `tags`,
    // and no `collection`, so the request is byte-identical to before.
    pub title: Option<String>,
    /// Tags the user typed, already normalized by
    /// `mold_core::normalize_request_tags`. The tag derived from the title
    /// stays *derived* and is composed at submit time.
    pub tags: Vec<String>,
    /// Collection display name, already normalized by
    /// `mold_core::validate_collection_name`.
    pub collection: Option<String>,
    /// Snapshot of the effective `generate.auto_tag_title` preference. Held
    /// on the form rather than re-read per call so the summary the user sees
    /// and the request that is submitted can never disagree; Settings
    /// refreshes it when the preference is toggled.
    pub auto_tag_title: bool,
    // 3-D mesh controls (`GenerateRequest.mesh`). Every field is
    // absent-until-touched: an untouched form ships no `mesh` block at all,
    // and a touched value is shipped exactly as the row shows it. The rows
    // exist only while `ModelCapabilities.mesh` (the recipe's profile block)
    // is present, and a model switch to a recipe without one clears them, so
    // a stale octree can never reach a raster model's admission.
    pub mesh: mold_core::MeshRequestOptions,
}

/// Immutable, lightweight provenance captured for one submitted generation.
/// Runtime-only output facts still come from `GenerateResponse`.
#[derive(Debug, Clone)]
pub struct GenerationMetadataSnapshot {
    pub params: GenerateParams,
    pub prompt: String,
    pub negative_prompt: Option<String>,
}

impl GenerationMetadataSnapshot {
    pub(crate) fn new(
        params: GenerateParams,
        prompt: String,
        negative_prompt: Option<String>,
    ) -> Self {
        Self {
            params,
            prompt,
            negative_prompt,
        }
    }
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

/// Whether a run built from `params` needs a prompt.
///
/// LTX-2 / LTX-Video runs that already carry a source image may go unprompted
/// — their text encoder pads to a fixed-width context, so `""` is a trained
/// input. Expect near-static, micro-motion output; it saves no VRAM, since the
/// prompt-context tensor is the same size either way. The TUI Create form has
/// no keyframe / source-video / extend controls, so the source image is the
/// only conditioning it can contribute.
pub(crate) fn prompt_required_for_params(params: &GenerateParams, config: &Config) -> bool {
    mold_core::prompt_required_with_conditioning(
        Some(&family_for_model(&params.model, config)),
        params.source_image_path.is_some(),
    )
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
            original_prompt: None,
            prompt_transform: None,
            quick_transform_snapshot: None,
            prepared_prompts: Vec::new(),
            prepared_prompt_transforms: Vec::new(),
            prepared_transform_snapshot: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            offload: false,
            upscale_model: None,
            source_image_path: None,
            reference_paths: Vec::new(),
            strength: 0.75,
            mask_image_path: None,
            identity_image_path: None,
            id_weight: mold_core::identity::ID_WEIGHT_DEFAULT,
            id_start_step: mold_core::identity::ID_START_STEP_DEFAULT,
            predict_duration: false,
            duration_prediction_supported: false,
            frames: 25,
            fps: 24,
            pipeline: None,
            enable_audio: None,
            spatial_upscale: None,
            temporal_upscale: None,
            guidance_overrides: Ltx2GuidanceOverrides::default(),
            sample_shift: None,
            control_image_path: None,
            control_model: None,
            control_scale: 1.0,
            title: None,
            tags: Vec::new(),
            collection: None,
            auto_tag_title: config.generate.auto_tag_title,
            mesh: mold_core::MeshRequestOptions::default(),
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
            ParamField::PredictDuration => {
                if self.predict_duration { "on" } else { "off" }.to_string()
            }
            ParamField::Duration => {
                if self.predict_duration && self.duration_prediction_supported {
                    "automatic · 1–20s".to_string()
                } else {
                    format!(
                        "{:.1}s · {}f",
                        self.frames as f64 / self.fps.max(1) as f64,
                        self.frames
                    )
                }
            }
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
            ParamField::IdentityImage => self
                .identity_image_path
                .as_deref()
                .map(|p| {
                    std::path::Path::new(p)
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or_else(|| p.to_string())
                })
                .unwrap_or_else(|| "\u{27e8}none\u{27e9}".to_string()),
            ParamField::IdentityWeight => format!("{:.2}", self.id_weight),
            ParamField::IdentityStartStep => self.id_start_step.to_string(),
            ParamField::References => match self.reference_paths.len() {
                0 => "\u{27e8}none\u{27e9}".to_string(),
                1 => "1 ordered file".to_string(),
                count => format!("{count} ordered files"),
            },
            ParamField::Strength => format!("{:.2}", self.strength),
            // The three mesh rows read "default" while untouched: the
            // profile's own default is what the renderer (`param_form`)
            // substitutes, because the parameter bag does not carry it.
            ParamField::Octree => self
                .mesh
                .octree_resolution
                .map(|value| value.to_string())
                .unwrap_or_else(|| "default".to_string()),
            ParamField::MeshThreshold => self
                .mesh
                .threshold
                .map(|value| format!("{value:.2}"))
                .unwrap_or_else(|| "default".to_string()),
            ParamField::TargetFaces => self
                .mesh
                .target_faces
                .map(crate::ui::preview::format_thousands)
                .unwrap_or_else(|| "off \u{00b7} raw surface".to_string()),
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
            ParamField::Pipeline => self
                .pipeline
                .map(|pipeline| pipeline.to_string())
                .unwrap_or_else(|| "auto".to_string()),
            ParamField::Audio => self
                .enable_audio
                .map(|enabled| if enabled { "on" } else { "off" })
                .unwrap_or("default")
                .to_string(),
            ParamField::SpatialUpscale => match self.spatial_upscale {
                None => "native".to_string(),
                Some(Ltx2SpatialUpscale::X1_5) => "1.5×".to_string(),
                Some(Ltx2SpatialUpscale::X2) => "2×".to_string(),
            },
            ParamField::TemporalUpscale => match self.temporal_upscale {
                None => "native".to_string(),
                Some(Ltx2TemporalUpscale::X2) => "2×".to_string(),
            },
            ParamField::StgScale => optional_guidance_value(self.guidance_overrides.stg_scale),
            ParamField::StgBlocks => self
                .guidance_overrides
                .stg_blocks
                .as_ref()
                .map(|blocks| {
                    blocks
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_else(|| "default".to_string()),
            ParamField::RescaleScale => {
                optional_guidance_value(self.guidance_overrides.rescale_scale)
            }
            ParamField::ModalityScale => {
                optional_guidance_value(self.guidance_overrides.modality_scale)
            }
            ParamField::GuidanceSkip => self
                .guidance_overrides
                .skip_step
                .map(|value| value.to_string())
                .unwrap_or_else(|| "default".to_string()),
            ParamField::SampleShift => self
                .sample_shift
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "default".to_string()),
            ParamField::ControlScale => format!("{:.1}", self.control_scale),
            ParamField::Title => self
                .title
                .clone()
                .unwrap_or_else(|| "\u{27e8}untitled\u{27e9}".to_string()),
            // The auto tag is disclosed on the row it would join, so a tag
            // the user did not type is visible before Generate.
            ParamField::Tags => {
                let auto = crate::ui::create_form::auto_tag_disclosure(self);
                match (self.tags.is_empty(), auto) {
                    (true, None) => "\u{27e8}none\u{27e9}".to_string(),
                    (true, Some(slug)) => format!("auto: {slug}"),
                    (false, None) => crate::ui::create_form::format_tag_input(&self.tags),
                    (false, Some(slug)) => format!(
                        "{} \u{00b7} auto: {slug}",
                        crate::ui::create_form::format_tag_input(&self.tags)
                    ),
                }
            }
            ParamField::Collection => self
                .collection
                .clone()
                .unwrap_or_else(|| "\u{27e8}none\u{27e9}".to_string()),
        }
    }
}

fn optional_guidance_value(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.1}"))
        .unwrap_or_else(|| "default".to_string())
}

fn adjust_optional_scale(current: Option<f64>, delta: i32, step: f64, max: f64) -> Option<f64> {
    if delta >= 0 {
        match current {
            None => Some(0.0),
            Some(value) if value + step <= max + f64::EPSILON => Some(value + step),
            Some(_) => None,
        }
    } else {
        match current {
            None => Some(max),
            Some(value) if value - step >= -f64::EPSILON => Some((value - step).max(0.0)),
            Some(_) => None,
        }
    }
}

fn adjust_optional_u32(current: Option<u32>, delta: i32, max: u32) -> Option<u32> {
    if delta >= 0 {
        match current {
            None => Some(0),
            Some(value) if value < max => Some(value + 1),
            Some(_) => None,
        }
    } else {
        match current {
            None => Some(max),
            Some(value) if value > 0 => Some(value - 1),
            Some(_) => None,
        }
    }
}

fn parse_stg_blocks_input(input: &str) -> std::result::Result<Option<Vec<u32>>, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let entries = trimmed
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return Err("List at least one transformer block.".to_string());
    }
    if entries.len() > MAX_STG_BLOCKS {
        return Err(format!("At most {MAX_STG_BLOCKS} blocks."));
    }
    let mut blocks = Vec::with_capacity(entries.len());
    for entry in entries {
        let block = entry
            .parse::<u32>()
            .map_err(|_| format!("“{entry}” is not a block index."))?;
        if block >= MAX_STG_BLOCK_INDEX {
            return Err(format!(
                "Block {block} is deeper than any LTX-2 checkpoint."
            ));
        }
        if blocks.contains(&block) {
            return Err(format!("Block {block} is listed twice."));
        }
        blocks.push(block);
    }
    Ok(Some(blocks))
}

/// Mirror `/api/models`' requestable video grid for the TUI's keyboard sliders.
#[derive(Debug, Clone, Copy)]
struct TuiVideoGrid {
    step: u32,
    offset: u32,
    min_frames: u32,
    fixed_fps: Option<u32>,
    runtime_seconds: Option<u32>,
    absolute_frames: Option<u32>,
    fixed_frames: u32,
}

impl TuiVideoGrid {
    fn snap_nearest(self, target: u32) -> u32 {
        if target <= self.offset {
            return self.offset;
        }
        ((target - self.offset + self.step / 2) / self.step)
            .saturating_mul(self.step)
            .saturating_add(self.offset)
    }
}

fn tui_max_video_frames(grid: TuiVideoGrid, fps: u32) -> u32 {
    let cap = if let Some(seconds) = grid.runtime_seconds {
        let duration_cap = seconds.saturating_mul(fps.max(1)).saturating_add(4);
        grid.absolute_frames
            .map_or(duration_cap, |absolute| duration_cap.min(absolute))
    } else {
        grid.fixed_frames
    };
    if cap < grid.offset {
        return cap;
    }
    cap - (cap - grid.offset) % grid.step
}

/// Repair every legacy/shared TUI field that H3 fixes or does not consume.
///
/// This is deliberately applied after persistence/default overlays and again
/// at submit time. Hidden controls are presentation only; this function is the
/// request-authority boundary that prevents stale state from weakening H3's
/// synchronized AV contract. It does not activate the compliance-gated family.
pub(crate) fn normalize_generate_params_for_family(params: &mut GenerateParams, family: &str) {
    if family == mold_core::manifest::HUNYUAN3D_FAMILY {
        // A 3-D render emits binary glTF and nothing else, so the format is
        // pinned rather than chosen — the same rule as
        // `GenerateRequest::pin_output_format_for_family` on the server, so
        // the form and admission can never disagree. The source image is
        // the family's ONLY conditioning and is kept; a mask, a ControlNet
        // and a LoRA name things a mesh does not have, and the server
        // refuses them rather than ignoring them, so stale values from a
        // previous raster model are cleared here instead of earning a 422.
        params.format = OutputFormat::Glb;
        params.mask_image_path = None;
        params.control_image_path = None;
        params.control_model = None;
        params.control_scale = 1.0;
        params.lora_path = None;
        params.lora_scale = 1.0;
        params.scheduler = None;
        params.upscale_model = None;
        return;
    }
    if !mold_core::minimax_h3::is_family(family) {
        return;
    }

    // Model-aware, and today every H3 layout takes the family grid: an
    // off-grid stale count snaps to the nearest `17n+5` point rather than to
    // one pinned clip length.
    params.frames =
        mold_core::minimax_h3::recommended_frames_for_model(family, &params.model, params.frames);
    params.fps = mold_core::minimax_h3::FIXED_FPS;
    params.format = OutputFormat::Mp4;
    params.enable_audio = Some(true);
    params.guidance = 0.0;
    params.strength = 1.0;

    params.scheduler = None;
    params.lora_path = None;
    params.lora_scale = 1.0;
    params.source_image_path = None;
    params.mask_image_path = None;
    params.control_image_path = None;
    params.control_model = None;
    params.control_scale = 1.0;
    params.pipeline = None;
    params.spatial_upscale = None;
    params.temporal_upscale = None;
    params.guidance_overrides = Ltx2GuidanceOverrides::default();
}

/// ◀▶ on the Octree row: walk the profile's allowlist from the current
/// value (the default while untouched). Never leaves the list, and never
/// returns to "untouched" — once the user has chosen, the request says so.
pub(crate) fn next_octree_resolution(
    allowed: &[u32],
    default: u32,
    current: Option<u32>,
    delta: i32,
) -> Option<u32> {
    if allowed.is_empty() {
        return current;
    }
    let anchor = current.unwrap_or(default);
    let index = allowed
        .iter()
        .position(|&value| value == anchor)
        .map(|index| index as i32)
        .unwrap_or(0);
    let next = (index + delta).clamp(0, allowed.len() as i32 - 1) as usize;
    Some(allowed[next])
}

/// ◀▶ on the Iso threshold row: five profile steps per press (the profile
/// step is 0.01, which is too fine for a keyboard) inside the profile's own
/// range, rounded to two decimals so repeated presses do not accumulate
/// binary drift into the recorded provenance.
pub(crate) fn next_mesh_threshold(
    control: &mold_core::FloatControl,
    current: Option<f32>,
    delta: i32,
) -> f32 {
    let anchor = current.map_or(control.default, f64::from);
    let step = control.step * 5.0;
    let next = ((anchor + f64::from(delta) * step) * 100.0).round() / 100.0;
    next.clamp(control.min, control.max) as f32
}

/// ◀▶ on the Target faces row: 10 000 triangles per press inside the
/// profile's bounds. Stepping below the minimum turns decimation OFF (`None`
/// keeps the raw surface), and stepping up from off starts at the minimum.
pub(crate) fn next_target_faces(
    min: u32,
    max: u32,
    current: Option<u32>,
    delta: i32,
) -> Option<u32> {
    const STEP: i64 = 10_000;
    match current {
        None if delta > 0 => Some(min.max(1)),
        None => None,
        Some(value) => {
            let next = i64::from(value) + i64::from(delta) * STEP;
            if next < i64::from(min) {
                // A single press from the minimum turns decimation off
                // rather than pinning the row at a floor it cannot leave.
                if value <= min {
                    None
                } else {
                    Some(min)
                }
            } else {
                Some(next.min(i64::from(max)) as u32)
            }
        }
    }
}

/// Every export container the in-process writer offers for a LOCAL `.glb`.
/// GLB itself is the stored form, not an export, so it is never listed.
pub(crate) const LOCAL_MESH_EXPORT_FORMATS: [mold_core::MeshExportFormat; 3] = [
    mold_core::MeshExportFormat::Obj,
    mold_core::MeshExportFormat::Stl,
    mold_core::MeshExportFormat::Ply,
];

/// The containers the export picker lists: the owning host's advertised
/// `capabilities.mesh.export_formats` minus GLB, or the local writer's set
/// when there is no host (a local print) or the host has not been polled.
pub(crate) fn mesh_export_formats_for(
    advertised: Option<&[mold_core::MeshExportFormat]>,
) -> Vec<mold_core::MeshExportFormat> {
    match advertised {
        Some(list) => list
            .iter()
            .copied()
            .filter(|format| *format != mold_core::MeshExportFormat::Glb)
            .collect(),
        None => LOCAL_MESH_EXPORT_FORMATS.to_vec(),
    }
}

/// Where an export lands: `<output_dir>/<stem>.<ext>`, beside the TUI's
/// other saves and named after the print, so `chair.glb` exports as
/// `chair.stl` exactly as `mold library export` names it.
pub(crate) fn mesh_export_target_path(
    output_dir: &std::path::Path,
    filename: &str,
    format: mold_core::MeshExportFormat,
) -> std::path::PathBuf {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| filename.to_string());
    output_dir.join(format!("{stem}.{}", format.extension()))
}

/// Transcode local `.glb` bytes with the same writer the server's export
/// route uses, so a local print exports byte-for-byte as a served one would.
pub(crate) fn export_local_mesh(
    glb: &[u8],
    format: mold_core::MeshExportFormat,
) -> Result<Vec<u8>, String> {
    use mold_inference::hunyuan3d::glb;
    if format == mold_core::MeshExportFormat::Glb {
        return Ok(glb.to_vec());
    }
    let mesh = glb::read_glb(glb).map_err(|e| e.to_string())?;
    Ok(match format {
        mold_core::MeshExportFormat::Glb => unreachable!("returned above"),
        mold_core::MeshExportFormat::Obj => glb::write_obj(&mesh).into_bytes(),
        mold_core::MeshExportFormat::Stl => glb::write_stl(&mesh),
        mold_core::MeshExportFormat::Ply => glb::write_ply(&mesh),
    })
}

/// What the Negative editor shows on cold start (#787 round 2). `App::new`
/// never runs `sync_generate_capabilities`, so the selected model's default
/// must be resolved here or a remembered wan model boots with an empty
/// editor and no expressible opt-out until the model is re-selected.
///
/// - saved text → restored verbatim (typed authority; text equal to the
///   default reads as untouched on the wire, exactly as it was saved);
/// - empty with the session's persisted `negative_cleared` marker → stays
///   empty, so an explicit `""` opt-out survives restart distinguishably;
/// - empty otherwise → the untouched state shows the advertised default
///   (this is also the upgrade path for pre-#787 sessions, which carry no
///   marker).
fn restored_negative_editor_text(
    saved: &str,
    negative_cleared: Option<bool>,
    advertised_default: &str,
) -> String {
    if !saved.trim().is_empty() {
        saved.to_string()
    } else if negative_cleared == Some(true) {
        String::new()
    } else {
        advertised_default.trim().to_string()
    }
}

/// Build the Negative editor textarea with the standard cursor and
/// placeholder styling, from prefill text (possibly empty).
fn negative_prompt_textarea(text: &str) -> TextArea<'static> {
    let mut textarea = if text.is_empty() {
        TextArea::default()
    } else {
        TextArea::new(text.lines().map(String::from).collect())
    };
    textarea.set_cursor_line_style(ratatui::style::Style::default());
    textarea.set_placeholder_text("Negative prompt (what to avoid)...");
    textarea
}

/// State for the Generate view.
pub struct GenerateState {
    pub prompt: TextArea<'static>,
    pub negative_prompt: TextArea<'static>,
    /// The selected model's advertised default negative prompt
    /// (`/api/models[].default_negative_prompt`, wan today; empty when
    /// none). Prefilled into the editor while it is untouched; at submit,
    /// editor text equal to this stays absent on the wire and a cleared
    /// editor ships the explicit `""` opt-out
    /// (`create_form::negative_prompt_wire_value`).
    pub negative_default: String,
    /// Restore-time explicit-clear authority (#787 round 3): true while an
    /// empty editor is a restored explicit `""` opt-out (session marker on
    /// cold start, `""` metadata on gallery reuse) rather than "untouched".
    /// Keeps the clear from being mistaken for the untouched state by
    /// default-change reconciliation and wire serialization; reset by an
    /// explicit model switch and by Reset Defaults.
    pub negative_explicit_clear: bool,
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
    /// Latest transient denoise preview from the shared SSE stream. This is
    /// separate from `preview_image`: completion replaces transient authority
    /// with the final print instead of letting a latent frame survive settle.
    pub live_preview_image: Option<image::DynamicImage>,
    /// Fixed-protocol render cache keyed by Preview-panel geometry. A new SSE
    /// frame invalidates it so Kitty/Sixel/iTerm2 repaint the latest pixels.
    pub live_preview_protocol: Option<(u16, u16, Protocol)>,
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
    /// Why the attached identity photo cannot be used right now — a rejected
    /// file at entry time, or the model gate after a switch to a checkpoint
    /// that does not advertise `supports_identity`. Rendered inline on the
    /// Photo row (and in the picker) and re-checked at dispatch, so the
    /// refusal is never only a late server error. The photo itself is kept:
    /// the user chose it, and switching back must not have lost it.
    pub identity_error: Option<String>,
    /// Non-blocking advisory (e.g. an admitted off-bucket size); rendered in
    /// the error row's slot with warning styling, never as an error.
    pub warning_message: Option<String>,
    pub model_description: String,
    /// Path of the most recently saved output — drives the activity
    /// strip's "done · saved to …" line. None when saving is disabled or
    /// the server kept the file.
    pub last_output_path: Option<std::path::PathBuf>,
    /// The batch this client last submitted that still has held children,
    /// for `^T`. Replaced by the next batch that settles and cleared by the
    /// next submission — a retry is offered for work this session still
    /// owns, never for a row somebody else admitted — and deliberately NOT
    /// cleared when a retry fails: the children stay retryable on the host,
    /// so the user must be able to press `^T` again once the cause is fixed.
    pub held_batch: Option<HeldBatch>,
    /// Monotonic fence for async Expand/Remix results.
    pub prompt_transform_token: u64,
    /// `tris · verts · bounds` of the most recent finished 3-D print, for
    /// the Preview caption. `None` after a raster or video completion, so a
    /// mesh summary never captions a picture.
    pub last_mesh_summary: Option<String>,
}

/// One held child this client can retry.
#[derive(Debug, Clone)]
pub struct HeldPrintRetry {
    pub authority: mold_core::GenerationBatchAuthority,
    pub job_id: String,
    pub index: u32,
}

/// The authenticated route of the host that admitted a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldHost {
    pub url: String,
    pub api_key: Option<String>,
}

/// Everything a retry needs and nothing the form can change underneath it:
/// the admitting host (the retry fence is that instance's authority), the
/// submission prompt history must record for the print the host actually
/// renders, and the children still held.
#[derive(Debug, Clone)]
pub struct HeldBatch {
    pub host: HeldHost,
    pub submission: crate::backend::OwnedBatchSubmission,
    pub retries: Vec<HeldPrintRetry>,
}

impl HeldBatch {
    /// The batch a settled report leaves behind for `^T`, or `None` when no
    /// child is retryable.
    pub fn from_outcomes(
        host: HeldHost,
        submission: crate::backend::OwnedBatchSubmission,
        outcomes: &[DurableGenerationChildOutcome],
    ) -> Option<Self> {
        let retries = outcomes
            .iter()
            .filter(|outcome| outcome.retryable)
            .map(|outcome| HeldPrintRetry {
                authority: outcome.authority.clone(),
                job_id: outcome.job_id.clone(),
                index: outcome.index,
            })
            .collect::<Vec<_>>();
        (!retries.is_empty()).then_some(Self {
            host,
            submission,
            retries,
        })
    }
}

impl GenerateState {
    fn clear_live_preview(&mut self) {
        self.live_preview_image = None;
        self.live_preview_protocol = None;
    }

    /// The checkpoint default or an explicit source-free LTX-2 recipe resolves
    /// whether the primary guidance control is adjustable.
    pub fn guidance_adjustable(&self) -> bool {
        !self.capabilities.supports_video || self.capabilities.supports_negative_prompt
    }

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
    /// Source paths currently queued for off-thread thumbnail decode.
    pub thumbnail_loading: std::collections::HashSet<std::path::PathBuf>,
    /// Most-recently decoded grid entries. Keeps image protocols bounded.
    pub thumbnail_lru: std::collections::VecDeque<usize>,
    /// Selected print's predecoded thumbnail protocol for the details panel.
    pub details_thumbnail_state: Option<(usize, StatefulProtocol, (u32, u32))>,
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
    /// Whether this machine's disk-backed prints can be moved to
    /// `<output_dir>/.trash/` (the metadata DB answered the last local
    /// scan). False ⇒ a local delete is the pre-trash hard delete.
    pub local_trash_available: bool,
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
            thumbnail_loading: std::collections::HashSet::new(),
            thumbnail_lru: std::collections::VecDeque::new(),
            details_thumbnail_state: None,
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
            local_trash_available: false,
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
    /// Editable print title from the gallery row (`generations.title` /
    /// `GalleryImage.title`). `None` for untitled prints and for hosts
    /// too old to send one — the UI then shows nothing in its place.
    pub title: Option<String>,
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

/// Settings display for `gallery.trash_retention_days`: the day count,
/// with `0` spelled out as the "keep forever" it means.
pub(crate) fn trash_retention_display(days: u32) -> String {
    if days == 0 {
        "0 (forever)".to_string()
    } else {
        days.to_string()
    }
}

/// What pressing `d` on a print will do on the machines that hold it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RemovalKind {
    /// Every owning machine moves the print to its gallery trash: the
    /// local DB-backed `.trash/` move, or a server whose capabilities
    /// advertise `gallery.trash`. Recoverable from any Trash view.
    Trash,
    /// At least one owning machine cannot trash (DB-less local scan, an
    /// older server, or a host whose capabilities are not known yet), so
    /// the TUI never promises recoverability for the whole print.
    Delete,
}

impl RemovalKind {
    /// Short key-hint label (`d Trash` / `d Delete`).
    pub(crate) fn hint_label(self) -> &'static str {
        match self {
            RemovalKind::Trash => "Trash",
            RemovalKind::Delete => "Delete",
        }
    }
}

/// Confirm copy for removing a print. Names every machine it will be
/// removed from when it exists on more than one. The trash wording is
/// used only when every owning machine can trash it (see
/// [`RemovalKind`]); otherwise neutral "Remove" copy that promises no
/// recovery — the older-server / DB-less delete really is permanent.
pub(crate) fn delete_confirm_message(filename: &str, machines: usize, kind: RemovalKind) -> String {
    match (kind, machines > 1) {
        (RemovalKind::Trash, true) => {
            format!("Move {filename} to the trash? It exists on {machines} machines.")
        }
        (RemovalKind::Trash, false) => format!("Move {filename} to the trash?"),
        (RemovalKind::Delete, true) => {
            format!("Remove {filename}? It exists on {machines} machines. This can't be undone.")
        }
        (RemovalKind::Delete, false) => format!("Remove {filename}? This can't be undone."),
    }
}

/// State for the Models view.
pub struct ModelsState {
    pub catalog: Vec<ModelInfoExtended>,
    pub selected: usize,
    pub filter: String,
    pub filtering: bool,
}

/// Generation-model names the create-form model picker (`Popup::ModelSelector`)
/// may offer: real generation models — never upscalers/utility rows — filtered
/// The inline sentence for a row this build cannot generate with.
///
/// The server names the obstacle on the row (`runtime_unavailable_reason`) —
/// a missing engine arm for the weight layout, a task with no qualified
/// route, or a binary built without the engine are three different answers
/// (#1276). An older server omits it, and the layout wording is the one that
/// was true of every download-only row it could publish.
fn runtime_unavailable_message(model: &ModelInfoExtended) -> String {
    model
        .runtime_unavailable_reason
        .clone()
        .unwrap_or_else(|| "No runtime for this layout in this build.".to_string())
}

/// by `query` (case-insensitive substring). `runtime_available: Some(false)`
/// marks a download-only row (e.g. the pruned NVFP4 H3 partitions) whose build
/// has no engine arm for it; picking one for generation would only earn a 501,
/// so it is excluded here. `None` (older servers, every other family) keeps
/// meaning runnable. The Models inventory view is unaffected — it lists
/// `self.models.catalog` directly and keeps showing every downloaded row.
fn generation_model_names(catalog: &[ModelInfoExtended], query: &str) -> Vec<String> {
    let query = query.to_lowercase();
    catalog
        .iter()
        .filter(|m| {
            m.is_generation_model()
                && m.runtime_available != Some(false)
                && m.name.to_lowercase().contains(&query)
        })
        .map(|m| m.name.clone())
        .collect()
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
    // Library (DB-surface `gallery.*` / `generate.*` keys)
    GalleryTrashRetentionDays,
    GenerateAutoTagTitle,
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
/// What the license review popup currently knows.
#[derive(Debug, Clone)]
pub enum LicenseListingState {
    Loading,
    Ready(Vec<mold_core::types::ThirdPartyLicenseStatus>),
    Failed(String),
}

pub enum Popup {
    Help,
    PromptSourceChoice {
        current_prompt: String,
        root_prompt: String,
        cursor: usize,
    },
    PromptAlternatives {
        snapshot: PromptTransformSnapshot,
        variants: Vec<mold_core::RemixVariant>,
        selected: Vec<bool>,
        cursor: usize,
    },
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
    /// Comma-separated LTX-2 transformer block indices. Invalid input stays
    /// visible and never reaches a generation request.
    StgBlocksInput {
        input: String,
        error: Option<String>,
    },
    /// Ordered `kind=path` MiniMax H3 reference input. Paths remain transient.
    ReferencesInput {
        input: String,
        error: Option<String>,
    },
    /// Local path to the PuLID face-identity photo. Committing opens the file
    /// no-follow and bounds-checks it through `mold_core::identity`, so a
    /// rejected photo never leaves the picker; the refusal stays visible here
    /// and on the row rather than arriving as a late server error.
    IdentityImageInput {
        input: String,
        error: Option<String>,
    },
    /// One File-under editor (Title, Tags, or Collection). Invalid input
    /// stays visible and never reaches a generation request.
    FilingInput {
        field: ParamField,
        input: String,
        error: Option<String>,
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
    /// Read-only listing of one host's third-party licenses.
    ///
    /// Deliberately holds no oneshot and settles nothing: accepting from here
    /// re-enters the SAME pull-time consent flow, so the settings path and the
    /// pull path can never disagree about what was shown.
    LicenseSettings {
        host_label: String,
        state: LicenseListingState,
        selected: usize,
    },
    LicenseReview {
        host_label: String,
        requirements: Vec<LicenseDownloadRequirement>,
        response: Option<tokio::sync::oneshot::Sender<bool>>,
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
    /// Library `x`: pick the container a stored `.glb` is exported as. The
    /// list is the owning host's advertised `capabilities.mesh.export_formats`
    /// (GLB itself excluded — it is the stored file), or every transcode the
    /// in-process writer offers for a local print.
    MeshExportPicker {
        filename: String,
        formats: Vec<mold_core::MeshExportFormat>,
        selected: usize,
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

/// Ceiling on the identity-photo path the picker will accept. Well above any
/// real filesystem path; it exists so a pasted blob cannot grow the popup
/// buffer without bound.
pub(crate) const IDENTITY_PATH_MAX_BYTES: usize = 4096;

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
    pub(crate) server_status_poll_in_flight: bool,
    pub history: crate::history::PromptHistory,
    /// Layout areas from the last render, used for mouse hit-testing.
    pub layout: LayoutAreas,
    /// Background server process spawned by the TUI (killed on quit).
    pub server_process: Option<std::process::Child>,
    /// Host id whose API key was supplied only for this TUI process. The key
    /// itself lives in `hosts::SESSION_API_KEYS` and is cleared on shutdown.
    pub(crate) session_api_key_host_id: Option<String>,
    /// A focused Library launch must never reinterpret a failed loopback HTTP
    /// scan as permission to walk and mutate the output directory directly.
    pub(crate) strict_gallery_authority: bool,
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

/// Verify that the requested Library authority is both reachable and usable
/// with the launch credential. `/health` is intentionally public, so it cannot
/// prove an authenticated gallery will work.
fn check_gallery_access(url: &str, api_key: Option<&str>) -> Result<()> {
    let client = crate::hosts::client_for(url, api_key);
    let runtime = tokio::runtime::Handle::current();
    std::thread::spawn(move || runtime.block_on(client.list_gallery()))
        .join()
        .map_err(|_| anyhow::anyhow!("gallery access probe panicked"))?
        .map(|_| ())
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

fn recoverable_framewise_upscale(
    jobs: &[mold_core::VideoUpscaleJob],
    filename: &str,
    model: &str,
) -> Option<mold_core::VideoUpscaleJob> {
    jobs.iter()
        .filter(|job| {
            job.model == model
                && matches!(
                    job.state,
                    mold_core::VideoUpscaleJobState::Queued
                        | mold_core::VideoUpscaleJobState::Running
                        | mold_core::VideoUpscaleJobState::Finalizing
                        | mold_core::VideoUpscaleJobState::Paused
                )
                && matches!(
                    &job.source,
                    mold_core::VideoUpscaleSource::Library { filename: source }
                        if source == filename
                )
        })
        .max_by_key(|job| job.updated_at_ms)
        .cloned()
}

impl App {
    pub fn new(host: Option<String>, local: bool, picker: Picker) -> Result<Self> {
        Self::new_with_launch_policy(
            host,
            local,
            picker,
            std::env::var("MOLD_API_KEY").ok(),
            false,
        )
    }

    pub(crate) fn new_with_launch_policy(
        host: Option<String>,
        local: bool,
        picker: Picker,
        api_key: Option<String>,
        strict_host: bool,
    ) -> Result<Self> {
        let config = Config::load_or_default();
        let api_key = api_key.filter(|key| !key.is_empty());

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
            } else if strict_host {
                anyhow::bail!(
                    "mold host {url} is unreachable; Library grid did not fall back to local files"
                )
            } else {
                (None, InferenceMode::Local)
            }
        } else if let Some(h) = env_host {
            let url = mold_core::client::normalize_host(&h);
            if check_server_health(&url) {
                (Some(url), InferenceMode::Auto)
            } else if strict_host {
                anyhow::bail!(
                    "mold host {url} is unreachable; Library grid did not fall back to local files"
                )
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

        if strict_host {
            if let Some(url) = server_url.as_deref() {
                check_gallery_access(url, api_key.as_deref()).with_context(|| {
                    format!(
                        "mold host {url} did not accept the Library credential; Library grid did not fall back to local files"
                    )
                })?;
            }
        }

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

        // Restore the Advanced accordion where the user left it.
        let advanced = crate::ui::create_form::AdvancedState::load();

        // Build initial catalog — try server first if connected, fall back to local
        let catalog = if let Some(ref url) = server_url {
            // Blocking fetch from server for startup catalog
            let rt = tokio::runtime::Handle::current();
            let url_clone = url.clone();
            let api_key_clone = api_key.clone();
            std::thread::spawn(move || {
                rt.block_on(async {
                    let client = crate::hosts::client_for(&url_clone, api_key_clone.as_deref());
                    client.list_models_extended().await.ok()
                })
            })
            .join()
            .ok()
            .flatten()
            .unwrap_or_else(|| build_local_model_catalog(&config))
        } else {
            build_local_model_catalog(&config)
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
        } else {
            // Model not found — only apply non-model-specific settings.
            // Skip width/height/steps/guidance/scheduler since they belong to
            // the missing model and would be wrong for the current default.
            session.apply_non_model_params(&mut params);
        }

        let family = family_for_model(&params.model, &config);
        normalize_generate_params_for_family(&mut params, &family);
        let selected_catalog_entry = catalog.iter().find(|model| model.name == params.model);
        let mut capabilities = capabilities_for_model(
            &family,
            &params.model,
            catalog
                .iter()
                .find(|model| model.name == params.model)
                .and_then(|model| model.supports_audio),
            catalog
                .iter()
                .find(|model| model.name == params.model)
                .and_then(|model| model.guidance_capabilities),
            catalog
                .iter()
                .find(|model| model.name == params.model)
                .and_then(|model| model.source_image),
            catalog
                .iter()
                .find(|model| model.name == params.model)
                .and_then(|model| model.supports_identity),
        );
        crate::model_info::apply_recipe_capabilities(
            &mut capabilities,
            selected_catalog_entry
                .and_then(|entry| entry.generation_profile.as_ref())
                .and_then(|profile| profile.recipe_for_pipeline(params.pipeline))
                .map(|recipe| &recipe.capabilities),
        );
        if capabilities.mesh.is_none() {
            params.mesh = mold_core::MeshRequestOptions::default();
        }
        capabilities.supports_duration_prediction = selected_catalog_entry.is_some_and(|entry| {
            entry.supports_duration_prediction == Some(true) && entry.runtime_ready != Some(false)
        });
        params.duration_prediction_supported = capabilities.supports_duration_prediction;
        if !params.duration_prediction_supported {
            params.predict_duration = false;
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

        // #787 round 2: resolve the selected model's advertised default at
        // startup — `sync_generate_capabilities` only runs on later changes,
        // so without this a remembered wan model cold-starts with an empty
        // default, no prefill, and no expressible opt-out. The catalog row
        // wins (local catalogs already plant the family constant); the
        // family constant backs a bare config-only model.
        let negative_default = crate::ui::create_form::effective_negative_default(
            catalog
                .iter()
                .find(|entry| entry.name == params.model)
                .and_then(|entry| entry.defaults.default_negative_prompt.as_deref()),
            &family,
        );
        let negative_prompt = negative_prompt_textarea(&restored_negative_editor_text(
            &session.last_negative,
            session.negative_cleared,
            &negative_default,
        ));
        // #787 round 3: the session's persisted marker stays live authority,
        // so a restored explicit clear survives later default reconciliation
        // (a fresher catalog row landing) and still serializes as `""`.
        let negative_explicit_clear =
            session.negative_cleared == Some(true) && session.last_negative.trim().is_empty();

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
        let machines = crate::hosts::MachinesState::load();
        let session_api_key_host_id = server_url.as_deref().and_then(|url| {
            api_key.as_ref().map(|_| {
                if crate::gallery_scan::is_loopback_url(url) {
                    crate::hosts::LOCAL_HOST_ID.to_string()
                } else {
                    machines
                        .registry
                        .hosts
                        .iter()
                        .find(|host| host.url == url)
                        .map(|host| host.id.clone())
                        .unwrap_or_else(|| crate::hosts::host_id_from_url(url))
                }
            })
        });
        let mut app = Self {
            active_view: View::Create,
            create_mode: CreateMode::default(),
            generate: GenerateState {
                prompt,
                negative_prompt,
                negative_default,
                negative_explicit_clear,
                params,
                focus: GenerateFocus::Prompt,
                param_index: 0,
                rows,
                advanced,
                param_scroll: 0,
                capabilities,
                progress: ProgressState::default(),
                live_preview_image: None,
                live_preview_protocol: None,
                preview_image: None,
                image_state: None,
                animation: None,
                generating: false,
                batch_remaining: 0,
                last_seed: None,
                last_generation_time_ms: None,
                error_message: None,
                identity_error: None,
                warning_message: None,
                model_description,
                last_output_path: None,
                held_batch: None,
                prompt_transform_token: 0,
                last_mesh_summary: None,
            },
            gallery: GalleryState::default(),
            models: ModelsState {
                catalog,
                selected: 0,
                filter: String::new(),
                filtering: false,
            },
            machines,
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
            server_status_poll_in_flight: false,
            history,
            layout: LayoutAreas::default(),
            server_process,
            session_api_key_host_id: None,
            strict_gallery_authority: strict_host,
            upscale_in_progress: false,
            upscale_task: None,
            upscale_tile_progress: None,
            upscale_progress: ProgressState::default(),
            connecting: false,
            show_timeline,
        };

        if let (Some(host_id), Some(key)) = (session_api_key_host_id, api_key.as_deref()) {
            crate::hosts::set_session_api_key(&host_id, key);
            app.session_api_key_host_id = Some(host_id);
        }

        // Spawn background gallery scan
        app.spawn_gallery_scan();

        Ok(app)
    }

    /// Spawn a merged all-hosts gallery scan: the local output dir, the
    /// connected server (when set), and every registered Machines host,
    /// fetched concurrently with per-host API keys. Offline hosts
    /// contribute nothing but don't break the scan.
    pub fn spawn_gallery_scan(&self) {
        let tx = self.bg_tx.clone();
        let sources =
            crate::gallery_scan::scan_sources(self.server_url.as_deref(), &self.machines.registry);
        let allow_local_fallback = !self.strict_gallery_authority;
        self.tokio_handle.spawn(async move {
            let scan = crate::gallery_scan::scan_all_hosts(sources, allow_local_fallback).await;
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
    pub fn spawn_server_status_fetch(&mut self) {
        if self.server_status_poll_in_flight {
            return;
        }
        let Some(ref url) = self.server_url else {
            return;
        };
        self.server_status_poll_in_flight = true;
        let tx = self.bg_tx.clone();
        let url = url.clone();
        self.tokio_handle.spawn(async move {
            let client = mold_core::MoldClient::new(&url);
            let status = tokio::time::timeout(crate::hosts::POLL_INTERVAL, client.server_status());
            let devices = tokio::time::timeout(crate::hosts::POLL_INTERVAL, client.devices());
            let capabilities =
                tokio::time::timeout(crate::hosts::POLL_INTERVAL, client.server_capabilities());
            let (status, devices, capabilities) = tokio::join!(status, devices, capabilities);
            match status.ok().and_then(|result| result.ok()) {
                Some(status) => {
                    let _ = tx.send(BackgroundEvent::ServerStatusUpdate(Some(Box::new(status))));
                    let _ = tx.send(BackgroundEvent::HostDevicesUpdate {
                        host_id: crate::hosts::LOCAL_HOST_ID.to_string(),
                        devices: devices.ok().and_then(|result| result.ok()),
                    });
                    let _ = tx.send(BackgroundEvent::HostCapabilitiesUpdate {
                        host_id: crate::hosts::LOCAL_HOST_ID.to_string(),
                        capabilities: capabilities
                            .ok()
                            .and_then(|result| result.ok())
                            .map(Box::new),
                    });
                }
                None => {
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
            let _ = tx.send(BackgroundEvent::ServerStatusPollFinished);
        });
    }

    /// Multi-host polling tick — driven by the 2 s resource-refresh timer.
    /// The plan (which hosts, whether a queue fetch rides along) lives in
    /// `hosts::MachinesState::poll_plan`.
    pub fn tick_host_polling(&mut self) {
        let plan = self
            .machines
            .poll_plan(self.active_view == View::Machines, &self.target);
        for request in plan.hosts {
            self.tokio_handle.spawn(crate::hosts::fetch_host_poll(
                request.entry,
                request.include_queue,
                self.bg_tx.clone(),
            ));
        }
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
    /// guidance, size, and video timing.
    fn apply_remote_model_defaults(&mut self, catalog: &[ModelInfoExtended]) {
        let model_name = &self.generate.params.model;
        if let Some(entry) = catalog.iter().find(|m| &m.name == model_name) {
            self.generate.params.steps = entry.defaults.default_steps;
            self.generate.params.guidance = entry.defaults.default_guidance;
            self.generate.params.width = entry.defaults.default_width;
            self.generate.params.height = entry.defaults.default_height;
            if let Some(frames) = entry.defaults.default_frames {
                self.generate.params.frames = frames;
            }
            if let Some(fps) = entry.defaults.default_fps {
                self.generate.params.fps = fps;
            }
            if !entry.defaults.description.is_empty() {
                self.generate.model_description = entry.defaults.description.clone();
            }
        }
        self.sync_generate_capabilities();
    }

    /// The selected checkpoint's source-image contract (#772): the server's
    /// advertised classification first, then the built-in manifest, whose
    /// tiers are structural and so answer for a server too old to advertise
    /// the field. `None` — an installed catalog checkpoint neither could
    /// classify — is read as "unknown", never as a contract, and leaves the
    /// name heuristic in [`capabilities_for_model`] as the last resort.
    fn source_image_contract(&self, model: &str) -> Option<mold_core::SourceImageCapability> {
        self.models
            .catalog
            .iter()
            .find(|entry| entry.name == model)
            .and_then(|entry| entry.source_image)
            .or_else(|| {
                mold_core::manifest::find_manifest(model)
                    .and_then(|manifest| manifest.defaults.source_image)
            })
    }

    /// Why an attached identity photo cannot be submitted against the current
    /// form, or `None` when it can.
    ///
    /// Every rule `mold_core::identity::validate_identity_conditioning` would
    /// apply to the *shape* of the request, asked here so the refusal is
    /// inline on the Photo row instead of a round trip away: the model gate,
    /// the LoRA and img2img pairings the milestone does not qualify, and both
    /// knob ranges. The range checks are unreachable from the rows, which
    /// cannot express an out-of-range value, but a restored session or a
    /// gallery reuse can carry one no row produced.
    ///
    /// Deliberately does NO file I/O — it runs on every model switch, and
    /// re-reading a 16 MiB photo there would be felt. The file is re-checked
    /// once at dispatch by [`Self::identity_dispatch_error`].
    fn identity_gate_error(&self) -> Option<String> {
        let params = &self.generate.params;
        params.identity_image_path.as_ref()?;
        if !self.generate.capabilities.supports_identity {
            return Some(mold_core::identity::identity_model_gate_message(
                &params.model,
            ));
        }
        // Neither pairing is qualified in milestone 1. The Create form can
        // hold both at once — LoRA and Source are their own sections — so
        // this is a reachable state, not a defensive check.
        if params.lora_path.is_some() {
            return Some(mold_core::identity::IDENTITY_LORA_CONFLICT.to_string());
        }
        if params.source_image_path.is_some() {
            return Some(mold_core::identity::IDENTITY_IMG2IMG_CONFLICT.to_string());
        }
        if let Err(message) = mold_core::identity::validate_id_weight(params.id_weight) {
            return Some(message);
        }
        if let Err(message) =
            mold_core::identity::validate_id_start_step(params.id_start_step, params.steps)
        {
            return Some(message);
        }
        None
    }

    /// The dispatch-time identity check: everything [`Self::identity_gate_error`]
    /// asks, plus one re-read of the photo itself.
    ///
    /// A file accepted at entry can be deleted, truncated, or swapped for a
    /// symlink before Generate is pressed. Every other conditioning input
    /// degrades to "absent" in that case; an identity reference must not,
    /// because the run would then render an ordinary print with a plausible
    /// wrong face and say nothing. `build_request` refuses the same way as a
    /// last line of defence — the file can still vanish between here and the
    /// read — but checking here is what puts the reason on the Photo row.
    fn identity_dispatch_error(&self) -> Option<String> {
        if let Some(message) = self.identity_gate_error() {
            return Some(message);
        }
        let path = self.generate.params.identity_image_path.as_deref()?;
        crate::identity::load_identity_image(path).err()
    }

    /// Recompute Create rows from the selected model's family and the current
    /// catalog's checkpoint-specific audio, guidance, and source-image facts.
    /// An incompatible model clears stale audio, source image, plus LTX-2
    /// pipeline and latent-upscale overrides before they can leak into
    /// another family.
    fn sync_generate_capabilities(&mut self) {
        let model = self.generate.params.model.clone();
        let family = family_for_model(&model, &self.config);
        let advertised_audio_support = self
            .models
            .catalog
            .iter()
            .find(|entry| entry.name == model)
            .and_then(|entry| entry.supports_audio);
        let advertised_guidance = self
            .models
            .catalog
            .iter()
            .find(|entry| entry.name == model)
            .and_then(|entry| entry.guidance_capabilities);
        let effective_guidance = self
            .generate
            .params
            .pipeline
            .map(|pipeline| {
                mold_core::GuidanceCapabilities::for_recipe(&family, &model, Some(pipeline))
            })
            .or(advertised_guidance);
        self.generate.capabilities = capabilities_for_model(
            &family,
            &model,
            advertised_audio_support,
            effective_guidance,
            self.source_image_contract(&model),
            self.models
                .catalog
                .iter()
                .find(|entry| entry.name == model)
                .and_then(|entry| entry.supports_identity),
        );
        // The resolved recipe profile is layered last: it is the single
        // authority for the 3-D rows and, on a mesh recipe, for the
        // strength / mask / negative gates.
        let recipe = self.active_generation_recipe();
        crate::model_info::apply_recipe_capabilities(
            &mut self.generate.capabilities,
            recipe.as_ref().map(|recipe| &recipe.capabilities),
        );
        self.generate.capabilities.supports_duration_prediction = self
            .models
            .catalog
            .iter()
            .find(|entry| entry.name == model)
            .is_some_and(|entry| {
                entry.supports_duration_prediction == Some(true)
                    && entry.runtime_ready != Some(false)
            });
        self.generate.params.duration_prediction_supported =
            self.generate.capabilities.supports_duration_prediction;
        // The mesh rows are gone, so a carried-in octree or iso-level has no
        // editor left to clear it — and a raster recipe refuses the block
        // outright rather than ignoring it.
        if self.generate.capabilities.mesh.is_none() {
            self.generate.params.mesh = mold_core::MeshRequestOptions::default();
        }
        if !self.generate.params.duration_prediction_supported {
            self.generate.params.predict_duration = false;
        }
        // #787: keep the Negative editor and its advertised default in step
        // with the selected model. The server's per-model advertisement wins;
        // the family constant covers local mode and older servers (it *is*
        // the engine's absence fallback). Only an editor still showing the
        // previous default follows the change — typed text and an explicit
        // clear are user authority.
        let next_default = crate::ui::create_form::effective_negative_default(
            self.models
                .catalog
                .iter()
                .find(|entry| entry.name == model)
                .and_then(|entry| entry.defaults.default_negative_prompt.as_deref()),
            &family,
        );
        if let Some(replacement) = crate::ui::create_form::negative_prompt_on_default_change(
            &self.generate.negative_prompt.lines().join("\n"),
            &self.generate.negative_default,
            &next_default,
            // A restored explicit clear (#787 round 3) is user authority even
            // while the tracked default was still empty at restore time.
            self.generate.negative_explicit_clear,
        ) {
            self.generate.negative_prompt = negative_prompt_textarea(&replacement);
        }
        self.generate.negative_default = next_default;
        normalize_generate_params_for_family(&mut self.generate.params, &family);
        if !self.generate.capabilities.supports_audio {
            self.generate.params.enable_audio = None;
        }
        // The Source row is gone, so a path carried in from persistence or a
        // gallery reuse has no editor left to clear it — and submitting it
        // would only earn a rejection the user cannot act on.
        if !self.generate.capabilities.supports_source_image {
            self.generate.params.source_image_path = None;
        }
        if !self.generate.capabilities.supports_references {
            self.generate.params.reference_paths.clear();
        }
        // `id_start_step` is bounded by the step count, which every model
        // switch can move; a restored 20 against a 4-step model would be
        // refused at admission for a value the form never let the user set.
        // The clamp runs BEFORE the gate below, so a value this repair fixes
        // never leaves a refusal on screen for a state that no longer exists.
        let step_ceiling = self.generate.params.steps.saturating_sub(1);
        self.generate.params.id_start_step = self.generate.params.id_start_step.min(step_ceiling);
        // Identity is the one conditioning reference a model switch does NOT
        // discard. Dropping a face silently would be worse than the stale
        // source-image path above: the print would render, look fine, and
        // simply not be that person. The photo is kept, the refusal is
        // raised, and dispatch is blocked until the user picks a qualified
        // checkpoint or clears the photo. The wording is `mold_core`'s.
        //
        // Assigned unconditionally, so a gate that now passes clears the
        // previous refusal rather than leaving it stale on the row.
        self.generate.identity_error = self.identity_gate_error();
        if !self.generate.capabilities.supports_video_upscale {
            self.generate.params.pipeline = None;
            self.generate.params.spatial_upscale = None;
            self.generate.params.temporal_upscale = None;
            self.generate.params.guidance_overrides = Ltx2GuidanceOverrides::default();
        }
        // Leaving wan clears its flow-shift override the same way leaving
        // LTX-2 clears the guider overrides above (#782).
        if !self.generate.capabilities.supports_flow_shift {
            self.generate.params.sample_shift = None;
        }
        self.refresh_create_rows();
    }

    /// Materialize a fixed recipe's effective guidance only in the frozen
    /// request. The live TUI form retains its guided value so switching back
    /// to an adjustable recipe restores the user's setting.
    fn normalize_fixed_guidance_for_submit(&self, params: &mut GenerateParams) {
        let family = family_for_model(&params.model, &self.config);
        let advertised = self
            .models
            .catalog
            .iter()
            .find(|entry| entry.name == params.model)
            .and_then(|entry| entry.guidance_capabilities);
        let capabilities = params
            .pipeline
            .map(|pipeline| {
                mold_core::GuidanceCapabilities::for_recipe(&family, &params.model, Some(pipeline))
            })
            .or(advertised)
            .unwrap_or_else(|| {
                mold_core::GuidanceCapabilities::for_recipe(&family, &params.model, None)
            });
        if let Some(fixed_scale) = capabilities.fixed_scale {
            params.guidance = fixed_scale;
        }
    }

    /// Re-resolve the primary guidance/negative-prompt contract after the
    /// user changes an explicit LTX-2 recipe. Returning to Auto must restore
    /// the server-advertised checkpoint contract instead of inferring from an
    /// opaque catalog ID, while preserving the selected row across reflow.
    fn sync_pipeline_guidance(&mut self) {
        if !self.generate.capabilities.supports_video_upscale {
            return;
        }
        let selected_row = self.generate.rows.get(self.generate.param_index).copied();
        self.sync_generate_capabilities();
        if let Some(selected_row) = selected_row {
            if let Some(index) = self
                .generate
                .rows
                .iter()
                .position(|row| *row == selected_row)
            {
                self.generate.param_index = index;
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

        if crate::gallery_scan::is_video_filename(&entry.filename()) {
            let filename = entry.filename();
            let handle = self.tokio_handle.spawn(async move {
                let Some(url) = server_url else {
                    let _ = tx.send(BackgroundEvent::UpscaleFailed(
                        "Framewise video upscale requires a running Mold server".into(),
                    ));
                    return;
                };
                let api_key = server_host_id
                    .as_deref()
                    .and_then(crate::hosts::api_key_for);
                let client = crate::hosts::client_for(&url, api_key.as_deref());
                let request = mold_core::CreateVideoUpscaleJobRequest {
                    source: mold_core::VideoUpscaleSource::Library {
                        filename: filename.clone(),
                    },
                    model: model_name,
                    tile_size: None,
                };
                let recoverable = match client.list_video_upscale_jobs().await {
                    Ok(jobs) => recoverable_framewise_upscale(&jobs, &filename, &request.model),
                    Err(error) => {
                        let _ = tx.send(BackgroundEvent::UpscaleFailed(error.to_string()));
                        return;
                    }
                };
                let mut job = match recoverable {
                    Some(job) if job.state == mold_core::VideoUpscaleJobState::Paused => {
                        match client.transition_video_upscale_job(&job.id, "resume").await {
                            Ok(job) => job,
                            Err(error) => {
                                let _ = tx.send(BackgroundEvent::UpscaleFailed(error.to_string()));
                                return;
                            }
                        }
                    }
                    Some(job) => job,
                    None => match client.create_video_upscale_job(&request).await {
                        Ok(job) => job,
                        Err(error) => {
                            let _ = tx.send(BackgroundEvent::UpscaleFailed(error.to_string()));
                            return;
                        }
                    },
                };
                let id = job.id.clone();
                let _ = tx.send(BackgroundEvent::FramewiseUpscaleStatus(job.clone()));
                while !job.state.is_terminal() {
                    tokio::time::sleep(std::time::Duration::from_millis(750)).await;
                    match client.get_video_upscale_job(&id).await {
                        Ok(next) => {
                            job = next;
                            let _ = tx.send(BackgroundEvent::FramewiseUpscaleStatus(job.clone()));
                        }
                        Err(error) => {
                            let _ = tx.send(BackgroundEvent::UpscaleFailed(error.to_string()));
                            return;
                        }
                    }
                }
            });
            self.upscale_task = Some(handle);
            return;
        }

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
                            | mold_core::SseProgressEvent::StageProgress { .. }
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
                    Some(&config.resolved_models_dir()),
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

        self.cleanup_runtime_authority();
    }

    fn cleanup_runtime_authority(&mut self) {
        if let Some(ref mut child) = self.server_process {
            tracing::info!(pid = child.id(), "stopping background server");
            let _ = child.kill();
            let _ = child.wait();
        }
        self.server_process = None;
        if let Some(host_id) = self.session_api_key_host_id.take() {
            crate::hosts::clear_session_api_key(&host_id);
        }
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
                .with_theme(self.settings.theme_preset)
                // An empty editor while a default is advertised is the
                // explicit "" opt-out; the marker is what keeps it
                // distinguishable from "untouched" across restarts (#787).
                // A deferred restore-time clear (#787 round 3) persists the
                // same way even while no default is known yet.
                .with_negative_cleared(
                    neg_text.is_empty()
                        && (self.generate.negative_explicit_clear
                            || !self.generate.negative_default.trim().is_empty()),
                );
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

        // Switching models is fresh authority: a deferred restore-time clear
        // marker (#787 round 3) belonged to the previous selection. Gallery
        // reuse re-arms it after this call when the print recorded `""`.
        self.generate.negative_explicit_clear = false;
        self.generate.params.model = model_name.clone();

        // Use server catalog defaults when connected to a remote server,
        // local config otherwise.
        let used_remote = if self.should_poll_remote() {
            if let Some(entry) = self.models.catalog.iter().find(|m| m.name == model_name) {
                self.generate.params.steps = entry.defaults.default_steps;
                self.generate.params.guidance = entry.defaults.default_guidance;
                self.generate.params.width = entry.defaults.default_width;
                self.generate.params.height = entry.defaults.default_height;
                if let Some(frames) = entry.defaults.default_frames {
                    self.generate.params.frames = frames;
                }
                if let Some(fps) = entry.defaults.default_fps {
                    self.generate.params.fps = fps;
                }
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
            if let Some(frames) = model_cfg.effective_frames() {
                self.generate.params.frames = frames;
            }
            if let Some(fps) = model_cfg.effective_fps() {
                self.generate.params.fps = fps;
            }

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
        // Apply saved per-model prefs last, so a user's explicit choices
        // override the manifest/catalog defaults we just restored. Only
        // generation params move — the prompt textareas stay as-is so a
        // model flip mid-typing doesn't wipe what the user is writing.
        self.apply_prefs_for_model(&model_name);
        // Capabilities and family-fixed request fields are authoritative over
        // persistence. In particular, an old H3 preference row may not
        // restore PNG, CFG guidance, strength, or an off-grid frame count.
        self.sync_generate_capabilities();
        self.generate.param_index = 0;
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
            p.format = f.parse().unwrap_or(mold_core::OutputFormat::Png);
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

    /// Fully resolved recipe advertised for the selected model and pipeline.
    /// Cloning keeps the keyboard mutation path borrow-simple; profiles are
    /// small catalog metadata, not runtime model state.
    fn active_generation_recipe(&self) -> Option<mold_core::GenerationRecipeProfile> {
        self.models
            .catalog
            .iter()
            .find(|entry| entry.name == self.generate.params.model)?
            .generation_profile
            .as_ref()?
            .recipe_for_pipeline(self.generate.params.pipeline)
            .cloned()
    }

    /// Whether Generate needs a non-empty prompt right now.
    ///
    /// The selected recipe's profile answers first (`capabilities.prompt`):
    /// `Ignored` admits an empty prompt outright, `Optional` admits one only
    /// with the source image that makes it optional (the advertised mode
    /// describes the CONDITIONED request), and `Required` is required. A
    /// family-only catalog with no profile falls back to the same core
    /// function the profile was emitted from, so the two can never disagree
    /// and neither carries a family allowlist.
    fn prompt_required_now(&self) -> bool {
        let has_source = self.generate.params.source_image_path.is_some();
        match self
            .active_generation_recipe()
            .map(|recipe| recipe.capabilities.prompt.mode)
        {
            Some(mold_core::PromptRequirement::Ignored) => false,
            Some(mold_core::PromptRequirement::Optional) => !has_source,
            Some(mold_core::PromptRequirement::Required) => true,
            None => prompt_required_for_params(&self.generate.params, &self.config),
        }
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
                        (KeyCode::Char('e' | 'E'), modifiers)
                            if modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            // Fall through to Expand/Remix action mapping.
                        }
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
                                if let Some(prompt) =
                                    self.history.prev(&current).map(str::to_owned)
                                {
                                    self.set_authored_prompt_text(&prompt);
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
                                if let Some(prompt) =
                                    self.history.next(&current).map(str::to_owned)
                                {
                                    self.set_authored_prompt_text(&prompt);
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
                            let previous_prompt = if self.generate.focus == GenerateFocus::Prompt {
                                Some(self.generate.prompt.lines().join("\n"))
                            } else {
                                None
                            };
                            let textarea = match self.generate.focus {
                                GenerateFocus::Prompt => &mut self.generate.prompt,
                                GenerateFocus::NegativePrompt => &mut self.generate.negative_prompt,
                                _ => unreachable!(),
                            };
                            textarea.input(event);
                            let prompt_changed = previous_prompt.is_some_and(|previous| {
                                previous != self.generate.prompt.lines().join("\n")
                            });
                            if prompt_changed {
                                self.generate.prompt_transform_token =
                                    self.generate.prompt_transform_token.wrapping_add(1);
                                self.retire_prompt_provenance();
                            }
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
        if let Some(Popup::LicenseReview { response, .. }) = self.popup.as_mut() {
            if let Some(response) = response.take() {
                let _ = response.send(false);
            }
        }
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
            if let Some(Popup::PromptSourceChoice {
                current_prompt,
                root_prompt,
                cursor,
            }) = &mut self.popup
            {
                match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Up | KeyCode::Down | KeyCode::Char('j' | 'k') => {
                        *cursor = 1usize.saturating_sub(*cursor);
                    }
                    KeyCode::Char('o' | 'O') => {
                        let source = root_prompt.clone();
                        self.close_popup();
                        self.start_prompt_transform_from(
                            PromptTransformOperation::Remix,
                            source,
                            mold_core::RemixSourceKind::Original,
                        );
                    }
                    KeyCode::Char('c' | 'C') => {
                        let source = current_prompt.clone();
                        self.close_popup();
                        self.start_prompt_transform_from(
                            PromptTransformOperation::Remix,
                            source,
                            mold_core::RemixSourceKind::Current,
                        );
                    }
                    KeyCode::Enter => {
                        let (source, kind) = if *cursor == 0 {
                            (root_prompt.clone(), mold_core::RemixSourceKind::Original)
                        } else {
                            (current_prompt.clone(), mold_core::RemixSourceKind::Current)
                        };
                        self.close_popup();
                        self.start_prompt_transform_from(
                            PromptTransformOperation::Remix,
                            source,
                            kind,
                        );
                    }
                    _ => {}
                }
                return;
            }
            enum PromptAlternativeEffect {
                None,
                Close,
                Apply {
                    snapshot: PromptTransformSnapshot,
                    variant: mold_core::RemixVariant,
                },
                Prepare {
                    snapshot: PromptTransformSnapshot,
                    variants: Vec<mold_core::RemixVariant>,
                },
                Retry(PromptTransformOperation),
                Restore(String),
            }
            let handled_prompt_popup = matches!(self.popup, Some(Popup::PromptAlternatives { .. }));
            let effect = if let Some(Popup::PromptAlternatives {
                snapshot,
                variants,
                selected,
                cursor,
            }) = &mut self.popup
            {
                match key.code {
                    KeyCode::Esc => PromptAlternativeEffect::Close,
                    KeyCode::Up | KeyCode::Char('k') if *cursor > 0 => {
                        *cursor -= 1;
                        PromptAlternativeEffect::None
                    }
                    KeyCode::Down | KeyCode::Char('j') if *cursor + 1 < variants.len() => {
                        *cursor += 1;
                        PromptAlternativeEffect::None
                    }
                    KeyCode::Char(' ') => {
                        if let Some(value) = selected.get_mut(*cursor) {
                            *value = !*value;
                        }
                        PromptAlternativeEffect::None
                    }
                    KeyCode::Enter => variants
                        .get(*cursor)
                        .cloned()
                        .map(|variant| PromptAlternativeEffect::Apply {
                            snapshot: snapshot.clone(),
                            variant,
                        })
                        .unwrap_or(PromptAlternativeEffect::None),
                    KeyCode::Char('b' | 'B') => {
                        let variants = variants
                            .iter()
                            .zip(selected.iter())
                            .filter_map(|(variant, selected)| selected.then_some(variant.clone()))
                            .collect::<Vec<_>>();
                        if variants.len() < 2 {
                            PromptAlternativeEffect::None
                        } else {
                            PromptAlternativeEffect::Prepare {
                                snapshot: snapshot.clone(),
                                variants,
                            }
                        }
                    }
                    KeyCode::Char('r' | 'R') => PromptAlternativeEffect::Retry(snapshot.operation),
                    KeyCode::Char('u' | 'U') => {
                        PromptAlternativeEffect::Restore(snapshot.source_prompt.clone())
                    }
                    _ => PromptAlternativeEffect::None,
                }
            } else {
                PromptAlternativeEffect::None
            };
            match effect {
                PromptAlternativeEffect::None => {}
                PromptAlternativeEffect::Close => self.close_popup(),
                PromptAlternativeEffect::Apply { snapshot, variant } => {
                    if let Some(message) = self.prompt_transform_staleness(&snapshot) {
                        self.generate.error_message = Some(message);
                        self.close_popup();
                    } else {
                        self.apply_prompt_variant(snapshot, variant);
                        self.close_popup();
                    }
                }
                PromptAlternativeEffect::Prepare { snapshot, variants } => {
                    if let Some(message) = self.prompt_transform_staleness(&snapshot) {
                        self.generate.error_message = Some(message);
                        self.close_popup();
                    } else {
                        self.prepare_prompt_variants(snapshot, variants);
                        self.close_popup();
                    }
                }
                PromptAlternativeEffect::Retry(operation) => {
                    self.close_popup();
                    self.start_prompt_transform(operation);
                }
                PromptAlternativeEffect::Restore(source) => {
                    self.set_authored_prompt_text(&source);
                    self.close_popup();
                }
            }
            if handled_prompt_popup {
                return;
            }
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
                            // The client never blocks a custom size (parity with
                            // the other four shells): a recipe refusal becomes an
                            // advisory, the entry is applied, and the server's own
                            // admission is the authority at submit time.
                            let advisory = self
                                .active_generation_recipe()
                                .and_then(|recipe| mold_core::resolution_advisory(&recipe, w, h));
                            self.generate.params.width = w;
                            self.generate.params.height = h;
                            self.generate.error_message = None;
                            self.generate.warning_message = advisory;
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
                Some(Popup::StgBlocksInput { input, error }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => match parse_stg_blocks_input(input) {
                        Ok(blocks) => {
                            self.generate.params.guidance_overrides.stg_blocks = blocks;
                            self.close_popup();
                        }
                        Err(message) => *error = Some(message),
                    },
                    KeyCode::Char(c) if c.is_ascii_digit() || matches!(c, ',' | ' ') => {
                        input.push(c);
                        *error = None;
                    }
                    KeyCode::Backspace => {
                        input.pop();
                        *error = None;
                    }
                    _ => {}
                },
                Some(Popup::ReferencesInput { input, error }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => match crate::h3_references::parse_reference_input(input) {
                        Ok(references) => {
                            self.generate.params.reference_paths = references;
                            self.close_popup();
                        }
                        Err(message) => *error = Some(message),
                    },
                    KeyCode::Char(c)
                        if input.len() + c.len_utf8()
                            <= crate::h3_references::MAX_REFERENCE_INPUT_BYTES =>
                    {
                        input.push(c);
                        *error = None;
                    }
                    KeyCode::Backspace => {
                        input.pop();
                        *error = None;
                    }
                    _ => {}
                },
                Some(Popup::IdentityImageInput { input, error }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        // An emptied field clears the photo — that is the only
                        // way back out of an attached reference, and it must
                        // not have to pass a file check to get there.
                        if input.trim().is_empty() {
                            self.generate.params.identity_image_path = None;
                            self.generate.identity_error = None;
                            self.close_popup();
                        } else {
                            match crate::identity::load_identity_image(input) {
                                Ok(_) => {
                                    self.generate.params.identity_image_path =
                                        Some(input.trim().to_string());
                                    self.generate.identity_error = None;
                                    self.close_popup();
                                }
                                Err(message) => {
                                    self.generate.identity_error = Some(message.clone());
                                    *error = Some(message);
                                }
                            }
                        }
                    }
                    KeyCode::Char(c) if input.len() + c.len_utf8() <= IDENTITY_PATH_MAX_BYTES => {
                        input.push(c);
                        *error = None;
                    }
                    KeyCode::Backspace => {
                        input.pop();
                        *error = None;
                    }
                    _ => {}
                },
                Some(Popup::FilingInput {
                    field,
                    input,
                    error,
                }) => match key.code {
                    KeyCode::Esc => self.close_popup(),
                    KeyCode::Enter => {
                        let field = *field;
                        let text = input.clone();
                        match crate::ui::create_form::commit_filing_input(
                            field,
                            &text,
                            &self.generate.params,
                        ) {
                            Ok(edit) => {
                                self.apply_filing_edit(edit);
                                self.close_popup();
                            }
                            Err(message) => *error = Some(message),
                        }
                    }
                    KeyCode::Char(c) => {
                        input.push(c);
                        *error = None;
                    }
                    KeyCode::Backspace => {
                        input.pop();
                        *error = None;
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
                            self.set_authored_prompt_text(&prompt);
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
                Some(Popup::LicenseSettings {
                    state, selected, ..
                }) => {
                    let rows = match state {
                        LicenseListingState::Ready(rows) => rows.len(),
                        _ => 0,
                    };
                    match key.code {
                        KeyCode::Char('j') | KeyCode::Down if *selected + 1 < rows => {
                            *selected += 1;
                        }
                        KeyCode::Char('k') | KeyCode::Up => {
                            *selected = selected.saturating_sub(1);
                        }
                        KeyCode::Esc | KeyCode::Char('q') => self.close_popup(),
                        _ => {}
                    }
                }
                Some(Popup::LicenseReview { response, .. }) => match key.code {
                    KeyCode::Char('y') | KeyCode::Enter => {
                        let response = response.take();
                        self.popup = None;
                        self.refresh_preview_protocol();
                        if let Some(response) = response {
                            let _ = response.send(true);
                        }
                    }
                    KeyCode::Esc | KeyCode::Char('n') => self.close_popup(),
                    _ => {}
                },
                Some(Popup::Info { .. }) => {
                    // Dismiss info popup on any key
                    self.close_popup();
                }
                Some(Popup::MeshExportPicker {
                    filename,
                    formats,
                    selected,
                }) => match key.code {
                    KeyCode::Char('j') | KeyCode::Down if *selected + 1 < formats.len() => {
                        *selected += 1;
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        *selected = selected.saturating_sub(1);
                    }
                    KeyCode::Enter => {
                        let filename = filename.clone();
                        let format = formats.get(*selected).copied();
                        self.close_popup();
                        if let Some(format) = format {
                            self.spawn_mesh_export(&filename, format);
                        }
                    }
                    KeyCode::Esc | KeyCode::Char('q') => self.close_popup(),
                    _ => {}
                },
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
                Some(Popup::PromptAlternatives { .. } | Popup::PromptSourceChoice { .. }) => {
                    unreachable!("handled above")
                }
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
                            // Double-click: select model and switch to Generate —
                            // unless the build has no engine arm for it, in which
                            // case double-clicking would only queue a request the
                            // server refuses with a 501. Keep the row selected
                            // (this inventory view still lists it) and say why
                            // inline instead of silently jumping to Create.
                            if was_selected {
                                let model = &self.models.catalog[relative_row];
                                if model.runtime_available == Some(false) {
                                    self.generate.error_message =
                                        Some(runtime_unavailable_message(model));
                                } else {
                                    let name = model.name.clone();
                                    self.update_model(&name);
                                    self.active_view = View::Create;
                                    self.generate.focus = GenerateFocus::Prompt;
                                }
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
            *filtered = generation_model_names(&self.models.catalog, filter);
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

    pub(crate) fn open_library(&mut self) {
        self.set_active_view(View::Library);
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
                            self.refresh_selected_host_queue();
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
                            self.refresh_selected_host_queue();
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
                        let host_label = self
                            .resource_info
                            .server_status
                            .as_ref()
                            .and_then(|status| status.hostname.clone())
                            .unwrap_or_else(|| url.clone());
                        self.tokio_handle.spawn(async move {
                            let client = mold_core::MoldClient::new(&url);
                            match crate::backend::pull_remote_model_with_consent(
                                &client,
                                host_label,
                                model_name.clone(),
                                tx.clone(),
                            )
                            .await
                            {
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
                            if let Err(msg) = crate::backend::pull_local_model_with_consent(
                                model_name.clone(),
                                tx.clone(),
                            )
                            .await
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
            Action::ExpandPrompt if self.active_view == View::Create => {
                self.start_prompt_transform(PromptTransformOperation::Expand);
            }
            Action::RemixPrompt if self.active_view == View::Create => {
                self.start_prompt_transform(PromptTransformOperation::Remix);
            }
            Action::RetryHeldPrints if self.active_view == View::Create => {
                self.retry_held_prints();
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
                if let Some(prompt) = self.history.prev(&current).map(str::to_owned) {
                    self.set_authored_prompt_text(&prompt);
                }
            }
            Action::HistoryNext
                if self.active_view == View::Create
                    && self.generate.focus == GenerateFocus::Prompt =>
            {
                let current = self.generate.prompt.lines().join("\n");
                if let Some(prompt) = self.history.next(&current).map(str::to_owned) {
                    self.set_authored_prompt_text(&prompt);
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
            Action::ReviewLicenses => {
                let host_label = if self.should_poll_remote() {
                    self.server_url.clone().unwrap_or_else(|| "server".into())
                } else {
                    "This device".to_string()
                };
                self.popup = Some(Popup::LicenseSettings {
                    host_label: host_label.clone(),
                    state: LicenseListingState::Loading,
                    selected: 0,
                });
                let tx = self.bg_tx.clone();
                let remote = self
                    .should_poll_remote()
                    .then(|| self.server_url.clone())
                    .flatten();
                // An authenticated host answers 401 to a bare client, which
                // would make this popup useless on exactly the fleet hosts a
                // user most needs to check. Resolve the connected server's
                // saved key the way every other authenticated TUI request
                // does.
                let api_key = remote.as_deref().and_then(|url| {
                    let machines = crate::hosts::MachinesState::load();
                    let host_id = if crate::gallery_scan::is_loopback_url(url) {
                        crate::hosts::LOCAL_HOST_ID.to_string()
                    } else {
                        machines
                            .registry
                            .hosts
                            .iter()
                            .find(|host| host.url == url)
                            .map(|host| host.id.clone())
                            .unwrap_or_else(|| crate::hosts::host_id_from_url(url))
                    };
                    crate::hosts::api_key_for(&host_id)
                });
                self.tokio_handle.spawn(async move {
                    match remote {
                        Some(url) => {
                            let client = crate::hosts::client_for(&url, api_key.as_deref());
                            match client.list_licenses().await {
                                Ok(licenses) => {
                                    let _ = tx.send(BackgroundEvent::LicenseListingLoaded {
                                        host_label,
                                        licenses,
                                    });
                                }
                                Err(error) => {
                                    // A host that cannot answer is never
                                    // papered over with local state: the
                                    // question is which machine accepted.
                                    let _ = tx.send(BackgroundEvent::LicenseListingFailed {
                                        message: format!(
                                            "{host_label} did not report its licenses: {error}"
                                        ),
                                        host_label,
                                    });
                                }
                            }
                        }
                        None => match mold_core::Config::mold_dir() {
                            Some(home) => {
                                let _ = tx.send(BackgroundEvent::LicenseListingLoaded {
                                    host_label,
                                    licenses: mold_core::license_acceptance::license_statuses(
                                        &home,
                                    ),
                                });
                            }
                            None => {
                                let _ = tx.send(BackgroundEvent::LicenseListingFailed {
                                    host_label,
                                    message:
                                        "Could not resolve this machine's Mold data directory."
                                            .to_string(),
                                });
                            }
                        },
                    }
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
                    let kind = self.removal_kind_for(entry);
                    self.request_confirm(
                        delete_confirm_message(&filename, machines, kind),
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
                let can_upscale = self
                    .gallery
                    .entries
                    .get(self.gallery.selected)
                    .is_some_and(|entry| self.can_upscale_entry(entry));
                if !can_upscale {
                    self.generate.error_message =
                        Some("Framewise video upscale is unavailable on this Mold host".into());
                    return;
                }
                let models = self.available_upscaler_models();
                self.popup = Some(Popup::UpscaleModelSelector {
                    filter: String::new(),
                    selected: 0,
                    filtered: models,
                    purpose: UpscalePickerPurpose::RunNow,
                });
            }
            Action::ExportMesh if self.active_view == View::Library => {
                self.open_mesh_export_picker();
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
            Action::MachinesToggleConnection if self.active_view == View::Machines => {
                if let crate::hosts::MachineRowId::Host(id) = self.machines.selected_row() {
                    if let Some(connected) = self.machines.toggle_connection(&id) {
                        self.gallery.dirty = true;
                        if !connected && self.target == crate::hosts::GenTarget::Host(id) {
                            self.target = crate::hosts::GenTarget::Auto;
                            self.target.save();
                        }
                        if connected {
                            self.tick_host_polling();
                        }
                    }
                }
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
                if let Some((host_id, job)) = self.machines.selected_cancellable_job() {
                    self.popup = Some(Popup::Confirm {
                        message: format!("Cancel {} job for {}?", job.state, job.model),
                        on_confirm: ConfirmAction::CancelHostJob {
                            host_id,
                            job_id: job.id,
                        },
                    });
                }
            }
            Action::MachinesLoadMoreQueue
                if self.active_view == View::Machines
                    && self.machines.focus == crate::hosts::MachinesFocus::Detail =>
            {
                if let Some((entry, limit, cursor)) = self.machines.begin_queue_continuation() {
                    self.tokio_handle.spawn(crate::hosts::fetch_host_queue_page(
                        entry,
                        limit,
                        cursor,
                        self.bg_tx.clone(),
                    ));
                }
            }
            Action::MachinesNextDevice if self.active_view == View::Machines => {
                self.machines.select_next_device();
            }
            Action::MachinesDevicePrev
                if self.active_view == View::Machines
                    && self.machines.focus == crate::hosts::MachinesFocus::Detail =>
            {
                self.machines.select_prev_device();
            }
            Action::MachinesDeviceNext
                if self.active_view == View::Machines
                    && self.machines.focus == crate::hosts::MachinesFocus::Detail =>
            {
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
                // An advisory describes the print that produced it.
                self.generate.warning_message = None;
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
        let active_recipe = self.active_generation_recipe();
        let size_presets = (field == ParamField::Size).then(|| {
            active_recipe.as_ref().map_or_else(
                || {
                    let (width, height) = self.model_default_size();
                    crate::ui::create_form::size_presets(width, height, 64)
                },
                |recipe| {
                    recipe
                        .resolution
                        .aspect_groups
                        .iter()
                        .flat_map(|group| &group.presets)
                        .map(|preset| (preset.width, preset.height))
                        .collect()
                },
            )
        });
        let video_grid = if matches!(
            field,
            ParamField::Duration | ParamField::Frames | ParamField::Fps
        ) {
            let entry = self
                .models
                .catalog
                .iter()
                .find(|entry| entry.name == self.generate.params.model);
            let family =
                crate::model_info::family_for_model(&self.generate.params.model, &self.config);
            let step = entry
                .and_then(|entry| entry.defaults.frame_step)
                .or_else(|| mold_core::validation::frame_step_for_family(&family))
                .unwrap_or(8)
                .max(1);
            let offset = entry
                .and_then(|entry| entry.defaults.frame_offset)
                .or_else(|| mold_core::validation::frame_offset_for_family(&family))
                .unwrap_or(1)
                .max(1);
            active_recipe
                .as_ref()
                .and_then(|recipe| recipe.temporal.as_ref())
                .map(|temporal| TuiVideoGrid {
                    step: temporal.frames.step,
                    offset: temporal.frame_offset,
                    min_frames: temporal.frames.min,
                    fixed_fps: match temporal.fps {
                        mold_core::FpsControl::Fixed { value } => Some(value),
                        mold_core::FpsControl::Adjustable { .. } => None,
                    },
                    runtime_seconds: temporal.max_duration_seconds,
                    absolute_frames: Some(temporal.frames.max),
                    fixed_frames: temporal.frames.max,
                })
                .or_else(|| {
                    Some(TuiVideoGrid {
                        step,
                        offset,
                        // Model-aware, matching the recipe path above: a
                        // compact H3 tag renders exactly one clip length, so
                        // its floor is that length and not the family's.
                        min_frames: mold_core::validation::min_frames_for_model(
                            &family,
                            &self.generate.params.model,
                        )
                        .unwrap_or(offset),
                        fixed_fps: mold_core::validation::fixed_fps_for_family(&family),
                        runtime_seconds: entry
                            .and_then(|entry| entry.defaults.max_runtime_seconds)
                            .or_else(|| {
                                mold_core::validation::max_runtime_seconds_for_family(&family)
                            }),
                        absolute_frames: entry
                            .and_then(|entry| entry.defaults.max_frames_absolute)
                            .or_else(|| {
                                mold_core::validation::max_frames_absolute_for_family(&family)
                            }),
                        fixed_frames: entry
                            .and_then(|entry| entry.defaults.max_frames)
                            .or_else(|| mold_core::validation::max_frames_for_family(&family))
                            .unwrap_or(257),
                    })
                })
        } else {
            None
        };
        let guidance_adjustable = self.generate.guidance_adjustable();
        let audio_required = self.generate.capabilities.audio_required;
        let mesh_profile = self.generate.capabilities.mesh.clone();
        let mesh_pinned = mesh_profile.is_some();
        let p = &mut self.generate.params;
        match field {
            ParamField::Size => {
                // Profile-aware servers provide exact, qualified presets.
                // The synthetic grid remains only as a one-release adapter.
                let presets = size_presets.unwrap_or_default();
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
                let (min, max, step) = active_recipe.as_ref().map_or((1, 200, 1), |recipe| {
                    (recipe.steps.min, recipe.steps.max, recipe.steps.step)
                });
                p.steps = (i64::from(p.steps) + i64::from(delta) * i64::from(step))
                    .clamp(i64::from(min), i64::from(max)) as u32;
            }
            ParamField::Guidance => {
                if !guidance_adjustable {
                    return;
                }
                let (min, max, step) = active_recipe.as_ref().map_or((0.0, 30.0, 0.5), |recipe| {
                    (
                        recipe.guidance.min,
                        recipe.guidance.max,
                        recipe.guidance.step,
                    )
                });
                p.guidance = (p.guidance + f64::from(delta) * step).clamp(min, max);
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
            ParamField::PredictDuration => {
                if p.duration_prediction_supported {
                    p.predict_duration = !p.predict_duration;
                }
            }
            ParamField::Duration => {
                let grid = video_grid.expect("duration has video grid");
                let fps = p.fps.max(1);
                let seconds = (p.frames as f64 / fps as f64 + delta as f64).max(0.1);
                let target = (seconds * fps as f64).round() as u32;
                p.frames = grid
                    .snap_nearest(target)
                    .clamp(grid.min_frames, tui_max_video_frames(grid, fps));
            }
            ParamField::Strength => {
                p.strength = (p.strength + delta as f64 * 0.05).clamp(0.0, 1.0);
            }
            // The three mesh rows walk the PROFILE's bounds, never a local
            // copy: the allowlist, the iso range and step, and the face
            // bounds all come from `capabilities.mesh`, which is the block
            // `validate_request_against_recipe` checks the request against.
            ParamField::Octree => {
                if let Some(profile) = mesh_profile.as_ref() {
                    p.mesh.octree_resolution = next_octree_resolution(
                        &profile.octree_resolutions,
                        profile.octree_default,
                        p.mesh.octree_resolution,
                        delta,
                    );
                }
            }
            ParamField::MeshThreshold => {
                if let Some(profile) = mesh_profile.as_ref() {
                    p.mesh.threshold = Some(next_mesh_threshold(
                        &profile.threshold,
                        p.mesh.threshold,
                        delta,
                    ));
                }
            }
            ParamField::TargetFaces => {
                if let Some(profile) = mesh_profile.as_ref() {
                    p.mesh.target_faces = next_target_faces(
                        profile.target_faces_min,
                        profile.target_faces_max,
                        p.mesh.target_faces,
                        delta,
                    );
                }
            }
            ParamField::IdentityWeight => {
                // The range is `mold_core::identity`'s, never a local copy.
                // Rounding to one decimal keeps repeated ◀▶ presses from
                // accumulating binary drift into the recorded provenance.
                let next = ((p.id_weight + delta as f64 * 0.1) * 10.0).round() / 10.0;
                p.id_weight = next.clamp(0.0, mold_core::identity::ID_WEIGHT_MAX);
            }
            ParamField::IdentityStartStep => {
                // `id_start_step` must stay strictly below `steps`; a form
                // that cannot express an invalid value never has to explain
                // one. `steps` is at least 1 everywhere it is adjustable.
                let ceiling = p.steps.saturating_sub(1);
                p.id_start_step = (i64::from(p.id_start_step) + i64::from(delta))
                    .clamp(0, i64::from(ceiling)) as u32;
            }
            ParamField::Frames => {
                let grid = video_grid.expect("frames has video grid");
                let current = grid
                    .snap_nearest(p.frames)
                    .clamp(grid.min_frames, tui_max_video_frames(grid, p.fps));
                p.frames = (current as i64 + delta as i64 * grid.step as i64).clamp(
                    i64::from(grid.min_frames),
                    i64::from(tui_max_video_frames(grid, p.fps)),
                ) as u32;
            }
            ParamField::Fps => {
                let grid = video_grid.expect("fps has video grid");
                if let Some(fixed_fps) = grid.fixed_fps {
                    p.fps = fixed_fps;
                    return;
                }
                p.fps = (p.fps as i32 + delta).clamp(1, 60) as u32;
                p.frames = p.frames.min(tui_max_video_frames(grid, p.fps));
            }
            ParamField::Pipeline => {
                p.pipeline = match (p.pipeline, delta >= 0) {
                    (None, true) | (Some(Ltx2PipelineMode::TwoStage), false) => {
                        Some(Ltx2PipelineMode::OneStage)
                    }
                    (Some(Ltx2PipelineMode::OneStage), true)
                    | (Some(Ltx2PipelineMode::TwoStageHq), false) => {
                        Some(Ltx2PipelineMode::TwoStage)
                    }
                    (Some(Ltx2PipelineMode::TwoStage), true)
                    | (Some(Ltx2PipelineMode::Distilled), false) => {
                        Some(Ltx2PipelineMode::TwoStageHq)
                    }
                    (Some(Ltx2PipelineMode::TwoStageHq), true) | (None, false) => {
                        Some(Ltx2PipelineMode::Distilled)
                    }
                    (Some(Ltx2PipelineMode::Distilled), true)
                    | (Some(Ltx2PipelineMode::OneStage), false) => None,
                    // These modes cannot be authored by the current TUI. If
                    // stale state reaches this boundary, recover to Auto.
                    (Some(_), _) => None,
                };
                p.format = OutputFormat::Mp4;
            }
            ParamField::Audio => {
                p.enable_audio = match (p.enable_audio, delta >= 0) {
                    (None, true) | (Some(false), false) => Some(true),
                    (Some(true), true) | (None, false) => Some(false),
                    (Some(false), true) | (Some(true), false) => None,
                };
                if p.enable_audio == Some(true) {
                    p.format = OutputFormat::Mp4;
                }
            }
            ParamField::SpatialUpscale => {
                p.spatial_upscale = match (p.spatial_upscale, delta >= 0) {
                    (None, true) | (Some(Ltx2SpatialUpscale::X2), false) => {
                        Some(Ltx2SpatialUpscale::X1_5)
                    }
                    (Some(Ltx2SpatialUpscale::X1_5), true) | (None, false) => {
                        Some(Ltx2SpatialUpscale::X2)
                    }
                    (Some(Ltx2SpatialUpscale::X2), true)
                    | (Some(Ltx2SpatialUpscale::X1_5), false) => None,
                };
            }
            ParamField::TemporalUpscale => {
                p.temporal_upscale = match p.temporal_upscale {
                    None => Some(Ltx2TemporalUpscale::X2),
                    Some(Ltx2TemporalUpscale::X2) => None,
                };
            }
            ParamField::StgScale => {
                p.guidance_overrides.stg_scale = adjust_optional_scale(
                    p.guidance_overrides.stg_scale,
                    delta,
                    0.5,
                    Ltx2GuidanceOverrides::MAX_SCALE,
                );
            }
            ParamField::RescaleScale => {
                p.guidance_overrides.rescale_scale =
                    adjust_optional_scale(p.guidance_overrides.rescale_scale, delta, 0.1, 1.0);
            }
            ParamField::ModalityScale => {
                p.guidance_overrides.modality_scale = adjust_optional_scale(
                    p.guidance_overrides.modality_scale,
                    delta,
                    0.5,
                    Ltx2GuidanceOverrides::MAX_SCALE,
                );
            }
            ParamField::GuidanceSkip => {
                p.guidance_overrides.skip_step = adjust_optional_u32(
                    p.guidance_overrides.skip_step,
                    delta,
                    Ltx2GuidanceOverrides::MAX_SKIP_STEP,
                );
            }
            ParamField::SampleShift => {
                // Wan flow shift (#782): 0 is not a valid shift, so the ramp
                // walks default → 1.0 → … → 16.0 → default. 16 is upstream's
                // own ceiling (the flf2v task ships shift 16).
                p.sample_shift = match adjust_optional_scale(p.sample_shift, delta, 1.0, 16.0) {
                    Some(value) if value < 1.0 => {
                        if delta >= 0 {
                            Some(1.0)
                        } else {
                            None
                        }
                    }
                    other => other,
                };
            }
            ParamField::ControlScale => {
                p.control_scale = (p.control_scale + delta as f64 * 0.1).clamp(0.0, 2.0);
            }
            ParamField::Format => {
                if audio_required {
                    p.format = OutputFormat::Mp4;
                    p.enable_audio = Some(true);
                } else if mesh_pinned {
                    // A mesh recipe has exactly one deliverable container.
                    // The row stays so the user can see what the print will
                    // be, but ◀▶ cannot walk it onto a raster format the
                    // server would only pin straight back to GLB.
                    p.format = OutputFormat::Glb;
                } else {
                    p.format = match p.format {
                        OutputFormat::Png => OutputFormat::Jpeg,
                        OutputFormat::Jpeg => OutputFormat::Gif,
                        OutputFormat::Gif => OutputFormat::Apng,
                        OutputFormat::Apng => OutputFormat::Webp,
                        OutputFormat::Webp => OutputFormat::Mp4,
                        OutputFormat::Mp4 => OutputFormat::Png,
                        // `wav` is not in the cycle: it is only valid for the
                        // LTX-2 text-to-audio pipeline, which the Create form has
                        // no control for. Cycling exits back to the raster start
                        // rather than offering a format that would 422.
                        OutputFormat::Wav => OutputFormat::Png,
                        // Nor are the mesh containers. A 3-D family emits GLB
                        // and nothing else, so the control is not a choice
                        // there; and offering GLB on a raster model would 422
                        // for the same reason `wav` would.
                        OutputFormat::Glb | OutputFormat::Obj => OutputFormat::Png,
                    };
                    if p.enable_audio == Some(true) && p.format != OutputFormat::Mp4 {
                        p.enable_audio = None;
                    }
                }
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
            | ParamField::StgBlocks
            | ParamField::SourceImage
            | ParamField::IdentityImage
            | ParamField::References
            | ParamField::MaskImage
            | ParamField::ControlImage
            | ParamField::ControlModel
            // File under is edited in its own popup; the adjust affordance
            // has nothing to cycle through.
            | ParamField::Title
            | ParamField::Tags
            | ParamField::Collection => {}
        }
        if field == ParamField::Pipeline {
            self.sync_pipeline_guidance();
            if let Some(recipe) = self.active_generation_recipe() {
                self.generate.params.width = recipe.defaults.width;
                self.generate.params.height = recipe.defaults.height;
                self.generate.params.steps = recipe.defaults.steps;
                self.generate.params.guidance = recipe.defaults.guidance;
                if let Some(frames) = recipe.defaults.frames {
                    self.generate.params.frames = frames;
                }
                if let Some(fps) = recipe.defaults.fps {
                    self.generate.params.fps = fps;
                }
            }
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

                // A mesh is never handed to `image::open`: there is no
                // raster inside a `.glb`, and downloading it only to fail
                // the decode would leave the pane blank after a multi-
                // megabyte round trip. The poster the server rendered at
                // save time IS the preview; it is served by the thumbnail
                // route and cached under the same key the grid uses.
                if crate::gallery_scan::is_mesh_filename(&filename) {
                    let poster = crate::thumbnails::thumbnail_path(&entry.path);
                    if let Ok(img) = image::open(&poster) {
                        let protocol = self.picker.new_resize_protocol(img.clone());
                        self.gallery.preview_image = Some(img);
                        self.gallery.image_state = Some(protocol);
                        self.gallery.animation = None;
                        return;
                    }
                    let tx = self.bg_tx.clone();
                    let fetch_url = url.clone();
                    let fetch_name = filename.clone();
                    self.tokio_handle.spawn(async move {
                        if let Some(data) = crate::gallery_scan::fetch_and_cache_mesh_poster(
                            &fetch_url,
                            &host_id,
                            &fetch_name,
                        )
                        .await
                        {
                            let _ = tx.send(BackgroundEvent::GalleryPreviewReady(data));
                        }
                    });
                    self.gallery.preview_image = None;
                    self.gallery.image_state = None;
                    self.gallery.animation = None;
                    return;
                }

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
                // A local mesh: only its cached poster is a picture. Without
                // one the pane stays empty rather than feeding glTF bytes to
                // a raster decoder.
                if crate::gallery_scan::is_mesh_filename(&entry.filename()) {
                    let poster = crate::thumbnails::thumbnail_path(&entry.path);
                    if let Ok(img) = image::open(&poster) {
                        let protocol = self.picker.new_resize_protocol(img.clone());
                        self.gallery.preview_image = Some(img);
                        self.gallery.image_state = Some(protocol);
                        self.gallery.animation = None;
                        return;
                    }
                    self.gallery.preview_image = None;
                    self.gallery.image_state = None;
                    self.gallery.animation = None;
                    return;
                }
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

        // Populate prompt fields. The model switches first so the Negative
        // restore below can resolve the restored model's advertised default:
        // metadata without a `negative_prompt` predates truthful recording,
        // and for wan that means the tuned default conditioned the render —
        // restoring an empty editor would flip the reuse into an explicit
        // empty-uncond opt-out (#787).
        self.generate.prompt = tui_textarea::TextArea::from(meta.prompt.lines());
        self.generate.params.original_prompt = if meta.prompt.trim().is_empty() {
            None
        } else {
            meta.original_prompt.clone()
        };
        self.generate.params.prompt_transform = if meta.prompt.trim().is_empty() {
            None
        } else {
            meta.prompt_transform.clone()
        };
        self.generate.params.quick_transform_snapshot = None;
        self.generate.params.prepared_prompts.clear();
        self.generate.params.prepared_prompt_transforms.clear();
        self.generate.params.prepared_transform_snapshot = None;
        self.update_model(&meta.model);
        let restored_negative = meta
            .negative_prompt
            .as_deref()
            .unwrap_or(&self.generate.negative_default)
            .to_string();
        self.generate.negative_prompt = negative_prompt_textarea(&restored_negative);
        // #787 round 3: a recorded `""` is the explicit opt-out. Keep that
        // authority live so a later default reconciliation (fresher catalog
        // row) does not mistake the empty editor for "untouched", and the
        // wire keeps shipping `""`.
        self.generate.negative_explicit_clear = meta
            .negative_prompt
            .as_deref()
            .is_some_and(|saved| saved.trim().is_empty());
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
        // The mesh rows restore the recipe's defaults: the server records no
        // octree / iso-level / face-target provenance in `OutputMetadata`
        // yet, so there is nothing truthful to restore, and carrying the
        // previous form's values into a reuse would attribute them to a
        // print they never shaped.
        self.generate.params.mesh = mold_core::MeshRequestOptions::default();
        if let Some(ref lora) = meta.lora {
            self.generate.params.lora_path = Some(lora.clone());
            self.generate.params.lora_scale = meta.lora_scale.unwrap_or(1.0);
        } else {
            self.generate.params.lora_path = None;
        }
        self.sync_generate_capabilities();

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
        self.gallery.thumbnail_loading.clear();
        self.gallery.thumbnail_lru.clear();
        self.gallery.details_thumbnail_state = None;
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

    /// Capabilities the TUI has polled for the machine behind `origin`.
    /// The connected `--host` server is polled under `LOCAL_HOST_ID`
    /// even when it is a different box, so an origin synthesized from
    /// that URL falls back to that slot.
    pub(crate) fn capabilities_for_origin(
        &self,
        origin: &GalleryOrigin,
    ) -> Option<&mold_core::ServerCapabilities> {
        if let Some(caps) = self.machines.capabilities.get(&origin.host_id) {
            return Some(caps);
        }
        if origin.url.is_some() && origin.url == self.server_url {
            return self.machines.capabilities.get(crate::hosts::LOCAL_HOST_ID);
        }
        None
    }

    /// Whether Library may offer its immediate upscale action for `entry`.
    /// Images retain the local/legacy fallback. Videos require the durable
    /// server pipeline and its explicit codec-backed capability.
    pub(crate) fn can_upscale_entry(&self, entry: &GalleryEntry) -> bool {
        if !crate::gallery_scan::is_video_filename(&entry.filename()) {
            return true;
        }
        let origin = entry.primary_origin();
        (origin.url.is_some() || self.server_url.is_some())
            && self
                .capabilities_for_origin(&origin)
                .is_some_and(|caps| caps.video_upscale.available)
    }

    /// Whether a print held by `origin` is moved to a trash (recoverable)
    /// or deleted outright when the TUI removes it there. A server whose
    /// capabilities have not been read yet is **not** assumed to trash.
    pub(crate) fn origin_can_trash(&self, origin: &GalleryOrigin) -> bool {
        match origin.url {
            None => self.gallery.local_trash_available,
            Some(_) => self
                .capabilities_for_origin(origin)
                .and_then(|caps| caps.gallery.trash.as_ref())
                .is_some_and(|trash| trash.enabled),
        }
    }

    /// What `d` does for `entry`: trash only when every owning machine
    /// can trash, else the honest hard-delete wording.
    pub(crate) fn removal_kind_for(&self, entry: &GalleryEntry) -> RemovalKind {
        if entry
            .owning_origins()
            .iter()
            .all(|origin| self.origin_can_trash(origin))
        {
            RemovalKind::Trash
        } else {
            RemovalKind::Delete
        }
    }

    /// Removal verb for the selected print's key hints.
    pub(crate) fn selected_removal_kind(&self) -> RemovalKind {
        self.gallery
            .entries
            .get(self.gallery.selected)
            .map(|entry| self.removal_kind_for(entry))
            .unwrap_or(RemovalKind::Delete)
    }

    /// Remove the currently selected gallery print on every machine that
    /// holds it: the trash where available (local DB-backed `.trash/`
    /// move, server `DELETE` which current servers treat as trash), a
    /// hard delete otherwise — plus the TUI's own thumbnail/preview cache
    /// entries either way.
    /// Library `x`: offer the export containers for the selected 3-D print.
    ///
    /// A remote print offers what its OWNING host advertises
    /// (`capabilities.mesh.export_formats`, the stored GLB excluded); a local
    /// print offers every transcode the in-process writer has. Anything but
    /// a `.glb` is refused by name rather than silently ignored.
    fn open_mesh_export_picker(&mut self) {
        let Some(entry) = self.gallery.entries.get(self.gallery.selected) else {
            return;
        };
        let filename = entry.filename();
        if !crate::gallery_scan::is_mesh_filename(&filename) {
            self.generate.error_message =
                Some("Export converts 3-D prints (.glb) only; press o to open this file".into());
            return;
        }
        let origin = entry.primary_origin();
        let advertised = if origin.url.is_some() {
            self.capabilities_for_origin(&origin)
                .and_then(|caps| caps.mesh.as_ref())
                .map(|mesh| mesh.export_formats.as_slice())
        } else {
            None
        };
        let formats = mesh_export_formats_for(advertised);
        if formats.is_empty() {
            self.generate.error_message =
                Some("This machine advertises no mesh export formats".into());
            return;
        }
        self.popup = Some(Popup::MeshExportPicker {
            filename,
            formats,
            selected: 0,
        });
    }

    /// Transcode the selected `.glb` into `format` and write it beside the
    /// TUI's other saves. Remote prints are converted by their owning host
    /// (`POST /api/gallery/export/:filename`, with that host's API key);
    /// local prints by the same writer the server uses. The gallery file is
    /// never renamed or replaced.
    fn spawn_mesh_export(&mut self, filename: &str, format: mold_core::MeshExportFormat) {
        let Some(entry) = self
            .gallery
            .entries
            .iter()
            .find(|entry| entry.filename() == filename)
            .cloned()
        else {
            return;
        };
        if self.config.is_output_disabled() {
            self.generate.error_message =
                Some("Export needs an output directory; output is disabled".into());
            return;
        }
        let target = mesh_export_target_path(&self.config.effective_output_dir(), filename, format);
        let origin = entry.primary_origin();
        let local_path = entry.path.clone();
        let filename = filename.to_string();
        let tx = self.bg_tx.clone();
        self.tokio_handle.spawn(async move {
            let bytes: Result<Vec<u8>, String> = match origin.url {
                Some(url) => {
                    let api_key = crate::hosts::api_key_for(&origin.host_id);
                    let client = crate::hosts::client_for(&url, api_key.as_deref());
                    client
                        .export_gallery_mesh(&filename, format)
                        .await
                        .map_err(|e| e.to_string())
                }
                None => tokio::task::spawn_blocking(move || {
                    let glb = std::fs::read(&local_path).map_err(|e| e.to_string())?;
                    export_local_mesh(&glb, format)
                })
                .await
                .unwrap_or_else(|e| Err(e.to_string())),
            };
            let outcome = match bytes {
                Ok(bytes) => {
                    if let Some(parent) = target.parent() {
                        let _ = tokio::fs::create_dir_all(parent).await;
                    }
                    tokio::fs::write(&target, &bytes)
                        .await
                        .map(|_| target)
                        .map_err(|e| e.to_string())
                }
                Err(e) => Err(e),
            };
            let _ = tx.send(match outcome {
                Ok(path) => BackgroundEvent::MeshExportComplete(path),
                Err(message) => BackgroundEvent::MeshExportFailed(message),
            });
        });
    }

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

        // Local disk copy first, synchronously: a failed trash move keeps
        // the tile (and the file) exactly where it was and says why,
        // instead of optimistically dropping a print that still exists.
        if self.gallery.local_trash_available && entry.path.is_file() {
            let result = match mold_db::open_default() {
                Ok(Some(db)) => crate::gallery_trash::trash_local_print(
                    &db,
                    &entry.path,
                    mold_core::time::now_epoch_ms(),
                ),
                Ok(None) => Err(anyhow::anyhow!("the metadata DB is disabled")),
                Err(e) => Err(e),
            };
            if let Err(e) = result {
                self.generate.error_message = Some(format!("Move to trash failed: {e:#}"));
                return;
            }
        }

        let entry = &self.gallery.entries[idx];
        let thumb_path = crate::thumbnails::thumbnail_path(&entry.path);
        let _ = std::fs::remove_file(&thumb_path);
        // Also remove the legacy (pre-host-scoping) cached copy.
        let cache_path = crate::gallery_scan::image_cache_dir().join(&filename);
        let _ = std::fs::remove_file(&cache_path);

        // Remove on every machine the print exists on. Remote removals go
        // through the OWNING host's client (with its saved API key) and
        // propagate errors through the background channel so the UI can
        // surface them and rescan — a silent fire-and-forget would mask
        // transient network errors and leave the removed tile "gone"
        // locally while a server still holds the file. The request is
        // the same `DELETE /api/gallery/image/:name` on every server:
        // current servers move the print to their trash, older ones
        // delete it — which is why the confirm copy only promises the
        // trash when every owning host advertises `gallery.trash`.
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
                if let Err(e) = client.trash_gallery_image(&filename).await {
                    let _ = tx.send(BackgroundEvent::GalleryDeleteFailed(e.to_string()));
                }
            });
        }
        // Without a local trash, remove the local file outright (covers
        // both local and server-backed entries where the TUI also saved a
        // copy during generation). With one, the move above already
        // emptied this path.
        if !self.gallery.local_trash_available && entry.path.is_file() {
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
    fn refresh_selected_host_queue(&mut self) {
        if let Some(host_id) = self.machines.selected_host().map(|entry| entry.id.clone()) {
            self.machines.request_queue_refresh(&host_id);
            self.tick_host_polling();
        }
    }

    fn open_model_selector(&mut self) {
        let mut models: Vec<String> = generation_model_names(&self.models.catalog, "");
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

    /// Surface the advisories a completed request carried
    /// (`x-mold-request-warning`, via `request_warnings`).
    ///
    /// These ride a request the host ACCEPTED and rendered — a filing it
    /// could not apply, a lip-dub clip it retimed — so they take the
    /// advisory slot (`!`, warning styling) and never `error_message`, whose
    /// `✗` would read as a failed render.
    ///
    /// Each advisory is taken WHOLE. The prose carries "; " as ordinary
    /// punctuation ("…were not applied; the print was generated and saved
    /// normally"), so splitting on it yields two dangling half-sentences.
    /// Several are joined with the separator the TUI already uses between
    /// independent facts, which the prose cannot contain. They also land in
    /// the Timeline, the per-generation record, which holds what the
    /// one-line slot clips.
    ///
    /// One-shot and chain completions share this so the two can never drift.
    fn surface_request_advisories(&mut self, advisories: &[String]) {
        if advisories.is_empty() {
            return;
        }
        for advisory in advisories {
            self.generate.progress.push_log(ProgressLogEntry {
                message: advisory.clone(),
                style: ProgressStyle::Warning,
            });
        }
        self.generate.warning_message = Some(advisories.join(" \u{00b7} "));
    }

    /// Prefill text for one File-under editor: the stored value in the
    /// shape the editor speaks.
    fn filing_editor_text(&self, field: ParamField) -> String {
        match field {
            ParamField::Title => self.generate.params.title.clone().unwrap_or_default(),
            ParamField::Tags => {
                crate::ui::create_form::format_tag_input(&self.generate.params.tags)
            }
            ParamField::Collection => self.generate.params.collection.clone().unwrap_or_default(),
            _ => String::new(),
        }
    }

    /// Store one validated File-under edit.
    fn apply_filing_edit(&mut self, edit: crate::ui::create_form::FilingEdit) {
        use crate::ui::create_form::FilingEdit;
        match edit {
            FilingEdit::Title(title) => self.generate.params.title = title,
            FilingEdit::Tags(tags) => self.generate.params.tags = tags,
            FilingEdit::Collection(collection) => self.generate.params.collection = collection,
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
            ParamField::Audio => self.adjust_field(ParamField::Audio, 1),
            ParamField::Pipeline => self.adjust_field(ParamField::Pipeline, 1),
            ParamField::SpatialUpscale => self.adjust_field(ParamField::SpatialUpscale, 1),
            ParamField::TemporalUpscale => self.adjust_field(ParamField::TemporalUpscale, 1),
            ParamField::StgBlocks => {
                let input = self
                    .generate
                    .params
                    .guidance_overrides
                    .stg_blocks
                    .as_ref()
                    .map(|blocks| {
                        blocks
                            .iter()
                            .map(u32::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                self.popup = Some(Popup::StgBlocksInput { input, error: None });
            }
            ParamField::References => {
                let input = crate::h3_references::format_reference_input(
                    &self.generate.params.reference_paths,
                );
                self.popup = Some(Popup::ReferencesInput { input, error: None });
            }
            ParamField::IdentityImage => {
                let input = self
                    .generate
                    .params
                    .identity_image_path
                    .clone()
                    .unwrap_or_default();
                self.popup = Some(Popup::IdentityImageInput {
                    input,
                    error: self.generate.identity_error.clone(),
                });
            }
            // File under — three validated one-line editors.
            ParamField::Title | ParamField::Tags | ParamField::Collection => {
                let input = self.filing_editor_text(field);
                self.popup = Some(Popup::FilingInput {
                    field,
                    input,
                    error: None,
                });
            }
            // Cycle format
            ParamField::Format => self.adjust_field(ParamField::Format, 1),
            // Cycle scheduler
            ParamField::Scheduler => {
                self.generate.params.scheduler = match self.generate.params.scheduler {
                    None => Some(Scheduler::Ddim),
                    Some(Scheduler::Ddim) => Some(Scheduler::EulerAncestral),
                    Some(Scheduler::EulerAncestral) => Some(Scheduler::UniPc),
                    Some(Scheduler::UniPc) => None,
                    // Wan flow solvers are not part of the SD scheduler row's
                    // cycle; model-specific EDM and stale Wan values reset to
                    // the manifest default.
                    Some(Scheduler::EdmDpmPp2m)
                    | Some(Scheduler::Euler)
                    | Some(Scheduler::DpmPp) => None,
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
        let mut default_frames = 25;
        let mut default_fps = 24;

        // Use server catalog defaults when connected (and not in local mode),
        // local config otherwise
        if let Some(entry) = if self.should_poll_remote() {
            self.models.catalog.iter().find(|m| m.name == model)
        } else {
            None
        } {
            self.generate.params.width = entry.defaults.default_width;
            self.generate.params.height = entry.defaults.default_height;
            if let Some(frames) = entry.defaults.default_frames {
                default_frames = frames;
            }
            if let Some(fps) = entry.defaults.default_fps {
                default_fps = fps;
            }
            self.generate.params.steps = entry.defaults.default_steps;
            self.generate.params.guidance = entry.defaults.default_guidance;
        } else {
            let mc = self.config.resolved_model_config(&model);
            self.generate.params.width = mc.effective_width(&self.config);
            self.generate.params.height = mc.effective_height(&self.config);
            if let Some(frames) = mc.effective_frames() {
                default_frames = frames;
            }
            if let Some(fps) = mc.effective_fps() {
                default_fps = fps;
            }
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
        self.generate.params.frames = default_frames;
        self.generate.params.fps = default_fps;
        self.generate.params.pipeline = None;
        self.generate.params.enable_audio = None;
        self.generate.params.spatial_upscale = None;
        self.generate.params.temporal_upscale = None;
        self.generate.params.guidance_overrides = Ltx2GuidanceOverrides::default();
        self.generate.params.strength = 0.75;
        self.generate.params.source_image_path = None;
        self.generate.params.reference_paths.clear();
        // Reset is the one control that always clears the identity photo —
        // including on a model that cannot take it, where it is the way back
        // out of the gate refusal.
        self.generate.params.identity_image_path = None;
        self.generate.params.id_weight = mold_core::identity::ID_WEIGHT_DEFAULT;
        self.generate.params.id_start_step = mold_core::identity::ID_START_STEP_DEFAULT;
        self.generate.identity_error = None;
        self.generate.params.mask_image_path = None;
        self.generate.params.control_image_path = None;
        self.generate.params.control_model = None;
        self.generate.params.control_scale = 1.0;
        // Reset Defaults is the form's explicit "start over" (it already
        // drops the source image, which is no more a model default than a
        // title is), so the creation-time filing goes with it.
        self.generate.params.title = None;
        self.generate.params.tags.clear();
        self.generate.params.collection = None;
        // Back to the recipe's own octree, iso-level and raw surface. The
        // GLB pin is re-applied by the family normalizer inside the sync
        // below, so the `Png` reset above never survives on a mesh recipe.
        self.generate.params.mesh = mold_core::MeshRequestOptions::default();
        self.sync_generate_capabilities();
        // #787 round 2: Reset Defaults is an explicit "give me the model's
        // defaults", not a model switch — the sync above deliberately
        // preserves a cleared or typed editor, but here both go back to the
        // advertised default (empty when the model has none), mirroring web
        // Reset settings and desktop ⌘N. That also resolves any deferred
        // restore-time clear marker (#787 round 3).
        self.generate.negative_prompt = negative_prompt_textarea(&self.generate.negative_default);
        self.generate.negative_explicit_clear = false;
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

        // ── Library ─────────────────────────────────────────────
        // How long a print moved to the trash survives before the
        // retention sweeper purges it; 0 keeps trashed prints forever.
        rows.push(SettingsRow::SectionHeader {
            name: "Library".into(),
        });
        rows.push(SettingsRow::Field {
            key: SettingsKey::GalleryTrashRetentionDays,
            label: "Trash (days)",
            field_type: SettingsFieldType::Number {
                min: 0.0,
                max: f64::from(mold_core::config::GALLERY_TRASH_RETENTION_MAX_DAYS),
                step: 1.0,
            },
        });
        // Whether a titled print picks up its own slug as a tag. Mirrors web
        // Settings > Library "Tag new prints with their title"; this is a
        // client decision, so the Create form's File under section reads it
        // and discloses the tag before Generate.
        rows.push(SettingsRow::Field {
            key: SettingsKey::GenerateAutoTagTitle,
            label: "Tag by title",
            field_type: SettingsFieldType::Bool,
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
                    options: vec![
                        "(none)",
                        "ddim",
                        "euler-ancestral",
                        "uni-pc",
                        "edm-dpm-pp-2m",
                    ],
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
            // Library
            SettingsKey::GalleryTrashRetentionDays => {
                trash_retention_display(cfg.gallery.trash_retention_days)
            }
            SettingsKey::GenerateAutoTagTitle => if cfg.generate.auto_tag_title {
                "on"
            } else {
                "off"
            }
            .into(),
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

    /// Persist a DB-surface config key (`expand.*`, `gallery.*`, …) after
    /// `self.config` has been mutated, through the same
    /// `mold_db::config_sync::persist_config_key` path the CLI and server
    /// use. A disabled/unavailable DB is a silent no-op (the TOML save
    /// that follows still records the value).
    fn persist_db_surface_key(&mut self, key: &str) {
        let db = match mold_db::open_default() {
            Ok(Some(db)) => db,
            _ => return,
        };
        if let Err(e) = mold_db::config_sync::persist_config_key(&db, &self.config, key) {
            self.settings.save_error = Some(format!("Save failed: {e:#}"));
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
            SettingsKey::GalleryTrashRetentionDays => {
                mold_core::config::GallerySettings::TRASH_RETENTION_DAYS_ENV
            }
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
            SettingsKey::GalleryTrashRetentionDays => {
                cfg.gallery.trash_retention_days =
                    (f64::from(cfg.gallery.trash_retention_days) + delta).clamp(min, max) as u32;
                // `gallery.*` is a DB-surface key: persist it through the
                // shared config_sync writer (what `mold config set` and
                // the server's `PUT /api/config/:key` use) as well as the
                // TOML, so the server's sweeper and every other surface
                // read the same value.
                self.persist_db_surface_key(
                    mold_core::config_keys::GALLERY_TRASH_RETENTION_DAYS_KEY,
                );
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
                            "edm-dpm-pp-2m" => Some(Scheduler::EdmDpmPp2m),
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
            SettingsKey::GenerateAutoTagTitle => {
                self.config.generate.auto_tag_title = !self.config.generate.auto_tag_title;
                // `generate.*` is a DB-surface key: persist it through the
                // shared config_sync writer so `mold run`, the server, and
                // every other surface read the same preference.
                self.persist_db_surface_key(mold_core::config_keys::GENERATE_AUTO_TAG_TITLE_KEY);
                // The Create form holds a snapshot so its summary and its
                // request can never disagree; refresh it now.
                self.generate.params.auto_tag_title = self.config.generate.auto_tag_title;
            }
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
        // Persistence, gallery reuse, and future call sites may all mutate the
        // shared parameter bag without touching the visible rows. Reassert the
        // selected family's request authority immediately before freezing it.
        self.sync_generate_capabilities();
        let prompt_text = self.generate.prompt.lines().join("\n").trim().to_string();
        if prompt_text.is_empty() && self.prompt_required_now() {
            self.generate.error_message = Some("Prompt is empty".to_string());
            return;
        }
        let h3_task = mold_core::minimax_h3::task_for_model(&self.generate.params.model);
        if h3_task == Some(mold_core::minimax_h3::Task::Ref2va) {
            if self.generate.params.reference_paths.is_empty() {
                self.generate.error_message = Some(
                    "MiniMax H3 Ref2VA requires at least one ordered image or video reference"
                        .to_string(),
                );
                return;
            }
            if self.generate.params.batch != 1 || !self.generate.params.prepared_prompts.is_empty()
            {
                self.generate.error_message = Some(
                    "MiniMax H3 ordered references require Batch 1; submit each attempt separately"
                        .to_string(),
                );
                return;
            }
        } else if !self.generate.params.reference_paths.is_empty() {
            self.generate.error_message = Some(
                "Ordered references require an explicitly authorized MiniMax H3 Ref2VA model"
                    .to_string(),
            );
            return;
        }

        // A required source image is the one contract the Create form cannot
        // express by hiding a row: the row is visible and simply unset, so
        // dispatch has to be the gate.
        if let Some(message) = crate::model_info::source_image_contract_error(
            self.source_image_contract(&self.generate.params.model),
            self.generate.params.source_image_path.is_some(),
        ) {
            self.generate.error_message = Some(message.to_string());
            return;
        }

        // An identity photo the current model cannot take — or one whose file
        // has gone away since it was picked — blocks dispatch rather than
        // rendering a print that silently has the wrong face in it.
        if let Some(message) = self.identity_dispatch_error() {
            self.generate.identity_error = Some(message.clone());
            self.generate.error_message = Some(message);
            return;
        }

        // Both File-under editors already refuse anything admission would,
        // so this only catches the state they cannot see: turning
        // `generate.auto_tag_title` back on behind a form already at the
        // tag cap. Refusing here beats a 422 after the queue accepts it.
        if let Err(message) = crate::ui::create_form::compose_filing_tags(
            &self.generate.params.tags,
            self.generate.params.title.as_deref(),
            self.generate.params.auto_tag_title,
        ) {
            self.generate.error_message = Some(message);
            return;
        }

        let generation_target = if self.generate.params.prepared_prompts.is_empty() {
            if let Some(snapshot) = self.generate.params.quick_transform_snapshot.as_ref() {
                if let Some(reason) = self.quick_transform_staleness(snapshot) {
                    self.generate.error_message = Some(format!(
                        "Applied Remix is stale because the {reason}; remix again or restore the source prompt"
                    ));
                    return;
                }
            }
            self.target.clone()
        } else {
            let Some(snapshot) = self.generate.params.prepared_transform_snapshot.clone() else {
                self.generate.error_message =
                    Some("Prepared Remix lost its frozen route; remix again".into());
                return;
            };
            let current_reference_fingerprint = if self.generate.params.model == snapshot.model
                && self.target == snapshot.target
                && self.prompt_transform_task() == snapshot.task
            {
                match self.reference_fingerprint_for_target(&self.target) {
                    Ok(fingerprint) => Some(fingerprint),
                    Err(error) => {
                        self.generate.error_message = Some(error);
                        return;
                    }
                }
            } else {
                None
            };
            let stale_reason = if self.generate.params.model != snapshot.model {
                Some("model changed")
            } else if self.target != snapshot.target {
                Some("generation target changed")
            } else if self.prompt_transform_task() != snapshot.task {
                Some("conditioning task changed")
            } else if current_reference_fingerprint.as_deref()
                != Some(snapshot.reference_fingerprint.as_str())
            {
                Some("ordered references changed")
            } else if self.generate.params.prepared_prompts.len()
                != self.generate.params.prepared_prompt_transforms.len()
                || self
                    .generate
                    .params
                    .prepared_prompt_transforms
                    .iter()
                    .any(|provenance| {
                        provenance.operation != snapshot.operation
                            || provenance.source_prompt != snapshot.source_prompt
                            || provenance.root_prompt != snapshot.root_prompt
                            || provenance.source_kind != snapshot.source_kind
                            || provenance.task != snapshot.task
                    })
            {
                Some("reviewed variation provenance changed")
            } else {
                None
            };
            if let Some(reason) = stale_reason {
                self.generate.error_message = Some(format!(
                    "Prepared Remix is stale because the {reason}; remix again"
                ));
                return;
            }
            snapshot.target
        };

        // Route by the frozen prepared target or the current sticky Machines target. Auto keeps
        // today's exact behavior; Local forces the in-process engine; a
        // Host target pins the run to that registry entry with its API
        // key and no silent fallback. Routing runs before any state
        // mutation so a stale target aborts cleanly.
        use crate::hosts::GenTarget;
        let mut route_mode = None;
        let mut route_host = None;
        let mut route_name = None;
        let mut api_key = None;
        match &generation_target {
            GenTarget::Auto => {
                api_key = std::env::var("MOLD_API_KEY")
                    .ok()
                    .filter(|key| !key.is_empty());
            }
            GenTarget::Local if h3_task.is_some() => {
                self.generate.error_message = Some(format!(
                    "MiniMax H3 has no in-process TUI runtime. Select an authorized mold server. {}",
                    mold_core::MINIMAX_H3_AUTHORIZATION_REQUIRED
                ));
                return;
            }
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
        // A new submission owns the pane; held children of the previous batch
        // are no longer this session's work to retry.
        self.generate.held_batch = None;
        self.generate.batch_remaining = self.generate.params.batch;
        self.generate.error_message = None;
        // An advisory describes the print that produced it.
        self.generate.warning_message = None;
        self.generate.progress.clear();
        self.generate.progress.mark_generation_start();
        self.generate.clear_live_preview();
        self.generate.preview_image = None;
        self.generate.image_state = None;
        self.generate.animation = None;
        self.generate.last_mesh_summary = None;

        // #787 tri-state: editor text equal to the advertised default stays
        // absent on the wire (the server/engine re-applies it; older servers
        // behave identically), a cleared editor ships the explicit `""`
        // opt-out when a default is advertised, and typed text replaces it.
        let negative_prompt = crate::ui::create_form::negative_prompt_wire_value(
            &self.generate.negative_prompt.lines().join("\n"),
            &self.generate.negative_default,
            self.generate.capabilities.supports_negative_prompt,
            self.generate.negative_explicit_clear,
        );

        // Resolve seed based on seed mode
        let resolved_seed = self
            .generate
            .params
            .seed_mode
            .resolve(self.generate.params.seed);
        let mut params = self.generate.params.clone();
        self.normalize_fixed_guidance_for_submit(&mut params);
        params.seed = Some(resolved_seed);
        // Nothing to expand when the conditioning carries the shot — expanding
        // "" would let the model invent the prompt. Mirrors the server's
        // `maybe_expand_prompt` guard for the remote path.
        if prompt_text.is_empty() {
            params.expand = false;
        }
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

    /// Retry every held child of the batch this client last submitted.
    ///
    /// The retry fence is the whole admission authority, which only the
    /// submitting client holds — a Machines queue row carries no batch
    /// identity — so this is where the action lives, and it is offered only
    /// for the work this session still owns.
    fn retry_held_prints(&mut self) {
        if self.generate.generating {
            return;
        }
        // The held batch carries its own host and submission; the form and
        // the Machines target may have moved on, and neither can retry a row
        // another instance admitted or should record a prompt that never
        // rendered. It is NOT cleared here: a failed retry leaves the children
        // held on the host, and the settling report replaces it.
        let Some(held) = self.generate.held_batch.clone() else {
            self.generate.error_message =
                Some("No held prints from this session to retry".to_string());
            return;
        };
        self.generate.generating = true;
        self.generate.error_message = None;
        self.generate.progress.mark_generation_start();
        let tx = self.bg_tx.clone();
        self.tokio_handle.spawn(async move {
            crate::backend::retry_held_prints(held, tx).await;
        });
    }

    fn set_prompt_text(&mut self, prompt: &str) {
        self.generate.prompt = TextArea::new(prompt.lines().map(String::from).collect());
        self.generate
            .prompt
            .set_cursor_line_style(ratatui::style::Style::default());
        self.generate.focus = GenerateFocus::Prompt;
    }

    fn set_authored_prompt_text(&mut self, prompt: &str) {
        self.set_prompt_text(prompt);
        self.retire_prompt_provenance();
    }

    fn retire_prompt_provenance(&mut self) {
        self.generate.params.original_prompt = None;
        self.generate.params.prompt_transform = None;
        self.generate.params.quick_transform_snapshot = None;
        self.generate.params.prepared_prompts.clear();
        self.generate.params.prepared_prompt_transforms.clear();
        self.generate.params.prepared_transform_snapshot = None;
    }

    /// Generation facts the expander renders after the family guide.
    fn prompt_transform_context(&self, family: &str) -> mold_core::ExpandContext {
        let params = &self.generate.params;
        let mut references = Vec::new();
        for reference in &params.reference_paths {
            references.push(mold_core::ExpandReference {
                kind: match reference.kind {
                    crate::h3_references::ReferenceKind::Image => {
                        mold_core::GenerationReferenceKind::Image
                    }
                    crate::h3_references::ReferenceKind::Video => {
                        mold_core::GenerationReferenceKind::Video
                    }
                    crate::h3_references::ReferenceKind::Audio => {
                        mold_core::GenerationReferenceKind::Audio
                    }
                },
                has_audio: reference.kind == crate::h3_references::ReferenceKind::Video
                    && crate::h3_references::video_has_audio(&reference.path),
                role: Some(mold_core::ExpandReferenceRole::Reference),
            });
        }
        if params.source_image_path.is_some() {
            let role = if mold_core::ExpandTask::for_family(family)
                != mold_core::ExpandTask::TextToImage
            {
                mold_core::ExpandReferenceRole::FirstFrame
            } else if family == "qwen-image-edit" {
                mold_core::ExpandReferenceRole::Edit
            } else {
                mold_core::ExpandReferenceRole::Source
            };
            references.push(mold_core::ExpandReference::image(role));
        }
        if params.identity_image_path.is_some() {
            references.push(mold_core::ExpandReference::image(
                mold_core::ExpandReferenceRole::Identity,
            ));
        }
        let video = mold_core::ExpandTask::for_family(family) != mold_core::ExpandTask::TextToImage;
        mold_core::ExpandContext {
            model: Some(params.model.clone()),
            width: (params.width > 0).then_some(params.width),
            height: (params.height > 0).then_some(params.height),
            frames: (video && params.frames > 0).then_some(params.frames),
            fps: (video && params.fps > 0).then_some(params.fps),
            clip_frames: None,
            negative_prompt_supported: None,
            audio: params.enable_audio,
            references,
            loras: params
                .lora_path
                .as_deref()
                .and_then(|path| {
                    std::path::Path::new(path)
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .map(str::to_string)
                })
                .into_iter()
                .collect(),
        }
    }

    fn prompt_transform_task(&self) -> mold_core::ExpandTask {
        if !self.generate.params.reference_paths.is_empty() {
            return mold_core::ExpandTask::ReferenceToAudioVideo;
        }
        let family = family_for_model(&self.generate.params.model, &self.config);
        mold_core::ExpandTask::for_conditioning(
            &family,
            None,
            self.generate.params.source_image_path.is_some(),
            false,
            false,
            0,
            false,
            (!(self.generate.params.predict_duration
                && self.generate.params.duration_prediction_supported))
                .then_some(self.generate.params.frames),
        )
    }

    /// Hash ordered reference content only after the selected route proves it
    /// has a syntactically valid API-key header. This keeps transform
    /// staleness exact without letting UI snapshotting read media before the
    /// same authentication precondition used by upload binding.
    fn reference_fingerprint_for_target(
        &self,
        target: &crate::hosts::GenTarget,
    ) -> std::result::Result<String, String> {
        let references = &self.generate.params.reference_paths;
        if references.is_empty() {
            return crate::h3_references::authority_fingerprint(None, references)
                .map_err(|error| error.to_string());
        }

        use crate::hosts::GenTarget;
        let (url, api_key) = match target {
            GenTarget::Host(id) => {
                let entry = self.machines.registry.get(id).ok_or_else(|| {
                    format!("Machine '{id}' is no longer saved. Pick a target in Machines (4).")
                })?;
                (entry.url.clone(), crate::hosts::api_key_for(id))
            }
            GenTarget::Auto => {
                let url = self.server_url.clone().ok_or_else(|| {
                    "MiniMax H3 references require an authorized remote Mold target".to_string()
                })?;
                let api_key = std::env::var("MOLD_API_KEY")
                    .ok()
                    .filter(|key| !key.is_empty());
                (url, api_key)
            }
            GenTarget::Local => {
                return Err(
                    "MiniMax H3 references require an authorized remote Mold target".to_string(),
                );
            }
        };
        let client = crate::hosts::client_for(&url, api_key.as_deref());
        crate::h3_references::authority_fingerprint(Some(&client), references)
            .map_err(|error| error.to_string())
    }

    fn start_prompt_transform(&mut self, operation: PromptTransformOperation) {
        if !self.generate.params.reference_paths.is_empty()
            && operation == PromptTransformOperation::Remix
        {
            self.generate.error_message = Some(
                "MiniMax H3 ordered references are Batch 1 only; use Expand for one reviewed prompt"
                    .to_string(),
            );
            return;
        }
        let current = self.generate.prompt.lines().join("\n").trim().to_string();
        if current.is_empty() {
            self.generate.error_message = Some("Prompt is empty".to_string());
            return;
        }
        let root = self
            .generate
            .params
            .original_prompt
            .clone()
            .filter(|root| !root.trim().is_empty());
        if operation == PromptTransformOperation::Remix
            && root.as_deref().is_some_and(|root| root.trim() != current)
        {
            self.popup = Some(Popup::PromptSourceChoice {
                current_prompt: current,
                root_prompt: root.expect("checked above"),
                cursor: 0,
            });
            return;
        }
        let source_kind = if operation == PromptTransformOperation::Expand {
            mold_core::RemixSourceKind::Current
        } else if root.is_some() {
            mold_core::RemixSourceKind::Original
        } else {
            mold_core::RemixSourceKind::Direct
        };
        let source_prompt = match source_kind {
            mold_core::RemixSourceKind::Original => root.clone().unwrap_or(current),
            mold_core::RemixSourceKind::Current | mold_core::RemixSourceKind::Direct => current,
        };
        self.start_prompt_transform_from(operation, source_prompt, source_kind);
    }

    fn start_prompt_transform_from(
        &mut self,
        operation: PromptTransformOperation,
        source_prompt: String,
        source_kind: mold_core::RemixSourceKind,
    ) {
        let current_prompt = self.generate.prompt.lines().join("\n").trim().to_string();
        let root = self
            .generate
            .params
            .original_prompt
            .clone()
            .filter(|root| !root.trim().is_empty());
        let family = family_for_model(&self.generate.params.model, &self.config);
        let task = self.prompt_transform_task();
        let reference_fingerprint = match self.reference_fingerprint_for_target(&self.target) {
            Ok(fingerprint) => fingerprint,
            Err(error) => {
                self.generate.error_message = Some(error);
                return;
            }
        };
        let snapshot = PromptTransformSnapshot {
            operation,
            model: self.generate.params.model.clone(),
            target: self.target.clone(),
            task,
            reference_fingerprint,
            source_prompt: source_prompt.clone(),
            current_prompt,
            root_prompt: root.clone(),
            source_kind,
        };
        let context = self.prompt_transform_context(&family);
        let request = mold_core::RemixRequest {
            source_prompt,
            root_prompt: root,
            source_kind,
            model_family: family,
            variations: if operation == PromptTransformOperation::Remix {
                3
            } else {
                1
            },
            style: None,
            task: Some(task),
            dimensions: Vec::new(),
            context: Some(context),
        };

        use crate::hosts::GenTarget;
        let (url, api_key) = match &self.target {
            GenTarget::Host(id) => match self.machines.registry.get(id) {
                Some(entry) => (Some(entry.url.clone()), crate::hosts::api_key_for(id)),
                None => {
                    self.generate.error_message = Some(format!(
                        "Machine '{id}' is no longer saved. Pick a target in Machines (4)."
                    ));
                    return;
                }
            },
            GenTarget::Local => (None, None),
            GenTarget::Auto => (
                self.server_url.clone(),
                std::env::var("MOLD_API_KEY")
                    .ok()
                    .filter(|key| !key.is_empty()),
            ),
        };
        self.generate.prompt_transform_token = self.generate.prompt_transform_token.wrapping_add(1);
        let token = self.generate.prompt_transform_token;
        self.generate.error_message = None;
        let tx = self.bg_tx.clone();
        self.tokio_handle.spawn(async move {
            crate::backend::run_prompt_transform(
                url, api_key, operation, request, snapshot, token, tx,
            )
            .await;
        });
    }

    fn prompt_provenance(
        snapshot: &PromptTransformSnapshot,
        dimensions: Vec<mold_core::RemixDimension>,
    ) -> mold_core::PromptTransformProvenance {
        mold_core::PromptTransformProvenance {
            operation: snapshot.operation,
            root_prompt: snapshot.root_prompt.clone(),
            source_prompt: snapshot.source_prompt.clone(),
            source_kind: snapshot.source_kind,
            task: snapshot.task,
            dimensions,
        }
    }

    fn prompt_transform_staleness(&self, snapshot: &PromptTransformSnapshot) -> Option<String> {
        if self.generate.params.model != snapshot.model {
            return Some("Remix is stale because the model changed; remix again".into());
        }
        if self.target != snapshot.target {
            return Some(
                "Remix is stale because the generation target changed; remix again".into(),
            );
        }
        if self.prompt_transform_task() != snapshot.task {
            return Some(
                "Remix is stale because the conditioning task changed; remix again".into(),
            );
        }
        let reference_fingerprint = match self.reference_fingerprint_for_target(&self.target) {
            Ok(fingerprint) => fingerprint,
            Err(error) => return Some(error),
        };
        if reference_fingerprint != snapshot.reference_fingerprint {
            return Some(
                "Remix is stale because the ordered references changed; remix again".into(),
            );
        }
        let current = self.generate.prompt.lines().join("\n").trim().to_string();
        if current != snapshot.current_prompt {
            return Some("Remix is stale because the current prompt changed; remix again".into());
        }
        if self.generate.params.original_prompt.as_deref() != snapshot.root_prompt.as_deref() {
            return Some("Remix is stale because the original prompt changed; remix again".into());
        }
        None
    }

    /// Validate reviewed Batch-1 semantics while deliberately ignoring the
    /// snapshotted host. Quick work may be submitted on the current target;
    /// prepared Batch N remains route-frozen in `start_generation`.
    fn quick_transform_staleness(&self, snapshot: &PromptTransformSnapshot) -> Option<String> {
        if self.generate.params.model != snapshot.model {
            return Some("model changed".into());
        }
        if self.prompt_transform_task() != snapshot.task {
            return Some("conditioning task changed".into());
        }
        let reference_fingerprint = match self.reference_fingerprint_for_target(&self.target) {
            Ok(fingerprint) => fingerprint,
            Err(error) => return Some(error),
        };
        if reference_fingerprint != snapshot.reference_fingerprint {
            return Some("ordered references changed".into());
        }
        let current = self.generate.prompt.lines().join("\n").trim().to_string();
        if current != snapshot.current_prompt {
            return Some("reviewed prompt changed".into());
        }
        let Some(provenance) = self.generate.params.prompt_transform.as_ref() else {
            return Some("transform provenance changed".into());
        };
        if provenance.operation != snapshot.operation
            || provenance.source_prompt != snapshot.source_prompt
            || provenance.root_prompt != snapshot.root_prompt
            || provenance.source_kind != snapshot.source_kind
            || provenance.task != snapshot.task
        {
            return Some("transform provenance changed".into());
        }
        None
    }

    fn response_matches_snapshot(
        snapshot: &PromptTransformSnapshot,
        response: &RemixResponse,
    ) -> bool {
        response.source_prompt == snapshot.source_prompt
            && response.root_prompt == snapshot.root_prompt
            && response.source_kind == snapshot.source_kind
            && response.task == snapshot.task
    }

    fn apply_prompt_variant(
        &mut self,
        snapshot: PromptTransformSnapshot,
        variant: mold_core::RemixVariant,
    ) {
        self.generate.params.original_prompt = Some(
            snapshot
                .root_prompt
                .clone()
                .unwrap_or_else(|| snapshot.source_prompt.clone()),
        );
        self.generate.params.prompt_transform =
            Some(Self::prompt_provenance(&snapshot, variant.dimensions));
        let mut applied_snapshot = snapshot;
        applied_snapshot.current_prompt = variant.prompt.clone();
        self.generate.params.quick_transform_snapshot = Some(applied_snapshot);
        self.generate.params.prepared_prompts.clear();
        self.generate.params.prepared_prompt_transforms.clear();
        self.generate.params.prepared_transform_snapshot = None;
        self.set_prompt_text(&variant.prompt);
    }

    fn prepare_prompt_variants(
        &mut self,
        snapshot: PromptTransformSnapshot,
        variants: Vec<mold_core::RemixVariant>,
    ) {
        self.generate.params.original_prompt = Some(
            snapshot
                .root_prompt
                .clone()
                .unwrap_or_else(|| snapshot.source_prompt.clone()),
        );
        self.generate.params.prepared_prompts = variants
            .iter()
            .map(|variant| variant.prompt.clone())
            .collect();
        self.generate.params.prepared_prompt_transforms = variants
            .iter()
            .map(|variant| Self::prompt_provenance(&snapshot, variant.dimensions.clone()))
            .collect();
        self.generate.params.quick_transform_snapshot = None;
        self.generate.params.prepared_transform_snapshot = Some(snapshot);
        self.generate.params.batch = variants.len() as u32;
        if let Some(first) = variants.first() {
            self.set_prompt_text(&first.prompt);
        }
    }

    /// Drain and process all pending background events.
    pub fn process_background_events(&mut self) {
        while let Ok(event) = self.bg_rx.try_recv() {
            match event {
                BackgroundEvent::Progress(sse) => self.handle_progress(sse),
                BackgroundEvent::PromptTransformComplete {
                    token,
                    operation,
                    snapshot,
                    response,
                } => {
                    if token != self.generate.prompt_transform_token {
                        continue;
                    }
                    if operation != snapshot.operation
                        || !Self::response_matches_snapshot(&snapshot, &response)
                    {
                        self.generate.error_message = Some(
                            "Prompt transform response did not match its frozen request; remix again"
                                .into(),
                        );
                        continue;
                    }
                    if let Some(message) = self.prompt_transform_staleness(&snapshot) {
                        self.generate.error_message = Some(message);
                        continue;
                    }
                    let selected = vec![false; response.variants.len()];
                    self.popup = Some(Popup::PromptAlternatives {
                        snapshot,
                        variants: response.variants,
                        selected,
                        cursor: 0,
                    });
                }
                BackgroundEvent::PromptTransformFailed { token, message } => {
                    if token == self.generate.prompt_transform_token {
                        self.generate.error_message = Some(message);
                    }
                }
                BackgroundEvent::GenerationComplete {
                    response,
                    from_local,
                    metadata_snapshot,
                } => {
                    self.generate.clear_live_preview();
                    let GenerationMetadataSnapshot {
                        params: submitted_params,
                        prompt: prompt_text,
                        negative_prompt,
                    } = *metadata_snapshot;
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

                    self.surface_request_advisories(&response.request_warnings);

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

                    let neg_text = negative_prompt.clone().unwrap_or_default();

                    // A titled print's file carries `~<slug>`, the same shape
                    // the server's gallery writes, so a local copy of a
                    // remote render is recognizable by the same name.
                    let title_slug = submitted_params
                        .title
                        .as_deref()
                        .and_then(mold_core::title_slug);

                    for (i, img_data) in response.images.iter().enumerate() {
                        let ext = img_data.format.extension();
                        let filename = mold_core::default_output_filename_titled(
                            &actual_model,
                            ts_secs,
                            ext,
                            response.images.len() as u32,
                            i as u32,
                            title_slug.as_deref(),
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
                        let filename = mold_core::default_output_filename_titled(
                            &actual_model,
                            ts_secs,
                            ext,
                            1,
                            0,
                            title_slug.as_deref(),
                        );
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

                    // Handle mesh output: save the `.glb`, cache its poster
                    // beside it under the same thumbnail key a gallery scan
                    // will look up (`thumbnails::thumbnail_path`), and show
                    // the poster — a raster decoder cannot read glTF, and the
                    // Preview panel has no 3-D renderer, so the poster is the
                    // only picture there is, exactly as it is in the grids.
                    self.generate.last_mesh_summary = None;
                    if let Some(ref mesh) = response.mesh {
                        let ext = mesh.format.extension();
                        let filename = mold_core::default_output_filename_titled(
                            &actual_model,
                            ts_secs,
                            ext,
                            1,
                            0,
                            title_slug.as_deref(),
                        );
                        if let Some(ref dir) = output_dir {
                            let path = dir.join(&filename);
                            if std::fs::write(&path, &mesh.data).is_ok() {
                                saved_path = path.clone();
                                if !mesh.poster.is_empty() {
                                    crate::thumbnails::save_thumbnail_bytes(&mesh.poster, &path)
                                        .ok();
                                }
                            }
                        }
                        if let Ok(img) = image::load_from_memory(&mesh.poster) {
                            let protocol = self.picker.new_resize_protocol(img.clone());
                            self.generate.preview_image = Some(img);
                            self.generate.image_state = Some(protocol);
                            self.generate.animation = None;
                        }
                        self.generate.last_mesh_summary = Some(crate::ui::preview::mesh_summary(
                            mesh.vertex_count,
                            mesh.face_count,
                            mesh.bounds_min,
                            mesh.bounds_max,
                        ));
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
                    } else if let Some(ref mesh) = response.mesh {
                        // A mesh has no raster size; the row records the
                        // poster's, exactly as the server's gallery does.
                        (mesh.poster_width, mesh.poster_height)
                    } else {
                        (submitted_params.width, submitted_params.height)
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

                    if (!response.images.is_empty()
                        || response.video.is_some()
                        || response.mesh.is_some())
                        && !saved_path.as_os_str().is_empty()
                    {
                        // The filing this print was submitted under. The
                        // local DB seeds tags and collection membership from
                        // these embedded fields on insert, so a forced-local
                        // render files itself exactly as a served one does.
                        let submitted_filing = crate::ui::create_form::compose_filing_tags(
                            &submitted_params.tags,
                            submitted_params.title.as_deref(),
                            submitted_params.auto_tag_title,
                        )
                        .map(|composed| composed.tags)
                        .unwrap_or_else(|_| submitted_params.tags.clone());
                        let meta = mold_core::OutputMetadata {
                            video_only: None,
                            attention_path: None,
                            int8_arm: None,
                            collection: submitted_params.collection.clone(),
                            tags: (!submitted_filing.is_empty()).then_some(submitted_filing),
                            title: submitted_params.title.clone(),
                            source_fit: None,
                            guidance_overrides: submitted_params
                                .guidance_overrides
                                .clone()
                                .into_option(),
                            sample_shift: submitted_params.sample_shift,
                            distill_strength_high: None,
                            distill_strength_low: None,
                            job_id: None,
                            prompt: prompt_text,
                            negative_prompt,
                            original_prompt: submitted_params.original_prompt.clone(),
                            prompt_transform: submitted_params.prompt_transform.clone(),
                            batch_id: submitted_params.batch_id.clone(),
                            batch_index: submitted_params.batch_index,
                            batch_count: submitted_params.batch_count,
                            output_mode: Some(mold_core::GenerationOutputMode::OneShot),
                            model: actual_model,
                            seed: response.seed_used,
                            steps: submitted_params.steps,
                            guidance: submitted_params.guidance,
                            width: entry_width,
                            height: entry_height,
                            generation_width: Some(entry_width),
                            generation_height: Some(entry_height),
                            strength: if submitted_params.source_image_path.is_some() {
                                Some(submitted_params.strength)
                            } else {
                                None
                            },
                            source_image_name: None,
                            source_image_sha256: None,
                            edit_image_sha256s: None,
                            references: None,
                            keyframes: None,
                            scheduler: submitted_params.scheduler,
                            output_format: Some(submitted_params.format),
                            cfg_plus: None,
                            lora: submitted_params.lora_path.clone(),
                            lora_scale: submitted_params
                                .lora_path
                                .as_ref()
                                .map(|_| submitted_params.lora_scale),
                            loras: submitted_params.lora_path.as_ref().map(|path| {
                                vec![mold_core::LoraWeight {
                                    path: path.clone(),
                                    scale: submitted_params.lora_scale,

                                    expert: None,
                                }]
                            }),
                            control_model: submitted_params.control_model.clone(),
                            control_scale: submitted_params
                                .control_image_path
                                .as_ref()
                                .and(submitted_params.control_model.as_ref())
                                .map(|_| submitted_params.control_scale),
                            upscale_model: None,
                            gif_preview: response.video.as_ref().map(|_| true),
                            enable_audio: submitted_params.enable_audio,
                            audio_file_path: None,
                            source_video_path: None,
                            extend_video_path: None,
                            extend_overlap_frames: None,
                            pipeline: response.video.as_ref().and_then(|video| video.pipeline),
                            pipeline_requested: Some(submitted_params.pipeline.is_some()),
                            duration_prediction_requested: Some(
                                submitted_params.predict_duration
                                    && submitted_params.duration_prediction_supported,
                            ),
                            source_preprocessing: response
                                .video
                                .as_ref()
                                .and_then(|video| video.source_preprocessing.clone()),
                            pipeline_provenance_sha256: response
                                .video
                                .as_ref()
                                .and_then(|video| video.pipeline_provenance_sha256.clone()),
                            ic_lora_control: None,
                            hdr_exr_dir: None,
                            hdr_exr_full_float: false,
                            retake_range: None,
                            spatial_upscale: submitted_params.spatial_upscale,
                            temporal_upscale: submitted_params.temporal_upscale,
                            chain_job_id: None,
                            chain: None,
                            version: mold_core::build_info::VERSION.to_string(),
                            frames: response.video.as_ref().map(|v| v.frames),
                            fps: response.video.as_ref().map(|v| v.fps),
                            id_image_name: None,
                            id_image_sha256: None,
                            id_weight: None,
                            id_start_step: None,
                            id_image_names: None,
                            id_image_sha256s: None,
                            true_cfg: None,
                            cfg_start_step: None,
                        };

                        if let (Ok(Some(db)), Some(output_dir)) =
                            (mold_db::open_default(), saved_path.parent())
                        {
                            let saved_format = response
                                .video
                                .as_ref()
                                .map(|video| video.format)
                                .or_else(|| response.mesh.as_ref().map(|mesh| mesh.format))
                                .or_else(|| response.images.first().map(|image| image.format))
                                .unwrap_or(submitted_params.format);
                            mold_db::persist::record_saved_output(
                                &db,
                                output_dir,
                                &saved_name,
                                &saved_path,
                                &mold_db::persist::OutputRecordParams {
                                    format: saved_format,
                                    metadata: &meta,
                                    source: mold_db::RecordSource::Tui,
                                    generation_time_ms: Some(
                                        response.generation_time_ms.try_into().unwrap_or(i64::MAX),
                                    ),
                                    backend: Some(mold_inference::compiled_backend_label()),
                                },
                            );
                        }

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
                                title: None,
                                origins: vec![GalleryOrigin::local()],
                            },
                        );
                        self.gallery.thumbnail_states.insert(0, None);
                        self.gallery.thumb_dimensions.insert(0, None);
                        self.gallery.thumb_fixed_cache.insert(0, None);
                        self.gallery.refresh_filter();

                        // Generate thumbnail in background. A mesh already
                        // has its poster cached above, and a raster decoder
                        // could not read the `.glb` anyway.
                        if response.mesh.is_none() {
                            self.tokio_handle.spawn(async move {
                                tokio::task::spawn_blocking(move || {
                                    crate::thumbnails::generate_thumbnail(&saved_path).ok();
                                })
                                .await
                                .ok();
                            });
                        }
                    }
                }
                BackgroundEvent::DurableGenerationBatchComplete {
                    outcomes,
                    prompt,
                    negative_prompt,
                    model,
                    host,
                } => {
                    self.generate.generating = false;
                    self.generate.batch_remaining = 0;
                    self.generate.clear_live_preview();
                    self.generate.progress.generation_started_at = None;
                    self.generate.progress.stage_started_at = None;

                    let mut failures = Vec::new();
                    let mut completed = 0_usize;
                    self.generate.held_batch = HeldBatch::from_outcomes(
                        host,
                        crate::backend::OwnedBatchSubmission {
                            prompt: prompt.clone(),
                            negative_prompt: negative_prompt.clone(),
                            model: model.clone(),
                        },
                        &outcomes,
                    );
                    for outcome in outcomes {
                        if let Some(seed) = outcome.seed {
                            self.generate.last_seed = Some(seed);
                            // Advance exactly as a singleton does, so a batch
                            // does not leave the form pinned to the seed the
                            // first sibling already rendered.
                            self.generate.params.seed =
                                self.generate.params.seed_mode.advance(seed);
                        }
                        if let Some(elapsed_ms) = outcome.generation_time_ms {
                            self.generate.last_generation_time_ms = Some(elapsed_ms);
                        }
                        if let Some(bytes) = outcome.preview_bytes.as_deref() {
                            if let Ok(img) = image::load_from_memory(bytes) {
                                let protocol = self.picker.new_resize_protocol(img.clone());
                                self.generate.preview_image = Some(img);
                                self.generate.image_state = Some(protocol);
                                self.generate.animation = None;
                            }
                        }
                        let mut filenames = [outcome.filename, outcome.original_filename]
                            .into_iter()
                            .flatten()
                            .collect::<Vec<_>>();
                        filenames.dedup();
                        if !filenames.is_empty() {
                            completed += 1;
                            let elapsed = outcome
                                .generation_time_ms
                                .map(|ms| format!(" ({:.1}s)", ms as f64 / 1000.0))
                                .unwrap_or_default();
                            self.generate.progress.push_log(ProgressLogEntry {
                                message: format!(
                                    "Batch {} saved {}{elapsed}",
                                    outcome.index,
                                    filenames.join(" + ")
                                ),
                                style: ProgressStyle::Done,
                            });
                        } else if let Some(error) = outcome.error {
                            let retry = if outcome.retryable {
                                format!(" (^T retries queue job {})", outcome.job_id)
                            } else {
                                String::new()
                            };
                            let message = format!("Batch {}: {error}{retry}", outcome.index);
                            self.generate.progress.push_log(ProgressLogEntry {
                                message: message.clone(),
                                style: if outcome.retryable {
                                    ProgressStyle::Warning
                                } else {
                                    ProgressStyle::Error
                                },
                            });
                            failures.push(message);
                        }
                    }
                    // A durable batch always renders on a remote host, which
                    // wrote the file into its own gallery: there is no local
                    // path for the activity strip to name, exactly as a
                    // remote singleton has none. The per-child log lines above
                    // carry the filenames.
                    self.generate.last_output_path = None;
                    self.generate.error_message =
                        (!failures.is_empty()).then(|| failures.join("; "));
                    if completed > 0 {
                        self.save_session();
                        mold_db::settings::record_last_model(&model);
                        let ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        self.history.push(crate::history::HistoryEntry {
                            prompt,
                            negative: negative_prompt.filter(|text| !text.is_empty()),
                            model,
                            timestamp: ts,
                        });
                    }
                    if self.should_poll_remote() {
                        self.gallery.scanning = true;
                        self.spawn_gallery_scan();
                    }
                }
                BackgroundEvent::Error(msg) => {
                    self.generate.generating = false;
                    self.generate.batch_remaining = 0;
                    self.generate.clear_live_preview();
                    self.generate.error_message = Some(msg);
                    self.generate.progress.generation_started_at = None;
                    self.generate.progress.stage_started_at = None;
                }
                BackgroundEvent::LicenseListingLoaded {
                    host_label,
                    licenses,
                } => {
                    // Fence on the HOST, not merely on "a popup is open".
                    // Closing a slow listing for one host and reopening
                    // against another would otherwise paint the first host's
                    // acceptances under the second one's heading — the exact
                    // "accepted on which machine?" confusion this screen
                    // exists to answer.
                    if let Some(Popup::LicenseSettings {
                        host_label: open_host,
                        state,
                        ..
                    }) = self.popup.as_mut()
                    {
                        if *open_host == host_label {
                            *state = LicenseListingState::Ready(licenses);
                        }
                    }
                }
                BackgroundEvent::LicenseListingFailed {
                    host_label,
                    message,
                } => {
                    if let Some(Popup::LicenseSettings {
                        host_label: open_host,
                        state,
                        ..
                    }) = self.popup.as_mut()
                    {
                        if *open_host == host_label {
                            *state = LicenseListingState::Failed(message);
                        }
                    }
                }
                BackgroundEvent::LicenseRequired {
                    host_label,
                    requirements,
                    response,
                } => {
                    self.popup = Some(Popup::LicenseReview {
                        host_label,
                        requirements,
                        response: Some(response),
                    });
                }
                BackgroundEvent::GalleryScanComplete(scan) => {
                    self.gallery.offline_hosts = scan.offline_hosts;
                    self.gallery.local_trash_available = scan.local_trash_available;
                    self.apply_gallery_scan(scan.entries);
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
                    self.gallery.thumbnail_loading.clear();
                    self.gallery.thumbnail_lru.clear();
                    self.gallery.details_thumbnail_state = None;
                    self.gallery.details_thumb = None;
                }
                BackgroundEvent::GalleryThumbnailReady { path, image } => {
                    self.gallery.thumbnail_loading.remove(&path);
                    let Some(image) = image else {
                        continue;
                    };
                    let Some(index) = self
                        .gallery
                        .entries
                        .iter()
                        .position(|entry| entry.path == path)
                    else {
                        continue;
                    };
                    if index >= self.gallery.thumbnail_states.len() {
                        continue;
                    }
                    self.gallery.thumb_dimensions[index] = Some((image.width(), image.height()));
                    if index == self.gallery.selected {
                        self.gallery.details_thumbnail_state = Some((
                            index,
                            self.picker.new_resize_protocol(image.clone()),
                            (image.width(), image.height()),
                        ));
                    }
                    self.gallery.thumbnail_states[index] =
                        Some(self.picker.new_resize_protocol(image));
                    self.gallery.thumbnail_lru.retain(|cached| *cached != index);
                    self.gallery.thumbnail_lru.push_back(index);
                    while self.gallery.thumbnail_lru.len() > 64 {
                        let Some(victim) = self.gallery.thumbnail_lru.pop_front() else {
                            break;
                        };
                        if victim == self.gallery.selected {
                            self.gallery.thumbnail_lru.push_back(victim);
                            continue;
                        }
                        if victim < self.gallery.thumbnail_states.len() {
                            self.gallery.thumbnail_states[victim] = None;
                            self.gallery.thumb_dimensions[victim] = None;
                            self.gallery.thumb_fixed_cache[victim] = None;
                        }
                    }
                    if index < self.gallery.thumb_fixed_cache.len() {
                        self.gallery.thumb_fixed_cache[index] = None;
                    }
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
                        self.models.catalog = build_local_model_catalog(&self.config);
                    }
                }
                BackgroundEvent::ModelRemoveComplete(model) => {
                    self.generate.progress.push_log(ProgressLogEntry {
                        message: format!("Removed model: {model}"),
                        style: ProgressStyle::Done,
                    });
                    // Refresh config and catalog
                    self.config = Config::load_or_default();
                    self.models.catalog = build_local_model_catalog(&self.config);
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
                        video_only: None,
                        attention_path: None,
                        int8_arm: None,
                        // An upscale of a filed print stays filed: the copy
                        // is the same picture, and losing its title and tags
                        // would strand it in the Library.
                        collection: source_meta.as_ref().and_then(|m| m.collection.clone()),
                        tags: source_meta.as_ref().and_then(|m| m.tags.clone()),
                        title: source_meta.as_ref().and_then(|m| m.title.clone()),
                        source_fit: None,
                        guidance_overrides: None,
                        sample_shift: None,
                        distill_strength_high: None,
                        distill_strength_low: None,
                        job_id: None,
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
                        prompt_transform: source_meta
                            .as_ref()
                            .and_then(|m| m.prompt_transform.clone()),
                        batch_id: source_meta.as_ref().and_then(|m| m.batch_id.clone()),
                        batch_index: source_meta.as_ref().and_then(|m| m.batch_index),
                        batch_count: source_meta.as_ref().and_then(|m| m.batch_count),
                        output_mode: source_meta.as_ref().and_then(|m| m.output_mode),
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
                        edit_image_sha256s: None,
                        references: None,
                        keyframes: source_meta
                            .as_ref()
                            .and_then(|metadata| metadata.keyframes.clone()),
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
                        extend_video_path: source_meta
                            .as_ref()
                            .and_then(|m| m.extend_video_path.clone()),
                        extend_overlap_frames: source_meta
                            .as_ref()
                            .and_then(|m| m.extend_overlap_frames),
                        pipeline: source_meta.as_ref().and_then(|m| m.pipeline),
                        pipeline_requested: source_meta.as_ref().and_then(|m| m.pipeline_requested),
                        duration_prediction_requested: source_meta
                            .as_ref()
                            .and_then(|m| m.duration_prediction_requested),
                        source_preprocessing: source_meta
                            .as_ref()
                            .and_then(|m| m.source_preprocessing.clone()),
                        pipeline_provenance_sha256: source_meta
                            .as_ref()
                            .and_then(|metadata| metadata.pipeline_provenance_sha256.clone()),
                        ic_lora_control: None,
                        hdr_exr_dir: None,
                        hdr_exr_full_float: false,
                        retake_range: source_meta.as_ref().and_then(|m| m.retake_range.clone()),
                        spatial_upscale: source_meta.as_ref().and_then(|m| m.spatial_upscale),
                        temporal_upscale: source_meta.as_ref().and_then(|m| m.temporal_upscale),
                        frames: None,
                        fps: None,
                        chain_job_id: None,
                        chain: None,
                        version: mold_core::build_info::VERSION.to_string(),
                        id_image_name: None,
                        id_image_sha256: None,
                        id_weight: None,
                        id_start_step: None,
                        id_image_names: None,
                        id_image_sha256s: None,
                        true_cfg: None,
                        cfg_start_step: None,
                    };

                    self.gallery.entries.insert(
                        0,
                        GalleryEntry {
                            path: saved_path.clone(),
                            metadata: meta,
                            generation_time_ms: Some(upscale_time_ms),
                            timestamp: ts,
                            server_url: None,
                            title: None,
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
                BackgroundEvent::FramewiseUpscaleStatus(job) => {
                    self.upscale_tile_progress = (job.total_frames > 0)
                        .then_some((job.completed_frames as usize, job.total_frames as usize));
                    self.upscale_progress.current_stage = Some(match job.state {
                        mold_core::VideoUpscaleJobState::Queued => {
                            "Framewise upscale queued".into()
                        }
                        mold_core::VideoUpscaleJobState::Running => format!(
                            "Framewise upscale {}/{} frames",
                            job.completed_frames, job.total_frames
                        ),
                        mold_core::VideoUpscaleJobState::Finalizing => {
                            "Finalizing Framewise upscale".into()
                        }
                        mold_core::VideoUpscaleJobState::Paused => {
                            "Framewise upscale paused".into()
                        }
                        mold_core::VideoUpscaleJobState::Completed => {
                            "Framewise upscale complete".into()
                        }
                        mold_core::VideoUpscaleJobState::Failed => job
                            .error
                            .clone()
                            .unwrap_or_else(|| "Framewise upscale failed".into()),
                        mold_core::VideoUpscaleJobState::Cancelled => {
                            "Framewise upscale cancelled".into()
                        }
                    });
                    if job.state.is_terminal() {
                        self.upscale_in_progress = false;
                        self.upscale_task = None;
                        if job.state == mold_core::VideoUpscaleJobState::Completed {
                            self.generate.progress.push_log(ProgressLogEntry {
                                message: format!(
                                    "Framewise upscale complete: {}",
                                    job.output_filename
                                        .as_deref()
                                        .unwrap_or("new Library video")
                                ),
                                style: ProgressStyle::Done,
                            });
                            self.spawn_gallery_scan();
                        }
                    }
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
                    self.machines
                        .apply_capabilities(host_id, capabilities.map(|boxed| *boxed));
                }
                BackgroundEvent::HostQueueUpdate { host_id, queue } => {
                    self.machines.apply_queue(host_id, queue);
                }
                BackgroundEvent::HostQueuePageLoaded {
                    host_id,
                    cursor,
                    queue,
                } => {
                    self.machines
                        .apply_queue_continuation(host_id, &cursor, queue);
                }
                BackgroundEvent::HostPollFinished { host_id } => {
                    if self.machines.finish_poll(&host_id) {
                        self.tick_host_polling();
                    }
                }
                BackgroundEvent::ServerStatusPollFinished => {
                    self.server_status_poll_in_flight = false;
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
                                            self.machines.request_queue_refresh(&entry.id);
                                            self.tick_host_polling();
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
                BackgroundEvent::MeshExportComplete(path) => {
                    // The Library has no timeline strip, so the path is
                    // shown where the user is looking: a dismiss-on-any-key
                    // popup, the TUI's toast. The Create timeline gets the
                    // same line so the export is in the session record.
                    self.generate.progress.push_log(ProgressLogEntry {
                        message: format!("Exported {}", path.display()),
                        style: ProgressStyle::Done,
                    });
                    self.popup = Some(Popup::Info {
                        message: format!("Exported to\n{}", path.display()),
                    });
                }
                BackgroundEvent::MeshExportFailed(message) => {
                    self.generate.error_message = Some(format!("Export failed: {message}"));
                    self.popup = Some(Popup::Info {
                        message: format!("Export failed: {message}"),
                    });
                }
                BackgroundEvent::CatalogRefreshed(models) => {
                    self.models.catalog = models;
                    if !self.models.catalog.is_empty()
                        && self.models.selected >= self.models.catalog.len()
                    {
                        self.models.selected = self.models.catalog.len() - 1;
                    }
                    self.sync_generate_capabilities();
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
                BackgroundEvent::ChainComplete {
                    stage_count,
                    request_warnings,
                } => {
                    self.generate.generating = false;
                    self.generate.clear_live_preview();
                    self.generate.progress.generation_started_at = None;
                    self.generate.progress.stage_started_at = None;
                    self.generate.progress.push_log(ProgressLogEntry {
                        message: format!("Chain complete: {stage_count} stages"),
                        style: ProgressStyle::Done,
                    });
                    // A sequence's filing is stamped on the stitched print, so
                    // a host that could not apply it reports it here exactly
                    // as it does for a one-shot.
                    self.surface_request_advisories(&request_warnings);
                }
                BackgroundEvent::ChainError(msg) => {
                    self.generate.generating = false;
                    self.generate.clear_live_preview();
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
        let live_preview = match &event {
            SseProgressEvent::Preview { image, .. } => decode_live_preview(image),
            _ => None,
        };
        let refresh_catalog = reduce_progress_state(&mut self.generate.progress, event);
        if self.generate.generating {
            if let Some(image) = live_preview {
                self.generate.live_preview_image = Some(image);
                self.generate.live_preview_protocol = None;
            }
        }
        if refresh_catalog {
            // Refresh config and catalog after pull
            self.config = Config::load_or_default();
            self.models.catalog = build_local_model_catalog(&self.config);
        }
    }
}

fn build_local_model_catalog(config: &Config) -> Vec<mold_core::ModelInfoExtended> {
    let mut catalog = mold_core::build_model_catalog(config, None, false);
    mold_core::qualify_catalog_generation_delivery(
        &mut catalog,
        mold_core::GenerationDeliveryCapabilities::new(
            cfg!(feature = "mp4"),
            cfg!(feature = "webp"),
        ),
    );
    // Empty profiles are deliberately unavailable. Keeping their rows would
    // let a legacy fallback synthesize controls the local binary cannot serve.
    catalog.retain(|entry| {
        entry
            .generation_profile
            .as_ref()
            .is_none_or(|profile| !profile.recipes.is_empty())
    });
    catalog
}

const MAX_LIVE_PREVIEW_ENCODED_BYTES: usize = 8 * 1024 * 1024;
const MAX_LIVE_PREVIEW_DIMENSION: u32 = 4096;
const MAX_LIVE_PREVIEW_DECODE_BYTES: u64 = 64 * 1024 * 1024;

fn decode_live_preview(encoded: &str) -> Option<image::DynamicImage> {
    use base64::Engine as _;

    if encoded.len() > MAX_LIVE_PREVIEW_ENCODED_BYTES {
        return None;
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    let mut reader =
        image::ImageReader::with_format(std::io::Cursor::new(bytes), image::ImageFormat::Png);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_LIVE_PREVIEW_DIMENSION);
    limits.max_image_height = Some(MAX_LIVE_PREVIEW_DIMENSION);
    limits.max_alloc = Some(MAX_LIVE_PREVIEW_DECODE_BYTES);
    reader.limits(limits);
    reader.decode().ok()
}

/// Stage label for a `Queued` event. The coordinator re-emits this event
/// whenever a waiting job's place in line changes, and the stage is one field
/// rewritten in place, so every position has to render something: skipping
/// one leaves the previous number on screen while the queue keeps draining.
/// Position 0 is the front of the line — legacy single-GPU dispatch also
/// announces 0 as it starts a job, which the running-state events overwrite
/// immediately. Kept in step with the CLI's `queued_status_message`.
fn queued_stage_label(position: usize) -> String {
    if position == 0 {
        "Queued (next up)".to_string()
    } else {
        format!("Queued (position {position})")
    }
}

fn reduce_progress_state(progress: &mut ProgressState, event: SseProgressEvent) -> bool {
    match event {
        SseProgressEvent::Preview { step, total, .. } => {
            // Preview frames are authoritative denoise progress too. Keeping
            // this synchronized matters when transport timing delivers the
            // preview after its paired DenoiseStep event.
            progress.denoise_step = step;
            progress.denoise_total = total;
        }
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
        SseProgressEvent::StageProgress {
            name,
            current,
            total,
        } => {
            progress.current_stage = Some(format!("{name} {current}/{total}"));
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
            progress.current_stage = Some(queued_stage_label(position));
        }
    }
    false
}

impl Drop for App {
    fn drop(&mut self) {
        // Setup failures can drop the app before the event loop gets a chance
        // to call `shutdown`. Never leave an auto-started server or ephemeral
        // launch credential behind in that path.
        self.cleanup_runtime_authority();
    }
}

#[cfg(test)]
mod tests {
    mod held_batch {
        use super::super::*;

        fn outcome(index: u32, retryable: bool) -> DurableGenerationChildOutcome {
            DurableGenerationChildOutcome {
                index,
                job_id: format!("job-{index}"),
                authority: mold_core::GenerationBatchAuthority {
                    batch_id: "batch".into(),
                    client_batch_id: "client".into(),
                    instance_id: "instance".into(),
                },
                filename: None,
                original_filename: None,
                error: retryable.then(|| "model missing".to_string()),
                retryable,
                seed: None,
                generation_time_ms: None,
                preview_bytes: None,
            }
        }

        fn host() -> HeldHost {
            HeldHost {
                url: "http://admitting:7680".into(),
                api_key: Some("key".into()),
            }
        }

        fn submission() -> crate::backend::OwnedBatchSubmission {
            crate::backend::OwnedBatchSubmission {
                prompt: "the prompt that rendered".into(),
                negative_prompt: None,
                model: "flux-dev:q8".into(),
            }
        }

        /// A retry goes to the host that admitted the batch with the
        /// submission that batch rendered — never to the current target or
        /// the current form — and only the held children ride along.
        #[test]
        fn a_held_batch_keeps_the_admitting_host_and_submission() {
            let held = HeldBatch::from_outcomes(
                host(),
                submission(),
                &[outcome(1, false), outcome(2, true), outcome(3, true)],
            )
            .expect("two children are held");
            assert_eq!(held.host, host());
            assert_eq!(held.submission.prompt, "the prompt that rendered");
            assert_eq!(
                held.retries.iter().map(|r| r.index).collect::<Vec<_>>(),
                vec![2, 3]
            );
        }

        #[test]
        fn a_batch_with_nothing_retryable_holds_nothing() {
            assert!(HeldBatch::from_outcomes(host(), submission(), &[outcome(1, false)]).is_none());
        }
    }

    use super::*;
    use base64::Engine as _;

    fn live_preview_png(width: u32, height: u32, rgba: [u8; 4]) -> String {
        let image = image::RgbaImage::from_pixel(width, height, image::Rgba(rgba));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .expect("encode preview PNG");
        base64::engine::general_purpose::STANDARD.encode(bytes.into_inner())
    }

    fn generation_metadata_snapshot(app: &App) -> Box<GenerationMetadataSnapshot> {
        let prompt = app.generate.prompt.lines().join("\n").trim().to_string();
        let negative = app
            .generate
            .negative_prompt
            .lines()
            .join("\n")
            .trim()
            .to_string();
        Box::new(GenerationMetadataSnapshot::new(
            app.generate.params.clone(),
            prompt,
            if negative.is_empty() {
                None
            } else {
                Some(negative)
            },
        ))
    }

    #[tokio::test]
    async fn live_preview_events_replace_the_create_preview_and_advance_progress() {
        let mut app = make_settings_test_app();
        app.generate.generating = true;

        app.bg_tx
            .send(BackgroundEvent::Progress(SseProgressEvent::Preview {
                image: live_preview_png(2, 1, [255, 0, 0, 255]),
                step: 3,
                total: 12,
            }))
            .unwrap();
        app.process_background_events();

        let first = app
            .generate
            .live_preview_image
            .as_ref()
            .expect("valid preview should be installed");
        assert_eq!((first.width(), first.height()), (2, 1));
        assert_eq!(first.to_rgba8().get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(app.generate.progress.denoise_step, 3);
        assert_eq!(app.generate.progress.denoise_total, 12);

        app.bg_tx
            .send(BackgroundEvent::Progress(SseProgressEvent::Preview {
                image: live_preview_png(1, 2, [0, 0, 255, 255]),
                step: 6,
                total: 12,
            }))
            .unwrap();
        app.process_background_events();

        let second = app.generate.live_preview_image.as_ref().unwrap();
        assert_eq!((second.width(), second.height()), (1, 2));
        assert_eq!(second.to_rgba8().get_pixel(0, 0).0, [0, 0, 255, 255]);
        assert_eq!(app.generate.progress.denoise_step, 6);
        assert!(app.generate.live_preview_protocol.is_none());
    }

    #[tokio::test]
    async fn malformed_live_preview_keeps_the_last_valid_frame() {
        let mut app = make_settings_test_app();
        app.generate.generating = true;
        app.bg_tx
            .send(BackgroundEvent::Progress(SseProgressEvent::Preview {
                image: live_preview_png(2, 1, [12, 34, 56, 255]),
                step: 1,
                total: 4,
            }))
            .unwrap();
        app.process_background_events();

        app.bg_tx
            .send(BackgroundEvent::Progress(SseProgressEvent::Preview {
                image: "not-base64".to_string(),
                step: 2,
                total: 4,
            }))
            .unwrap();
        app.process_background_events();

        let retained = app.generate.live_preview_image.as_ref().unwrap();
        assert_eq!(retained.to_rgba8().get_pixel(0, 0).0, [12, 34, 56, 255]);
        assert_eq!(app.generate.progress.denoise_step, 2);
        assert_eq!(app.generate.progress.denoise_total, 4);
    }

    #[test]
    fn live_preview_decode_rejects_oversized_dimensions() {
        let encoded = live_preview_png(MAX_LIVE_PREVIEW_DIMENSION + 1, 1, [1, 2, 3, 255]);
        assert!(decode_live_preview(&encoded).is_none());
    }

    #[tokio::test]
    async fn live_preview_renders_through_the_fixed_protocol_with_visible_progress() {
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.generating = true;
        app.bg_tx
            .send(BackgroundEvent::Progress(SseProgressEvent::Preview {
                image: live_preview_png(16, 8, [40, 80, 120, 255]),
                step: 7,
                total: 20,
            }))
            .unwrap();
        app.process_background_events();

        let text = render_view_to_string(&mut app, 110, 40);
        assert!(
            text.contains("Developing\u{2026} 7/20 \u{00b7} 35%"),
            "live image must retain readable denoise progress:\n{text}"
        );
        assert!(
            app.generate.live_preview_protocol.is_some(),
            "rendering should populate the fixed-protocol cache"
        );
        let protocol_area = app
            .generate
            .live_preview_protocol
            .as_ref()
            .unwrap()
            .2
            .area();
        assert!(
            protocol_area.width > 2,
            "the 16px-wide latent must upscale beyond its native two terminal cells: {protocol_area:?}"
        );
    }

    #[tokio::test]
    async fn generation_error_clears_transient_live_preview_state() {
        let mut app = make_settings_test_app();
        app.generate.generating = true;
        app.bg_tx
            .send(BackgroundEvent::Progress(SseProgressEvent::Preview {
                image: live_preview_png(2, 2, [1, 2, 3, 255]),
                step: 1,
                total: 3,
            }))
            .unwrap();
        app.bg_tx
            .send(BackgroundEvent::Error("fixture failure".to_string()))
            .unwrap();
        app.process_background_events();

        assert!(!app.generate.generating);
        assert!(app.generate.live_preview_image.is_none());
        assert!(app.generate.live_preview_protocol.is_none());
    }

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
    fn bounded_stage_progress_keeps_inner_work_visible() {
        let mut state = ProgressState::default();
        reduce_progress_state(
            &mut state,
            SseProgressEvent::StageProgress {
                name: "Encoding prompt (Gemma, conditional)".to_string(),
                current: 17,
                total: 48,
            },
        );

        assert_eq!(
            state.current_stage.as_deref(),
            Some("Encoding prompt (Gemma, conditional) 17/48")
        );
        assert_eq!(state.denoise_step, 0);
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
                video_only: None,
                attention_path: None,
                int8_arm: None,
                collection: None,
                tags: None,
                title: None,
                source_fit: None,
                guidance_overrides: None,
                sample_shift: None,
                distill_strength_high: None,
                distill_strength_low: None,
                job_id: None,
                prompt: "test".to_string(),
                negative_prompt: None,
                original_prompt: None,
                prompt_transform: None,
                batch_id: None,
                batch_index: None,
                batch_count: None,
                output_mode: None,
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
                edit_image_sha256s: None,
                references: None,
                keyframes: None,
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
                extend_video_path: None,
                extend_overlap_frames: None,
                pipeline: None,
                pipeline_requested: None,
                duration_prediction_requested: None,
                pipeline_provenance_sha256: None,
                source_preprocessing: None,
                ic_lora_control: None,
                hdr_exr_dir: None,
                hdr_exr_full_float: false,
                retake_range: None,
                spatial_upscale: None,
                temporal_upscale: None,
                chain_job_id: None,
                chain: None,
                version: "0.3.1".to_string(),
                frames: None,
                fps: None,
                id_image_name: None,
                id_image_sha256: None,
                id_weight: None,
                id_start_step: None,
                id_image_names: None,
                id_image_sha256s: None,
                true_cfg: None,
                cfg_start_step: None,
            },
            generation_time_ms: Some(5000),
            timestamp: 1234,
            server_url: None,
            title: None,
            origins: Vec::new(),
        };
        assert_eq!(entry.filename(), "mold-flux-1234.png");
    }

    #[test]
    fn gallery_entry_filename_unknown_for_empty_path() {
        let entry = GalleryEntry {
            path: std::path::PathBuf::new(),
            metadata: mold_core::OutputMetadata {
                video_only: None,
                attention_path: None,
                int8_arm: None,
                collection: None,
                tags: None,
                title: None,
                source_fit: None,
                guidance_overrides: None,
                sample_shift: None,
                distill_strength_high: None,
                distill_strength_low: None,
                job_id: None,
                prompt: "test".to_string(),
                negative_prompt: None,
                original_prompt: None,
                prompt_transform: None,
                batch_id: None,
                batch_index: None,
                batch_count: None,
                output_mode: None,
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
                edit_image_sha256s: None,
                references: None,
                keyframes: None,
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
                extend_video_path: None,
                extend_overlap_frames: None,
                pipeline: None,
                pipeline_requested: None,
                duration_prediction_requested: None,
                pipeline_provenance_sha256: None,
                source_preprocessing: None,
                ic_lora_control: None,
                hdr_exr_dir: None,
                hdr_exr_full_float: false,
                retake_range: None,
                spatial_upscale: None,
                temporal_upscale: None,
                chain_job_id: None,
                chain: None,
                version: "0.0.0".to_string(),
                frames: None,
                fps: None,
                id_image_name: None,
                id_image_sha256: None,
                id_weight: None,
                id_start_step: None,
                id_image_names: None,
                id_image_sha256s: None,
                true_cfg: None,
                cfg_start_step: None,
            },
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            title: None,
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
            video_only: None,
            attention_path: None,
            int8_arm: None,
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            job_id: None,
            prompt: "a test prompt".to_string(),
            negative_prompt: Some("blurry".to_string()),
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            output_mode: None,
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
            edit_image_sha256s: None,
            references: None,
            keyframes: None,
            scheduler: None,
            output_format: Some(mold_core::OutputFormat::Png),
            cfg_plus: None,
            lora: Some("/path/to/adapter.safetensors".to_string()),
            lora_scale: Some(0.8),
            loras: Some(vec![mold_core::LoraWeight {
                path: "/path/to/adapter.safetensors".to_string(),
                scale: 0.8,

                expert: None,
            }]),
            control_model: None,
            control_scale: None,
            upscale_model: None,
            gif_preview: None,
            enable_audio: None,
            audio_file_path: None,
            source_video_path: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            pipeline: None,
            pipeline_requested: None,
            duration_prediction_requested: None,
            pipeline_provenance_sha256: None,
            source_preprocessing: None,
            ic_lora_control: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            chain_job_id: None,
            chain: None,
            version: "0.3.1".to_string(),
            frames: None,
            fps: None,
            id_image_name: None,
            id_image_sha256: None,
            id_weight: None,
            id_start_step: None,
            id_image_names: None,
            id_image_sha256s: None,
            true_cfg: None,
            cfg_start_step: None,
        }
    }

    fn make_test_entry() -> GalleryEntry {
        GalleryEntry {
            path: std::path::PathBuf::from("/home/user/.mold/output/mold-flux-1234.png"),
            metadata: make_test_metadata(),
            generation_time_ms: Some(5000),
            timestamp: 1234,
            server_url: None,
            title: None,
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
                negative_default: String::new(),
                negative_explicit_clear: false,
                params,
                focus: GenerateFocus::Navigation,
                param_index: 0,
                rows,
                advanced,
                param_scroll: 0,
                capabilities: caps,
                progress: ProgressState::default(),
                live_preview_image: None,
                live_preview_protocol: None,
                preview_image: None,
                image_state: None,
                animation: None,
                generating: false,
                batch_remaining: 0,
                last_seed: None,
                last_generation_time_ms: None,
                error_message: None,
                identity_error: None,
                warning_message: None,
                model_description: String::new(),
                last_output_path: None,
                held_batch: None,
                prompt_transform_token: 0,
                last_mesh_summary: None,
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
            server_status_poll_in_flight: false,
            // Start test apps with an empty in-memory history — avoids
            // reaching into whatever MOLD_DB_PATH currently points at,
            // which was the source of flakes in parallel runs.
            history: crate::history::PromptHistory::empty(),
            layout: LayoutAreas::default(),
            server_process: None,
            session_api_key_host_id: None,
            strict_gallery_authority: false,
            upscale_in_progress: false,
            upscale_task: None,
            upscale_tile_progress: None,
            upscale_progress: ProgressState::default(),
            connecting: false,
            show_timeline: true,
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_app_cleans_up_an_autostarted_server() {
        let mut app = make_settings_test_app();
        let child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn disposable server stand-in");
        let pid = child.id();
        app.server_process = Some(child);

        drop(app);

        let still_running = std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        assert!(!still_running, "App::drop must reap its background server");
    }

    /// Helper: find the row_index for a given SettingsKey.
    fn find_settings_row(app: &App, key: SettingsKey) -> usize {
        let rows = app.build_settings_rows();
        rows.iter()
            .position(|r| matches!(r, SettingsRow::Field { key: k, .. } if *k == key))
            .unwrap_or_else(|| panic!("SettingsKey {key:?} not found in rows"))
    }

    /// `{:<LABEL_WIDTH}` pads the label column, so a label that is exactly
    /// `LABEL_WIDTH` wide renders flush against its value ("Tag by titleon").
    /// Every label must leave at least one space.
    #[tokio::test]
    async fn every_settings_label_fits_its_column_with_a_gap() {
        let app = make_settings_test_app();
        for row in app.build_settings_rows() {
            if let SettingsRow::Field { label, .. } = row {
                assert!(
                    label.chars().count() < crate::ui::settings::LABEL_WIDTH,
                    "settings label {label:?} is {} chars; the column is {} wide and needs \
                     room for a separating space",
                    label.chars().count(),
                    crate::ui::settings::LABEL_WIDTH
                );
            }
        }
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

    /// Host-RAM pressure has to reach the strip in every state a waiting job
    /// can be in — a queue that stops moving because the host is out of
    /// schedulable RAM looks identical to a slow one otherwise.
    /// The queue stage is one field the strip rewrites in place, so a repeat
    /// of `Queued` reads as a live position. Every position must render —
    /// front of the line included, or the last number the client was told
    /// stays on screen while the queue keeps moving. Wording is kept in step
    /// with the CLI's `queued_status_message`.
    #[test]
    fn every_queued_position_renders_a_stage_label() {
        assert_eq!(queued_stage_label(3), "Queued (position 3)");
        assert_eq!(queued_stage_label(1), "Queued (position 1)");
        assert_eq!(queued_stage_label(0), "Queued (next up)");
    }

    #[test]
    fn repeated_queued_events_rewrite_the_stage_in_place() {
        let mut progress = ProgressState::default();
        for position in (0..=2).rev() {
            reduce_progress_state(
                &mut progress,
                mold_core::SseProgressEvent::Queued {
                    position,
                    id: "job-7".into(),
                },
            );
        }
        assert_eq!(
            progress.current_stage.as_deref(),
            Some("Queued (next up)"),
            "the newest position replaces the previous one rather than stacking"
        );
    }

    #[tokio::test]
    async fn activity_line_reports_host_ram_pressure_only_when_the_host_is_under_it() {
        use crate::ui::chrome::activity_line;
        let mut app = make_settings_test_app();

        let mut status = mold_core::ServerStatus {
            version: "0.22.0".into(),
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
            host_memory: None,
            durable_media: None,
        };

        // A server too old to report the field is the baseline: whatever the
        // strip said before this telemetry existed, it still says.
        app.resource_info.server_status = Some(status.clone());
        let baseline = activity_line(&app).1;
        assert!(!baseline.contains("RAM"), "{baseline}");

        // Nor does a host with room to spare.
        status.host_memory = Some(mold_core::HostMemorySnapshot {
            total_bytes: 64 * 1024_u64.pow(3),
            available_bytes: 48 * 1024_u64.pow(3),
            headroom_bytes: 40 * 1024_u64.pow(3),
            safety_floor_bytes: 10 * 1024_u64.pow(3),
            reclaimable_zfs_arc_bytes: None,
        });
        app.resource_info.server_status = Some(status.clone());
        assert_eq!(activity_line(&app).1, baseline);

        // Under one safety floor of headroom, every state names the pressure.
        status.host_memory = Some(mold_core::HostMemorySnapshot {
            headroom_bytes: 3 * 1024_u64.pow(3),
            ..status.host_memory.unwrap()
        });
        app.resource_info.server_status = Some(status.clone());
        let idle = activity_line(&app).1;
        assert!(
            idle.ends_with(" · RAM tight · 3.0 GB schedulable"),
            "{idle}"
        );

        app.generate.generating = true;
        let generating = activity_line(&app).1;
        assert!(generating.contains("RAM tight"), "{generating}");

        app.generate.generating = false;
        app.generate.last_generation_time_ms = Some(4000);
        let done = activity_line(&app).1;
        assert!(done.contains("RAM tight"), "{done}");
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
            host_memory: None,
            durable_media: None,
        });
        let (state, text) = host_chip(&app);
        assert_eq!(state, ChipState::Ready);
        assert!(text.starts_with("studio"), "{text}");
        // A reported queue depth surfaces once a job state exists (the
        // idle line deliberately omits it, matching the mockup).
        app.generate.last_generation_time_ms = Some(1000);
        let (_, line) = crate::ui::chrome::activity_line(&app);
        assert!(line.contains("queue load 2 reported"), "{line}");
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

    /// #787: selecting a wan model prefills the tuned default into the
    /// Negative editor and records it as the advertised default; leaving the
    /// family clears an untouched editor, while typed text survives the
    /// switch. The wire tri-state itself is pinned by
    /// `create_form::negative_prompt_wire_value`'s unit tests.
    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn wan_model_prefills_the_advertised_default_negative() {
        let wan_default = mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT;
        let mut app = make_settings_test_app();
        assert_eq!(app.generate.negative_default, "");

        app.update_model("wan21-t2v-1.3b:bf16");
        assert_eq!(app.generate.negative_default, wan_default);
        assert_eq!(
            app.generate.negative_prompt.lines().join("\n"),
            wan_default,
            "an untouched editor shows the tuned default"
        );

        // Leaving the family withdraws the untouched default.
        app.update_model("flux2-klein:q8");
        assert_eq!(app.generate.negative_default, "");
        assert_eq!(app.generate.negative_prompt.lines().join("\n"), "");

        // Typed text is user authority across model switches.
        app.generate.negative_prompt = tui_textarea::TextArea::from(["hands"]);
        app.update_model("wan21-t2v-1.3b:bf16");
        assert_eq!(app.generate.negative_default, wan_default);
        assert_eq!(app.generate.negative_prompt.lines().join("\n"), "hands");
    }

    /// #787 round 3: gallery reuse of a print recorded with the explicit
    /// `""` opt-out arms the live marker, so a later capability sync (a
    /// fresher catalog row for the same model) keeps the clear instead of
    /// mistaking the empty editor for "untouched" — and an explicit model
    /// switch resolves the marker back to ordinary prefill rules.
    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn gallery_reuse_explicit_clear_survives_default_reconciliation() {
        let mut app = make_settings_test_app();
        let mut entry = make_test_entry();
        entry.metadata.model = "wan21-t2v-1.3b:bf16".to_string();
        entry.metadata.negative_prompt = Some(String::new());
        app.gallery.entries = vec![entry];
        app.gallery.selected = 0;

        app.load_gallery_into_generate();
        assert!(app.generate.negative_explicit_clear);
        assert_eq!(app.generate.negative_prompt.lines().join("\n"), "");

        // The reconciliation that runs when capabilities refresh must not
        // prefill over the restored opt-out.
        app.sync_generate_capabilities();
        assert_eq!(app.generate.negative_prompt.lines().join("\n"), "");
        assert_eq!(
            crate::ui::create_form::negative_prompt_wire_value(
                &app.generate.negative_prompt.lines().join("\n"),
                &app.generate.negative_default,
                true,
                app.generate.negative_explicit_clear,
            )
            .as_deref(),
            Some(""),
            "the reused opt-out must keep serializing as the explicit \"\""
        );

        // Reuse of a print with an absent negative is NOT an explicit clear.
        let mut untouched = make_test_entry();
        untouched.metadata.model = "wan21-t2v-1.3b:bf16".to_string();
        untouched.metadata.negative_prompt = None;
        app.gallery.entries = vec![untouched];
        app.gallery.selected = 0;
        app.load_gallery_into_generate();
        assert!(!app.generate.negative_explicit_clear);
        assert_eq!(
            app.generate.negative_prompt.lines().join("\n"),
            mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT,
            "absence predates truthful recording: the default conditioned the render"
        );
    }

    #[tokio::test]
    async fn gallery_reuse_restores_only_provenance_backed_by_a_prompt() {
        let mut app = make_settings_test_app();
        let mut transformed = make_test_entry();
        transformed.metadata.prompt = "an expanded lighthouse".into();
        transformed.metadata.original_prompt = Some("a lighthouse".into());
        app.gallery.entries = vec![transformed];
        app.gallery.selected = 0;

        app.load_gallery_into_generate();
        assert_eq!(
            app.generate.params.original_prompt.as_deref(),
            Some("a lighthouse")
        );

        let mut promptless = make_test_entry();
        promptless.metadata.prompt.clear();
        promptless.metadata.original_prompt = Some("stale source".into());
        app.gallery.entries = vec![promptless];
        app.load_gallery_into_generate();
        assert_eq!(app.generate.params.original_prompt, None);
    }

    /// #787 round 2: `App::new` never runs `sync_generate_capabilities`, so
    /// cold start wires `restored_negative_editor_text` (with
    /// `create_form::effective_negative_default`) to resolve the editor.
    /// The pure contract is pinned here because `App::new` itself spawns
    /// real IO.
    #[test]
    fn cold_start_negative_editor_resolution() {
        let wan_default = mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT;
        // Saved text restores verbatim, marker or not.
        assert_eq!(
            restored_negative_editor_text("hands", None, wan_default),
            "hands"
        );
        assert_eq!(
            restored_negative_editor_text("hands", Some(true), wan_default),
            "hands"
        );
        // Untouched (no marker — including pre-marker sessions) prefills the
        // advertised default instead of booting with an empty editor.
        assert_eq!(
            restored_negative_editor_text("", None, wan_default),
            wan_default
        );
        assert_eq!(
            restored_negative_editor_text("", Some(false), wan_default),
            wan_default
        );
        // A persisted explicit clear survives restart as the "" opt-out.
        assert_eq!(
            restored_negative_editor_text("", Some(true), wan_default),
            ""
        );
        assert_eq!(
            crate::ui::create_form::negative_prompt_wire_value(
                &restored_negative_editor_text("", Some(true), wan_default),
                wan_default,
                true,
                true,
            )
            .as_deref(),
            Some("")
        );
        // No default → nothing to prefill.
        assert_eq!(restored_negative_editor_text("", None, ""), "");
    }

    /// #787 round 2: Reset Defaults goes through the model-switch
    /// preservation semantics in `sync_generate_capabilities`, which
    /// deliberately keep a cleared or typed editor — the reset itself must
    /// then explicitly return the editor to the model's default, mirroring
    /// web Reset settings and desktop ⌘N.
    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn reset_defaults_restores_the_negative_editor_to_the_model_default() {
        let wan_default = mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT;
        let mut app = make_settings_test_app();
        app.update_model("wan21-t2v-1.3b:bf16");
        assert_eq!(app.generate.negative_prompt.lines().join("\n"), wan_default);

        // A cleared editor is user authority on a model switch — but Reset
        // Defaults is an explicit return to the model's own defaults.
        app.generate.negative_prompt = tui_textarea::TextArea::default();
        app.reset_params_to_model_defaults();
        assert_eq!(
            app.generate.negative_prompt.lines().join("\n"),
            wan_default,
            "reset must re-prefill the advertised default over a cleared editor"
        );

        // Typed text resets the same way.
        app.generate.negative_prompt = tui_textarea::TextArea::from(["hands"]);
        app.reset_params_to_model_defaults();
        assert_eq!(app.generate.negative_prompt.lines().join("\n"), wan_default);

        // A model without a default resets the editor to empty.
        app.update_model("flux2-klein:q8");
        app.generate.negative_prompt = tui_textarea::TextArea::from(["hands"]);
        app.reset_params_to_model_defaults();
        assert_eq!(app.generate.negative_prompt.lines().join("\n"), "");
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
                title: None,
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
        // The rendered row is the whole evidence for this test, and a bare
        // "got Create" tells you nothing about *why* a column moved. Anything
        // that changes the strip's width budget — the host chip, the version
        // string, a renamed tab — shifts these columns, so print the row.
        let rendered_row: String = (0..tab_bar.width)
            .map(|x| buf[(tab_bar.x + x, tab_row)].symbol())
            .collect();

        for (digit, expected) in digit_to_view {
            let col = (0..tab_bar.width)
                .find(|&x| buf[(tab_bar.x + x, tab_row)].symbol() == *digit)
                .unwrap_or_else(|| {
                    panic!("digit {digit} not found in rendered tab bar: {rendered_row:?}")
                });
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
                "clicking on digit '{digit}' at col {click_col} should land on {expected:?}, got {:?}\nrendered row: {rendered_row:?}\nversion: {:?}",
                click_app.active_view,
                mold_core::build_info::version_string(),
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
    async fn settings_trash_retention_row_is_listed_and_displays_forever_for_zero() {
        let mut app = make_settings_test_app();
        let row = find_settings_row(&app, SettingsKey::GalleryTrashRetentionDays);
        match &app.build_settings_rows()[row] {
            SettingsRow::Field {
                label, field_type, ..
            } => {
                assert_eq!(*label, "Trash (days)");
                assert!(
                    matches!(field_type, SettingsFieldType::Number { min, max, step }
                        if *min == 0.0 && *max == 3650.0 && *step == 1.0),
                    "0..=3650 days, one day per step: {field_type:?}"
                );
            }
            other => panic!("expected a field row, got {other:?}"),
        }
        assert_eq!(
            app.settings_display_value(&SettingsKey::GalleryTrashRetentionDays),
            "30",
            "default retention is 30 days"
        );
        app.config.gallery.trash_retention_days = 0;
        assert_eq!(
            app.settings_display_value(&SettingsKey::GalleryTrashRetentionDays),
            "0 (forever)"
        );
        assert_eq!(
            App::settings_env_override(&SettingsKey::GalleryTrashRetentionDays),
            None,
            "no env override unless MOLD_GALLERY_TRASH_RETENTION_DAYS is set"
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_trash_retention_round_trips_through_the_db_surface() {
        // `gallery.trash_retention_days` is a DB-surface key: adjusting
        // the row must land in the settings table the server's sweeper
        // and `mold config get` read, clamped to 0..=3650.
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            assert_eq!(app.config.gallery.trash_retention_days, 30);
            app.settings_adjust_number(SettingsKey::GalleryTrashRetentionDays, -31.0, 0.0, 3650.0);
            assert_eq!(app.config.gallery.trash_retention_days, 0, "clamped at 0");
            app.settings_adjust_number(SettingsKey::GalleryTrashRetentionDays, 7.0, 0.0, 3650.0);
            assert_eq!(app.config.gallery.trash_retention_days, 7);

            let db = mold_db::open_default().unwrap().expect("isolated DB");
            let mut stored = mold_core::config::GallerySettings::default();
            let applied = mold_db::config_sync::hydrate_gallery_from_db(&db, &mut stored).unwrap();
            assert!(applied, "the DB surface holds the gallery row");
            assert_eq!(stored.trash_retention_days, 7);
            assert_eq!(
                mold_core::config_keys::get_value(
                    &app.config,
                    mold_core::config_keys::GALLERY_TRASH_RETENTION_DAYS_KEY
                )
                .map(|v| v.display())
                .ok(),
                Some("7".to_string()),
                "the shared config_keys view agrees"
            );
        });
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
            mesh: None,
            request_warnings: Vec::new(),
            audio: None,
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
        let metadata_snapshot = generation_metadata_snapshot(&app);
        app.bg_tx
            .send(BackgroundEvent::GenerationComplete {
                response: Box::new(response),
                from_local: false,
                metadata_snapshot,
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

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn local_generation_complete_preserves_guidance_overrides_in_gallery_metadata() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.active_view = View::Create;
            app.generate.generating = true;
            app.generate.batch_remaining = 1;
            app.generate.prompt = TextArea::from(["a guidance provenance test"]);
            let submitted_guidance = Ltx2GuidanceOverrides {
                stg_scale: Some(1.5),
                stg_blocks: Some(vec![1, 3, 5]),
                rescale_scale: Some(0.7),
                modality_scale: Some(3.0),
                skip_step: Some(1),
            };
            app.generate.params.guidance_overrides = submitted_guidance.clone();

            let response = GenerateResponse {
                mesh: None,
                request_warnings: Vec::new(),
                audio: None,
                images: vec![mold_core::ImageData {
                    data: vec![0u8; 4],
                    format: OutputFormat::Png,
                    width: 64,
                    height: 64,
                    index: 0,
                }],
                generation_time_ms: 100,
                model: "ltx-2.3:22b-distilled".to_string(),
                seed_used: 42,
                video: None,
                gpu: None,
            };
            let metadata_snapshot = generation_metadata_snapshot(&app);
            app.bg_tx
                .send(BackgroundEvent::GenerationComplete {
                    response: Box::new(response),
                    from_local: true,
                    metadata_snapshot,
                })
                .unwrap();

            // The user may edit the form while inference is still running.
            // Completion metadata must use the submitted request snapshot.
            app.generate.params.guidance_overrides = Ltx2GuidanceOverrides::default();

            app.process_background_events();

            assert_eq!(app.gallery.entries.len(), 1);
            assert_eq!(
                app.gallery.entries[0].metadata.guidance_overrides,
                Some(submitted_guidance),
                "local gallery metadata must record submitted LTX-2 guidance overrides"
            );
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn local_video_completion_preserves_the_runtime_resolved_pipeline() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.active_view = View::Create;
            app.generate.generating = true;
            app.generate.batch_remaining = 1;
            app.generate.prompt = TextArea::from(["a runtime pipeline provenance test"]);
            app.generate.negative_prompt = TextArea::from(["submitted negative"]);
            app.generate.params.format = OutputFormat::Mp4;
            app.generate.params.steps = 28;
            app.generate.params.guidance = 4.0;

            let response = GenerateResponse {
                mesh: None,
                request_warnings: Vec::new(),
                audio: None,
                images: Vec::new(),
                generation_time_ms: 100,
                model: "ltx-2.3-22b-dev:fp8".to_string(),
                seed_used: 42,
                video: Some(mold_core::VideoData {
                    video_only: None,
                    attention_path: None,
                    int8_arm: None,
                    data: b"test-mp4".to_vec(),
                    format: OutputFormat::Mp4,
                    width: 1216,
                    height: 704,
                    frames: 97,
                    fps: 24,
                    pipeline: Some(mold_core::Ltx2PipelineMode::TwoStageHq),
                    pipeline_provenance_sha256: None,
                    source_preprocessing: None,
                    thumbnail: Vec::new(),
                    gif_preview: Vec::new(),
                    has_audio: true,
                    duration_ms: None,
                    audio_sample_rate: None,
                    audio_channels: None,
                }),
                gpu: None,
            };
            let metadata_snapshot = generation_metadata_snapshot(&app);
            app.bg_tx
                .send(BackgroundEvent::GenerationComplete {
                    response: Box::new(response),
                    from_local: true,
                    metadata_snapshot,
                })
                .unwrap();

            // The user can prepare their next draft while inference is running.
            // Every durable field must come from the submitted request snapshot,
            // while runtime-only fields such as pipeline come from the response.
            app.generate.prompt = TextArea::from(["a later draft"]);
            app.generate.negative_prompt = TextArea::from(["later negative"]);
            app.generate.params.steps = 99;
            app.generate.params.guidance = 9.0;

            app.process_background_events();

            assert_eq!(app.gallery.entries.len(), 1);
            assert_eq!(
                app.gallery.entries[0].metadata.pipeline,
                Some(mold_core::Ltx2PipelineMode::TwoStageHq),
                "local TUI metadata must retain the pipeline the engine actually ran"
            );
            assert_eq!(
                app.gallery.entries[0].metadata.prompt,
                "a runtime pipeline provenance test"
            );
            assert_eq!(
                app.gallery.entries[0].metadata.negative_prompt.as_deref(),
                Some("submitted negative")
            );
            assert_eq!(app.gallery.entries[0].metadata.steps, 28);
            assert!((app.gallery.entries[0].metadata.guidance - 4.0).abs() < f64::EPSILON);

            let saved_path = app.gallery.entries[0].path.clone();
            let saved_dir = saved_path.parent().unwrap();
            let saved_name = saved_path.file_name().unwrap().to_str().unwrap();
            let db = mold_db::open_default().unwrap().unwrap();
            let persisted = db
                .get(saved_dir, saved_name)
                .unwrap()
                .expect("the local TUI save must survive a Library rescan");
            assert_eq!(
                persisted.metadata.pipeline,
                Some(mold_core::Ltx2PipelineMode::TwoStageHq),
                "the runtime pipeline must survive process restart and gallery reconciliation"
            );
        });
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
                mesh: None,
                request_warnings: Vec::new(),
                audio: None,
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
            let metadata_snapshot = generation_metadata_snapshot(&app);
            app.bg_tx
                .send(BackgroundEvent::GenerationComplete {
                    response: Box::new(response),
                    from_local: false,
                    metadata_snapshot,
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
    async fn profile_size_row_cycles_only_qualified_z_image_presets() {
        let mut app = make_settings_test_app();
        app.models.catalog = mold_core::build_model_catalog(&app.config, None, false);
        app.generate.params.model = "z-image-turbo:q4".to_string();
        app.generate.params.width = 1024;
        app.generate.params.height = 1024;

        let recipe = app.active_generation_recipe().expect("profile");
        let expected = recipe
            .resolution
            .aspect_groups
            .iter()
            .flat_map(|group| &group.presets)
            .map(|preset| (preset.width, preset.height))
            .collect::<std::collections::HashSet<_>>();
        assert!(expected.contains(&(1280, 720)));
        assert!(expected.contains(&(720, 1280)));
        for _ in 0..expected.len() {
            assert!(expected.contains(&(app.generate.params.width, app.generate.params.height)));
            app.adjust_field(ParamField::Size, 1);
        }
        assert_eq!(
            (app.generate.params.width, app.generate.params.height),
            (1024, 1024),
            "the exact qualified profile ring must wrap without synthesizing a canvas"
        );
    }

    #[test]
    fn video_duration_ceiling_tracks_fps_and_the_absolute_guard() {
        let ltx2_grid = TuiVideoGrid {
            step: 8,
            offset: 1,
            min_frames: 1,
            fixed_fps: None,
            runtime_seconds: Some(20),
            absolute_frames: Some(604),
            fixed_frames: 481,
        };
        assert_eq!(tui_max_video_frames(ltx2_grid, 12), 241);
        assert_eq!(tui_max_video_frames(ltx2_grid, 24), 481);
        assert_eq!(tui_max_video_frames(ltx2_grid, 48), 601);
        assert_eq!(
            tui_max_video_frames(
                TuiVideoGrid {
                    runtime_seconds: None,
                    absolute_frames: None,
                    fixed_frames: 1,
                    ..ltx2_grid
                },
                24,
            ),
            1
        );
    }

    #[test]
    fn h3_video_grid_keeps_the_five_offset_and_fixed_fps() {
        let h3_grid = TuiVideoGrid {
            step: mold_core::minimax_h3::FRAME_STEP,
            offset: mold_core::minimax_h3::FRAME_OFFSET,
            min_frames: mold_core::minimax_h3::MIN_FRAMES,
            fixed_fps: Some(mold_core::minimax_h3::FIXED_FPS),
            runtime_seconds: Some(mold_core::minimax_h3::MAX_DURATION_SECONDS),
            absolute_frames: Some(mold_core::minimax_h3::MAX_FRAMES),
            fixed_frames: mold_core::minimax_h3::MAX_FRAMES,
        };

        assert_eq!(h3_grid.snap_nearest(120), 124);
        assert_eq!(h3_grid.snap_nearest(240), 243);
        // Grid rounding is independent from the model cap; the UI clamps the
        // rounded value through `tui_max_video_frames` before storing it.
        assert_eq!(h3_grid.snap_nearest(360), 362);
        assert_eq!(tui_max_video_frames(h3_grid, 24), 345);
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
    async fn audio_row_cycles_default_on_off() {
        use crate::ui::create_form::{AdvSection, CreateRow};
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        app.generate.capabilities = crate::model_info::capabilities_for_family("ltx2");
        app.generate.advanced.open = true;
        app.generate.advanced.expanded = Some(AdvSection::Video);
        app.refresh_create_rows();
        let audio_idx = app
            .generate
            .rows
            .iter()
            .position(|r| *r == CreateRow::SectionField(AdvSection::Video, ParamField::Audio))
            .expect("LTX-2 Video section must expose Audio");
        app.generate.param_index = audio_idx;

        assert_eq!(app.generate.params.enable_audio, None);
        app.increment_param(1);
        assert_eq!(app.generate.params.enable_audio, Some(true));
        assert_eq!(
            app.generate.params.format,
            OutputFormat::Mp4,
            "audio output must select the only compatible container"
        );
        app.increment_param(1);
        assert_eq!(app.generate.params.enable_audio, Some(false));
        app.increment_param(1);
        assert_eq!(app.generate.params.enable_audio, None);

        app.dispatch_action(Action::Confirm);
        assert_eq!(app.generate.params.enable_audio, Some(true));
    }

    #[tokio::test]
    async fn h3_capability_sync_freezes_mandatory_av_wire_defaults() {
        use crate::ui::create_form::CreateRow;

        let mut app = make_settings_test_app();
        app.generate.params.model = mold_core::minimax_h3::FL2VA_COMFY.into();
        app.generate.params.enable_audio = Some(false);
        app.generate.params.format = OutputFormat::Gif;
        app.generate.params.guidance = 7.5;
        app.generate.params.strength = 0.25;
        app.generate.params.frames = 25;
        app.generate.params.fps = 30;
        app.generate.params.scheduler = Some(Scheduler::Ddim);
        app.generate.params.lora_path = Some("stale.safetensors".into());
        app.generate.params.source_image_path = Some("stale.png".into());
        app.generate.params.mask_image_path = Some("stale-mask.png".into());
        app.generate.params.control_image_path = Some("stale-control.png".into());
        app.generate.params.control_model = Some("stale-control".into());
        app.config.models.insert(
            app.generate.params.model.clone(),
            mold_core::ModelConfig {
                family: Some(mold_core::minimax_h3::FAMILY.into()),
                ..Default::default()
            },
        );

        app.sync_generate_capabilities();

        assert!(app.generate.capabilities.audio_required);
        assert_eq!(app.generate.params.enable_audio, Some(true));
        assert_eq!(app.generate.params.format, OutputFormat::Mp4);
        assert_eq!(app.generate.params.guidance, 0.0);
        assert_eq!(app.generate.params.strength, 1.0);
        // The params carried a stale 25, which snaps to the nearest grid
        // point rather than to the default clip length.
        assert_eq!(
            app.generate.params.frames,
            mold_core::minimax_h3::MIN_FRAMES
        );
        assert_eq!(app.generate.params.fps, mold_core::minimax_h3::FIXED_FPS);
        assert_eq!(app.generate.params.scheduler, None);
        assert_eq!(app.generate.params.lora_path, None);
        assert_eq!(app.generate.params.source_image_path, None);
        assert_eq!(app.generate.params.mask_image_path, None);
        assert_eq!(app.generate.params.control_image_path, None);
        assert_eq!(app.generate.params.control_model, None);
        assert!(!app.generate.rows.iter().any(|row| matches!(
            row,
            CreateRow::Field(ParamField::Audio) | CreateRow::SectionField(_, ParamField::Audio)
        )));

        app.adjust_field(ParamField::Format, 1);
        assert_eq!(app.generate.params.format, OutputFormat::Mp4);
        assert_eq!(app.generate.params.enable_audio, Some(true));

        // The clip length is adjustable now: one step down from the default
        // is the previous grid point.
        app.generate.params.frames = mold_core::minimax_h3::DEFAULT_COMPACT_FRAMES;
        app.adjust_field(ParamField::Frames, -1);
        assert_eq!(
            app.generate.params.frames,
            mold_core::minimax_h3::DEFAULT_COMPACT_FRAMES - mold_core::minimax_h3::FRAME_STEP
        );
        app.generate.params.fps = 12;
        app.adjust_field(ParamField::Fps, 1);
        assert_eq!(app.generate.params.fps, mold_core::minimax_h3::FIXED_FPS);

        // An off-grid 25 snaps to the family floor, then steps one grid
        // point up.
        app.generate.params.frames = 25;
        app.adjust_field(ParamField::Frames, 1);
        assert_eq!(
            app.generate.params.frames,
            mold_core::minimax_h3::MIN_FRAMES + mold_core::minimax_h3::FRAME_STEP
        );
    }

    #[test]
    fn h3_boot_session_overlay_is_repaired_before_form_construction() {
        let mut params = GenerateParams::from_config(&Config::default());
        params.model = mold_core::minimax_h3::FL2VA_COMFY.into();
        params.frames = 25;
        params.fps = 30;
        params.source_image_path = Some("stale.png".into());
        let session = crate::session::TuiSession {
            guidance: Some(7.5),
            format: Some("png".into()),
            scheduler: Some("ddim".into()),
            lora_path: Some("stale.safetensors".into()),
            strength: Some(0.25),
            ..Default::default()
        };

        session.apply_to_params(&mut params);
        normalize_generate_params_for_family(&mut params, "minimax_h3");

        // 25 is off the grid; it repairs to the nearest valid clip length.
        assert_eq!(params.frames, mold_core::minimax_h3::MIN_FRAMES);
        assert_eq!(params.fps, mold_core::minimax_h3::FIXED_FPS);
        assert_eq!(params.format, OutputFormat::Mp4);
        assert_eq!(params.enable_audio, Some(true));
        assert_eq!(params.guidance, 0.0);
        assert_eq!(params.strength, 1.0);
        assert_eq!(params.scheduler, None);
        assert_eq!(params.lora_path, None);
        assert_eq!(params.source_image_path, None);
    }

    // ── 3-D mesh family (Hunyuan3D) ─────────────────────────────────────

    /// The family normalizer pins GLB and clears every raster-only input a
    /// mesh recipe refuses, while keeping the source image — the family's
    /// only conditioning.
    #[test]
    fn mesh_family_normalizer_pins_glb_clears_mask_and_keeps_the_source() {
        let mut params = GenerateParams::from_config(&Config::default());
        params.model = mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL.into();
        params.format = OutputFormat::Png;
        params.source_image_path = Some("chair.png".into());
        params.mask_image_path = Some("stale-mask.png".into());
        params.control_image_path = Some("stale-control.png".into());
        params.control_model = Some("canny".into());
        params.lora_path = Some("stale.safetensors".into());
        params.mesh.octree_resolution = Some(320);

        normalize_generate_params_for_family(&mut params, mold_core::manifest::HUNYUAN3D_FAMILY);

        assert_eq!(params.format, OutputFormat::Glb);
        assert_eq!(params.source_image_path.as_deref(), Some("chair.png"));
        assert_eq!(params.mask_image_path, None);
        assert_eq!(params.control_image_path, None);
        assert_eq!(params.control_model, None);
        assert_eq!(params.lora_path, None);
        assert_eq!(
            params.mesh.octree_resolution,
            Some(320),
            "the mesh knobs are the family's own and survive"
        );

        // A raster family is untouched by the mesh arm.
        let mut raster = GenerateParams::from_config(&Config::default());
        raster.format = OutputFormat::Jpeg;
        raster.mask_image_path = Some("mask.png".into());
        normalize_generate_params_for_family(&mut raster, "sd15");
        assert_eq!(raster.format, OutputFormat::Jpeg);
        assert_eq!(raster.mask_image_path.as_deref(), Some("mask.png"));
    }

    /// The ◀▶ helpers walk the PROFILE's bounds and never leave them.
    #[test]
    fn mesh_row_adjusters_stay_inside_the_profile_bounds() {
        let allowed = mold_core::validation::MESH_OCTREE_RESOLUTIONS;
        let default = mold_core::validation::MESH_DEFAULT_OCTREE_RESOLUTION;
        // Untouched starts from the default and steps along the allowlist.
        assert_eq!(next_octree_resolution(allowed, default, None, 1), Some(320));
        assert_eq!(
            next_octree_resolution(allowed, default, None, -1),
            Some(192)
        );
        assert_eq!(
            next_octree_resolution(allowed, default, Some(384), 1),
            Some(384),
            "the top rung is a wall, not a wrap"
        );
        assert_eq!(
            next_octree_resolution(allowed, default, Some(128), -1),
            Some(128)
        );
        // A value off the list (older session) re-anchors at the first rung.
        assert_eq!(
            next_octree_resolution(allowed, default, Some(200), 1),
            Some(192)
        );

        let control = mold_core::FloatControl {
            default: mold_core::validation::MESH_DEFAULT_THRESHOLD,
            min: 0.0,
            max: 1.0,
            step: mold_core::validation::MESH_THRESHOLD_STEP,
            mode: mold_core::ControlMode::Adjustable,
            note: None,
        };
        assert!((next_mesh_threshold(&control, None, -1) - 0.55).abs() < 1e-6);
        assert!((next_mesh_threshold(&control, Some(0.98), 1) - 1.0).abs() < 1e-6);
        assert!((next_mesh_threshold(&control, Some(0.02), -1)).abs() < 1e-6);

        let (min, max) = (
            mold_core::validation::MESH_MIN_TARGET_FACES,
            mold_core::validation::MESH_MAX_TARGET_FACES,
        );
        assert_eq!(next_target_faces(min, max, None, 1), Some(min));
        assert_eq!(next_target_faces(min, max, None, -1), None);
        assert_eq!(
            next_target_faces(min, max, Some(min), -1),
            None,
            "back to off"
        );
        assert_eq!(
            next_target_faces(min, max, Some(min), 1),
            Some(min + 10_000)
        );
        assert_eq!(next_target_faces(min, max, Some(max), 1), Some(max));
        assert_eq!(next_target_faces(min, max, Some(15_000), -1), Some(5_000));
    }

    /// With the recipe's profile loaded, the Format row cannot leave GLB,
    /// Reset to model defaults lands on GLB, and the mesh knobs go back to
    /// the recipe's defaults. The GLB pin comes from the family normalizer,
    /// the rows from the profile block.
    #[tokio::test]
    async fn mesh_recipe_pins_the_format_row_and_reset_keeps_it() {
        let mut app = make_settings_test_app();
        app.models.catalog = mold_core::build_model_catalog(&app.config, None, false);
        app.generate.params.model = mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL.to_string();
        app.sync_generate_capabilities();

        assert!(
            app.generate.capabilities.mesh.is_some(),
            "the built-in profile carries the mesh block"
        );
        assert!(!app.generate.capabilities.supports_strength);
        assert!(!app.generate.capabilities.supports_mask);
        assert!(!app.generate.capabilities.supports_negative_prompt);
        assert_eq!(app.generate.params.format, OutputFormat::Glb);
        assert!(app
            .generate
            .rows
            .contains(&crate::ui::create_form::CreateRow::AdvancedHeader));
        app.generate.advanced.open = true;
        app.refresh_create_rows();
        assert!(app
            .generate
            .rows
            .contains(&crate::ui::create_form::CreateRow::Section(
                crate::ui::create_form::AdvSection::Mesh
            )));

        for _ in 0..8 {
            app.adjust_field(ParamField::Format, 1);
            assert_eq!(app.generate.params.format, OutputFormat::Glb);
        }
        app.adjust_field(ParamField::Format, -1);
        assert_eq!(app.generate.params.format, OutputFormat::Glb);

        app.adjust_field(ParamField::Octree, 1);
        assert_eq!(app.generate.params.mesh.octree_resolution, Some(320));
        app.adjust_field(ParamField::MeshThreshold, -1);
        assert_eq!(app.generate.params.mesh.threshold, Some(0.55));
        app.adjust_field(ParamField::TargetFaces, 1);
        assert_eq!(
            app.generate.params.mesh.target_faces,
            Some(mold_core::validation::MESH_MIN_TARGET_FACES)
        );

        app.reset_params_to_model_defaults();
        assert_eq!(app.generate.params.format, OutputFormat::Glb);
        assert_eq!(
            app.generate.params.mesh,
            mold_core::MeshRequestOptions::default()
        );

        // Switching to a raster recipe drops the block and un-pins the row.
        app.generate.params.model = "flux2-klein:q8".to_string();
        app.generate.params.mesh.octree_resolution = Some(384);
        app.sync_generate_capabilities();
        assert!(app.generate.capabilities.mesh.is_none());
        assert_eq!(
            app.generate.params.mesh,
            mold_core::MeshRequestOptions::default(),
            "a raster recipe refuses the block, so it is cleared"
        );
    }

    /// A recipe whose profile says `prompt.mode == ignored` admits an empty
    /// prompt at Generate. The refusal, when any, must be about something
    /// else (here: the missing source image).
    #[tokio::test]
    async fn mesh_recipe_admits_an_empty_prompt() {
        let mut app = make_settings_test_app();
        app.models.catalog = mold_core::build_model_catalog(&app.config, None, false);
        app.generate.params.model = mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL.to_string();
        app.sync_generate_capabilities();
        app.generate.prompt = TextArea::default();

        assert!(!app.prompt_required_now());
        app.start_generation();
        assert_ne!(
            app.generate.error_message.as_deref(),
            Some("Prompt is empty"),
            "{:?}",
            app.generate.error_message
        );

        // The same gate still refuses an empty prompt on a text model.
        app.generate.params.model = "flux2-klein:q8".to_string();
        app.sync_generate_capabilities();
        assert!(app.prompt_required_now());
        app.start_generation();
        assert_eq!(
            app.generate.error_message.as_deref(),
            Some("Prompt is empty")
        );
    }

    fn tiny_png(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |_, _| image::Rgb([200, 120, 40]));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }

    /// A finished mesh saves its `.glb`, caches the poster under the same
    /// thumbnail key a gallery scan looks up, shows the poster in the
    /// Preview, captions with the statistics, and files a GLB gallery row.
    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn generation_complete_saves_the_glb_and_its_poster() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.active_view = View::Create;
            app.generate.generating = true;
            app.generate.batch_remaining = 1;
            app.generate.params.model = mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL.to_string();
            app.generate.params.format = OutputFormat::Glb;
            app.generate.params.source_image_path = Some("chair.png".into());

            let poster = tiny_png(32, 24);
            let response = GenerateResponse {
                request_warnings: Vec::new(),
                audio: None,
                images: vec![],
                video: None,
                mesh: Some(mold_core::MeshData {
                    data: b"glTF-bytes".to_vec(),
                    format: OutputFormat::Glb,
                    vertex_count: 24_576,
                    face_count: 49_152,
                    bounds_min: [-0.5, -0.4, -0.3],
                    bounds_max: [0.5, 0.4, 0.3],
                    textured: false,
                    poster: poster.clone(),
                    poster_width: 32,
                    poster_height: 24,
                }),
                generation_time_ms: 4_000,
                model: mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL.to_string(),
                seed_used: 9,
                gpu: None,
            };
            let metadata_snapshot = generation_metadata_snapshot(&app);
            app.bg_tx
                .send(BackgroundEvent::GenerationComplete {
                    response: Box::new(response),
                    from_local: true,
                    metadata_snapshot,
                })
                .unwrap();
            app.process_background_events();

            let saved = app
                .generate
                .last_output_path
                .clone()
                .expect("a local mesh render is saved");
            assert_eq!(saved.extension().and_then(|e| e.to_str()), Some("glb"));
            assert_eq!(std::fs::read(&saved).unwrap(), b"glTF-bytes");
            let cached_poster = crate::thumbnails::thumbnail_path(&saved);
            assert_eq!(
                std::fs::read(&cached_poster).unwrap(),
                poster,
                "the poster is cached under the gallery's thumbnail key"
            );
            assert!(
                app.generate.preview_image.is_some(),
                "the poster is the preview"
            );
            assert_eq!(
                app.generate.last_mesh_summary.as_deref(),
                Some("49,152 tris \u{00b7} 24,576 verts \u{00b7} 1.00\u{00d7}0.80\u{00d7}0.60")
            );
            assert_eq!(app.gallery.entries.len(), 1);
            let meta = &app.gallery.entries[0].metadata;
            assert_eq!(meta.output_format, Some(OutputFormat::Glb));
            assert_eq!(
                (meta.width, meta.height),
                (32, 24),
                "poster size, as the server records"
            );
            assert!(
                app.generate
                    .progress
                    .log
                    .iter()
                    .any(|entry| entry.message.starts_with("Saved ")
                        && entry.message.contains(".glb"))
            );

            // The Library detail pane shows the cached poster and never opens
            // the `.glb` itself; without a poster it stays empty.
            app.gallery.selected = 0;
            app.load_gallery_preview();
            assert!(app.gallery.preview_image.is_some());
            std::fs::remove_file(&cached_poster).unwrap();
            app.load_gallery_preview();
            assert!(app.gallery.preview_image.is_none());
        });
    }

    /// The export picker lists what the host advertises minus the stored
    /// form, the target is named after the print, and a local transcode
    /// goes through the same writer the server's export route uses.
    #[test]
    fn mesh_export_helpers_name_the_target_and_transcode_locally() {
        use mold_core::MeshExportFormat::{Glb, Obj, Ply, Stl};
        assert_eq!(mesh_export_formats_for(None), vec![Obj, Stl, Ply]);
        assert_eq!(
            mesh_export_formats_for(Some(&[Glb, Obj, Stl, Ply])),
            vec![Obj, Stl, Ply],
            "the stored GLB is never an export"
        );
        assert_eq!(mesh_export_formats_for(Some(&[Glb])), Vec::<_>::new());

        let target =
            mesh_export_target_path(std::path::Path::new("/out"), "mold-hunyuan3d-1.glb", Stl);
        assert_eq!(
            target,
            std::path::PathBuf::from("/out/mold-hunyuan3d-1.stl")
        );

        let mesh = mold_inference::hunyuan3d::mesh::Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
            normals: None,
            uvs: None,
            vertex_colors: None,
        };
        let glb = mold_inference::hunyuan3d::glb::write_glb(
            &mesh,
            &mold_inference::hunyuan3d::glb::GlbMaterial::default(),
            None,
        )
        .unwrap();
        let stl = export_local_mesh(&glb, Stl).unwrap();
        assert_eq!(stl.len(), 80 + 4 + 50, "one binary STL triangle");
        let obj = String::from_utf8(export_local_mesh(&glb, Obj).unwrap()).unwrap();
        assert!(obj.contains("v 1"), "{obj}");
        assert!(obj.contains("f 1 2 3") || obj.contains("f 1/"), "{obj}");
        assert!(!export_local_mesh(&glb, Ply).unwrap().is_empty());
        assert_eq!(export_local_mesh(&glb, Glb).unwrap(), glb);
        assert!(export_local_mesh(b"not a glb", Stl).is_err());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn library_x_opens_the_export_picker_only_for_a_mesh() {
        let mut app = make_settings_test_app();
        app.active_view = View::Library;
        // Only the path matters for this gate; the metadata literal is
        // large, so it is deserialized from its required fields.
        let entry = |name: &str| GalleryEntry {
            path: std::path::PathBuf::from(name),
            metadata: serde_json::from_value(serde_json::json!({
                "prompt": "",
                "model": mold_core::manifest::HUNYUAN3D_DEFAULT_MODEL,
                "seed": 1,
                "steps": 5,
                "guidance": 5.0,
                "width": 512,
                "height": 512,
                "version": "test"
            }))
            .expect("metadata deserializes with defaults"),
            generation_time_ms: None,
            timestamp: 0,
            server_url: None,
            title: None,
            origins: vec![GalleryOrigin::local()],
        };
        app.gallery.entries = vec![entry("chair.glb"), entry("still.png")];
        app.gallery.thumbnail_states = vec![None, None];
        app.gallery.thumb_dimensions = vec![None, None];
        app.gallery.thumb_fixed_cache = vec![None, None];
        app.gallery.refresh_filter();

        app.gallery.selected = 0;
        app.dispatch_action(Action::ExportMesh);
        match &app.popup {
            Some(Popup::MeshExportPicker {
                filename, formats, ..
            }) => {
                assert_eq!(filename, "chair.glb");
                assert_eq!(formats, &LOCAL_MESH_EXPORT_FORMATS.to_vec());
            }
            other => panic!("expected the export picker, got {}", other.is_some()),
        }
        app.popup = None;

        app.gallery.selected = 1;
        app.dispatch_action(Action::ExportMesh);
        assert!(app.popup.is_none(), "a still has nothing to export");
        assert!(app
            .generate
            .error_message
            .as_deref()
            .is_some_and(|m| m.contains(".glb")));
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn h3_switch_away_and_back_cannot_restore_invalid_preferences() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.config.models.insert(
                mold_core::minimax_h3::FL2VA_COMFY.into(),
                mold_core::ModelConfig {
                    family: Some("minimax_h3".into()),
                    default_frames: Some(25),
                    default_fps: Some(30),
                    ..Default::default()
                },
            );
            app.config.models.insert(
                "flux-dev:q4".into(),
                mold_core::ModelConfig {
                    family: Some("flux".into()),
                    ..Default::default()
                },
            );

            app.update_model(mold_core::minimax_h3::FL2VA_COMFY);
            app.generate.params.format = OutputFormat::Png;
            app.generate.params.guidance = 7.5;
            app.generate.params.strength = 0.25;
            app.generate.params.scheduler = Some(Scheduler::Ddim);
            app.generate.params.lora_path = Some("stale.safetensors".into());
            app.generate.params.source_image_path = Some("stale.png".into());
            app.update_model("flux-dev:q4");
            app.update_model(mold_core::minimax_h3::FL2VA_COMFY);

            // The remembered 25 is off the grid, so it repairs to the
            // nearest valid clip length rather than to one pinned count.
            assert_eq!(
                app.generate.params.frames,
                mold_core::minimax_h3::MIN_FRAMES
            );
            assert_eq!(app.generate.params.fps, mold_core::minimax_h3::FIXED_FPS);
            assert_eq!(app.generate.params.format, OutputFormat::Mp4);
            assert_eq!(app.generate.params.enable_audio, Some(true));
            assert_eq!(app.generate.params.guidance, 0.0);
            assert_eq!(app.generate.params.strength, 1.0);
            assert_eq!(app.generate.params.scheduler, None);
            assert_eq!(app.generate.params.lora_path, None);
            assert_eq!(app.generate.params.source_image_path, None);
        });
    }

    #[tokio::test]
    async fn changing_away_from_mp4_clears_audio_override() {
        let mut app = make_settings_test_app();
        app.generate.params.format = OutputFormat::Mp4;
        app.generate.params.enable_audio = Some(true);

        app.adjust_field(ParamField::Format, 1);

        assert_ne!(app.generate.params.format, OutputFormat::Mp4);
        assert_eq!(
            app.generate.params.enable_audio, None,
            "a non-MP4 container cannot retain explicit audio output authority"
        );
    }

    #[tokio::test]
    async fn pipeline_row_cycles_source_free_ltx2_recipes_and_selects_mp4() {
        use crate::ui::create_form::{AdvSection, CreateRow};
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        app.generate.params.model = "ltx-2.3-22b-dev:fp8".into();
        app.config.models.insert(
            app.generate.params.model.clone(),
            mold_core::ModelConfig {
                family: Some("ltx2".into()),
                ..Default::default()
            },
        );
        app.sync_generate_capabilities();
        app.generate.advanced.open = true;
        app.generate.advanced.expanded = Some(AdvSection::Video);
        app.refresh_create_rows();
        let pipeline_idx = app
            .generate
            .rows
            .iter()
            .position(|row| {
                *row == CreateRow::SectionField(AdvSection::Video, ParamField::Pipeline)
            })
            .expect("LTX-2 Video section must expose Pipeline");
        app.generate.param_index = pipeline_idx;

        assert_eq!(app.generate.params.pipeline, None);
        app.increment_param(1);
        assert_eq!(
            app.generate.params.pipeline,
            Some(Ltx2PipelineMode::OneStage)
        );
        assert_eq!(app.generate.params.format, OutputFormat::Mp4);
        assert!(
            app.generate.capabilities.supports_negative_prompt,
            "the one-stage recipe uses CFG even when Auto inherited a distilled checkpoint default"
        );
        app.dispatch_action(Action::Confirm);
        assert_eq!(
            app.generate.params.pipeline,
            Some(Ltx2PipelineMode::TwoStage)
        );
        app.increment_param(-1);
        assert_eq!(
            app.generate.params.pipeline,
            Some(Ltx2PipelineMode::OneStage)
        );
        app.increment_param(-1);
        assert_eq!(app.generate.params.pipeline, None);
        app.increment_param(-1);
        assert_eq!(
            app.generate.params.pipeline,
            Some(Ltx2PipelineMode::Distilled)
        );
    }

    #[tokio::test]
    async fn explicit_distilled_pipeline_disables_primary_guidance() {
        let mut app = make_settings_test_app();
        app.generate.params.model = "ltx-2.3-22b-dev:fp8".into();
        app.config.models.insert(
            app.generate.params.model.clone(),
            mold_core::ModelConfig {
                family: Some("ltx2".into()),
                ..Default::default()
            },
        );
        app.generate.capabilities = crate::model_info::capabilities_for_model(
            "ltx2",
            &app.generate.params.model,
            None,
            None,
            None,
            None,
        );
        app.generate.params.pipeline = Some(Ltx2PipelineMode::Distilled);
        app.generate.params.guidance = 7.0;
        app.sync_pipeline_guidance();

        app.adjust_field(ParamField::Guidance, 1);

        assert_eq!(app.generate.params.guidance, 7.0);
        let mut submitted = app.generate.params.clone();
        app.normalize_fixed_guidance_for_submit(&mut submitted);
        assert_eq!(submitted.guidance, 1.0);
        assert!(!app.generate.guidance_adjustable());
        assert!(!app.generate.capabilities.supports_negative_prompt);
    }

    #[tokio::test]
    async fn returning_pipeline_to_auto_restores_advertised_catalog_guidance() {
        let mut app = make_settings_test_app();
        let model = "cv:opaque-ltx2-checkpoint";
        app.generate.params.model = model.into();
        app.config.models.insert(
            model.into(),
            mold_core::ModelConfig {
                family: Some("ltx2".into()),
                ..Default::default()
            },
        );
        let mut entry = make_test_catalog_entry(model, 8, 1.0, 1216, 704, "Opaque LTX-2");
        entry.info.family = "ltx2".into();
        entry.guidance_capabilities = Some(mold_core::GuidanceCapabilities::FIXED_ONE);
        app.models.catalog.push(entry);

        app.sync_generate_capabilities();
        assert!(!app.generate.capabilities.supports_negative_prompt);

        app.generate.params.pipeline = Some(Ltx2PipelineMode::OneStage);
        app.sync_pipeline_guidance();
        assert!(app.generate.capabilities.supports_negative_prompt);

        app.generate.params.pipeline = None;
        app.sync_pipeline_guidance();
        assert!(
            !app.generate.capabilities.supports_negative_prompt,
            "Auto must restore the server-advertised fixed guidance contract"
        );
    }

    // ── identity conditioning (PuLID-FLUX, #1231) ───────────────

    /// The Identity rows follow the server's advertisement and nothing else.
    #[tokio::test]
    async fn identity_rows_follow_the_advertised_capability() {
        use crate::ui::create_form::{AdvSection, CreateRow};

        let mut app = make_settings_test_app();
        let model = "flux-dev:q8";
        app.generate.params.model = model.into();
        let mut entry = make_test_catalog_entry(model, 20, 3.5, 1024, 1024, "FLUX dev");
        entry.supports_identity = Some(true);
        app.models.catalog.push(entry);
        app.generate.advanced = crate::ui::create_form::AdvancedState {
            open: true,
            expanded: Some(AdvSection::Identity),
        };

        app.sync_generate_capabilities();
        assert!(app.generate.capabilities.supports_identity);
        assert!(app
            .generate
            .rows
            .contains(&CreateRow::Section(AdvSection::Identity)));
        assert!(app.generate.rows.contains(&CreateRow::SectionField(
            AdvSection::Identity,
            ParamField::IdentityImage
        )));

        // The same model against a server that does not advertise the field.
        app.models.catalog[0].supports_identity = None;
        app.sync_generate_capabilities();
        assert!(!app.generate.capabilities.supports_identity);
        assert!(!app
            .generate
            .rows
            .contains(&CreateRow::Section(AdvSection::Identity)));
    }

    /// Switching to a checkpoint that cannot take the photo keeps the photo
    /// (the user picked it, and switching back must not have lost it), raises
    /// `mold_core`'s own refusal, and blocks dispatch.
    #[tokio::test]
    async fn an_unqualified_model_keeps_the_photo_and_refuses_it() {
        let mut app = make_settings_test_app();
        let qualified = "flux-dev:q8";
        let mut entry = make_test_catalog_entry(qualified, 20, 3.5, 1024, 1024, "FLUX dev");
        entry.supports_identity = Some(true);
        app.models.catalog.push(entry);
        app.models.catalog.push(make_test_catalog_entry(
            "flux2-klein:q8",
            4,
            0.0,
            1024,
            1024,
            "Klein",
        ));

        app.generate.params.model = qualified.into();
        app.generate.params.identity_image_path = Some("/photos/ada.png".into());
        app.sync_generate_capabilities();
        assert_eq!(app.generate.identity_error, None);

        app.generate.params.model = "flux2-klein:q8".into();
        app.sync_generate_capabilities();
        assert_eq!(
            app.generate.params.identity_image_path.as_deref(),
            Some("/photos/ada.png"),
            "a model switch must never silently drop the face"
        );
        assert_eq!(
            app.generate.identity_error.as_deref(),
            Some(mold_core::identity::identity_model_gate_message("flux2-klein:q8").as_str()),
            "the refusal is mold-core's sentence, not a restatement"
        );

        // Reset is the way back out; it clears the photo and both knobs.
        app.generate.params.id_weight = 2.0;
        app.generate.params.id_start_step = 1;
        app.reset_params_to_model_defaults();
        assert_eq!(app.generate.params.identity_image_path, None);
        assert_eq!(app.generate.identity_error, None);
        assert_eq!(
            app.generate.params.id_weight,
            mold_core::identity::ID_WEIGHT_DEFAULT
        );
        assert_eq!(
            app.generate.params.id_start_step,
            mold_core::identity::ID_START_STEP_DEFAULT
        );
    }

    /// Both knobs are bounded by `mold_core::identity`, so the form can never
    /// express a value admission would refuse.
    #[tokio::test]
    async fn identity_knobs_clamp_to_the_core_bounds() {
        let mut app = make_settings_test_app();
        app.generate.params.steps = 20;

        app.adjust_field(ParamField::IdentityWeight, 1);
        assert_eq!(app.generate.params.id_weight, 1.1);
        for _ in 0..64 {
            app.adjust_field(ParamField::IdentityWeight, 1);
        }
        assert_eq!(
            app.generate.params.id_weight,
            mold_core::identity::ID_WEIGHT_MAX
        );
        for _ in 0..128 {
            app.adjust_field(ParamField::IdentityWeight, -1);
        }
        assert_eq!(app.generate.params.id_weight, 0.0);
        assert!(mold_core::identity::validate_id_weight(app.generate.params.id_weight).is_ok());

        app.adjust_field(ParamField::IdentityStartStep, -1);
        assert_eq!(app.generate.params.id_start_step, 0);
        for _ in 0..64 {
            app.adjust_field(ParamField::IdentityStartStep, 1);
        }
        assert_eq!(
            app.generate.params.id_start_step, 19,
            "the ceiling is steps - 1, so `id_start_step < steps` always holds"
        );
        assert!(mold_core::identity::validate_id_start_step(
            app.generate.params.id_start_step,
            app.generate.params.steps
        )
        .is_ok());
    }

    /// A restored start step that a lower-step model cannot honour is pulled
    /// back onto the grid rather than left to fail at admission.
    #[tokio::test]
    async fn a_model_switch_pulls_the_start_step_below_the_new_step_count() {
        let mut app = make_settings_test_app();
        app.generate.params.steps = 20;
        app.generate.params.id_start_step = 15;
        app.sync_generate_capabilities();
        assert_eq!(app.generate.params.id_start_step, 15);

        app.generate.params.steps = 4;
        app.sync_generate_capabilities();
        assert_eq!(app.generate.params.id_start_step, 3);
        assert!(mold_core::identity::validate_id_start_step(
            app.generate.params.id_start_step,
            app.generate.params.steps
        )
        .is_ok());
    }

    /// Milestone 1 qualifies neither pairing, and the Create form can hold
    /// both at once — LoRA and Source are their own Advanced sections — so
    /// the refusal has to be inline rather than a round trip away. The
    /// wording is `mold_core::identity`'s const, not a restatement.
    #[tokio::test]
    async fn identity_refuses_the_lora_and_img2img_pairings_inline() {
        let mut app = make_settings_test_app();
        let model = "flux-dev:q8";
        let mut entry = make_test_catalog_entry(model, 20, 3.5, 1024, 1024, "FLUX dev");
        entry.supports_identity = Some(true);
        app.models.catalog.push(entry);
        app.generate.params.model = model.into();
        app.generate.params.identity_image_path = Some("/photos/ada.png".into());
        app.sync_generate_capabilities();
        assert_eq!(app.generate.identity_error, None);

        app.generate.params.lora_path = Some("/loras/pixel.safetensors".into());
        app.sync_generate_capabilities();
        assert_eq!(
            app.generate.identity_error.as_deref(),
            Some(mold_core::identity::IDENTITY_LORA_CONFLICT)
        );

        app.generate.params.lora_path = None;
        app.generate.params.source_image_path = Some("/photos/scene.png".into());
        app.sync_generate_capabilities();
        assert_eq!(
            app.generate.identity_error.as_deref(),
            Some(mold_core::identity::IDENTITY_IMG2IMG_CONFLICT)
        );

        // Removing the conflict clears the refusal rather than leaving it
        // stale on the row.
        app.generate.params.source_image_path = None;
        app.sync_generate_capabilities();
        assert_eq!(app.generate.identity_error, None);

        // Neither pairing is a problem without a photo.
        app.generate.params.identity_image_path = None;
        app.generate.params.lora_path = Some("/loras/pixel.safetensors".into());
        app.sync_generate_capabilities();
        assert_eq!(app.generate.identity_error, None);
    }

    /// A restored print whose start step is at or past the new step count is
    /// repaired first and validated second, so the user is never left staring
    /// at a refusal for a state the repair already fixed.
    #[tokio::test]
    async fn a_restored_out_of_range_start_step_is_clamped_before_it_is_judged() {
        let mut app = make_settings_test_app();
        let model = "flux-dev:q8";
        let mut entry = make_test_catalog_entry(model, 20, 3.5, 1024, 1024, "FLUX dev");
        entry.supports_identity = Some(true);
        app.models.catalog.push(entry);
        app.generate.params.model = model.into();
        app.generate.params.identity_image_path = Some("/photos/ada.png".into());

        // Restored state: start step at the step count, which admission
        // refuses (the rule is strictly less than).
        app.generate.params.steps = 4;
        app.generate.params.id_start_step = 20;
        app.sync_generate_capabilities();
        assert_eq!(app.generate.params.id_start_step, 3);
        assert_eq!(
            app.generate.identity_error, None,
            "the clamp runs first, so no refusal survives for a repaired value"
        );
        assert!(mold_core::identity::validate_id_start_step(
            app.generate.params.id_start_step,
            app.generate.params.steps
        )
        .is_ok());

        // Exactly at the boundary is the same repair.
        app.generate.params.id_start_step = 4;
        app.sync_generate_capabilities();
        assert_eq!(app.generate.params.id_start_step, 3);
        assert_eq!(app.generate.identity_error, None);
    }

    /// A photo accepted at entry can be deleted or replaced before Generate
    /// is pressed. Dispatch re-reads it so the refusal lands on the Photo row
    /// instead of the run silently becoming an ordinary render.
    #[tokio::test]
    async fn dispatch_rechecks_the_photo_file_the_gate_does_not_touch() {
        let mut app = make_settings_test_app();
        let model = "flux-dev:q8";
        let mut entry = make_test_catalog_entry(model, 20, 3.5, 1024, 1024, "FLUX dev");
        entry.supports_identity = Some(true);
        app.models.catalog.push(entry);
        app.generate.params.model = model.into();
        app.generate.params.steps = 20;

        let dir = tempfile::tempdir().unwrap();
        let photo = dir.path().join("ada.png");
        std::fs::write(&photo, IDENTITY_TEST_PNG).unwrap();
        app.generate.params.identity_image_path = Some(photo.to_string_lossy().to_string());
        app.sync_generate_capabilities();
        assert_eq!(app.identity_dispatch_error(), None);

        // The cheap gate deliberately does no file I/O — it runs on every
        // model switch — so only the dispatch check notices the file is gone.
        std::fs::remove_file(&photo).unwrap();
        assert_eq!(app.identity_gate_error(), None);
        let error = app
            .identity_dispatch_error()
            .expect("a vanished photo must refuse dispatch");
        assert!(
            error.starts_with("Identity photo could not be opened"),
            "{error}"
        );

        // The model gate still outranks the file check: a request that could
        // not run anyway names the reason it could not run.
        std::fs::write(&photo, IDENTITY_TEST_PNG).unwrap();
        app.models.catalog[0].supports_identity = None;
        app.sync_generate_capabilities();
        assert_eq!(
            app.identity_dispatch_error(),
            Some(mold_core::identity::identity_model_gate_message(model))
        );
    }

    /// A genuine 1x1 RGBA PNG — the smallest payload
    /// `identity::validate_id_image_bytes` accepts.
    const IDENTITY_TEST_PNG: [u8; 67] = [
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00,
        0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    /// With no photo attached there is nothing to gate: the knobs alone are
    /// never a reason to refuse a perfectly ordinary render.
    #[tokio::test]
    async fn knobs_without_a_photo_never_gate_a_render() {
        let mut app = make_settings_test_app();
        app.generate.params.id_weight = 2.5;
        app.generate.params.id_start_step = 2;
        app.sync_generate_capabilities();
        assert_eq!(app.generate.identity_error, None);
    }

    #[tokio::test]
    async fn ltx2_upscale_rows_cycle_native_modes() {
        use crate::ui::create_form::{AdvSection, CreateRow};
        let mut app = make_settings_test_app();
        app.active_view = View::Create;
        app.generate.focus = GenerateFocus::Parameters;
        app.generate.capabilities = crate::model_info::capabilities_for_family("ltx2");
        app.generate.advanced.open = true;
        app.generate.advanced.expanded = Some(AdvSection::Video);
        app.refresh_create_rows();

        let spatial_idx = app
            .generate
            .rows
            .iter()
            .position(|row| {
                *row == CreateRow::SectionField(AdvSection::Video, ParamField::SpatialUpscale)
            })
            .expect("LTX-2 Video section must expose spatial upscale");
        app.generate.param_index = spatial_idx;
        app.increment_param(1);
        assert_eq!(
            app.generate.params.spatial_upscale,
            Some(Ltx2SpatialUpscale::X1_5)
        );
        app.dispatch_action(Action::Confirm);
        assert_eq!(
            app.generate.params.spatial_upscale,
            Some(Ltx2SpatialUpscale::X2)
        );
        app.increment_param(1);
        assert_eq!(app.generate.params.spatial_upscale, None);

        let temporal_idx = app
            .generate
            .rows
            .iter()
            .position(|row| {
                *row == CreateRow::SectionField(AdvSection::Video, ParamField::TemporalUpscale)
            })
            .expect("LTX-2 Video section must expose temporal upscale");
        app.generate.param_index = temporal_idx;
        app.increment_param(1);
        assert_eq!(
            app.generate.params.temporal_upscale,
            Some(Ltx2TemporalUpscale::X2)
        );
        app.increment_param(-1);
        assert_eq!(app.generate.params.temporal_upscale, None);
    }

    #[tokio::test]
    async fn ltx2_guidance_rows_cycle_optional_bounded_values() {
        let mut app = make_settings_test_app();

        app.adjust_field(ParamField::StgScale, 1);
        assert_eq!(app.generate.params.guidance_overrides.stg_scale, Some(0.0));
        app.adjust_field(ParamField::StgScale, 1);
        assert_eq!(app.generate.params.guidance_overrides.stg_scale, Some(0.5));

        app.adjust_field(ParamField::RescaleScale, -1);
        assert_eq!(
            app.generate.params.guidance_overrides.rescale_scale,
            Some(1.0)
        );
        app.adjust_field(ParamField::RescaleScale, 1);
        assert_eq!(app.generate.params.guidance_overrides.rescale_scale, None);

        app.adjust_field(ParamField::ModalityScale, -1);
        assert_eq!(
            app.generate.params.guidance_overrides.modality_scale,
            Some(mold_core::Ltx2GuidanceOverrides::MAX_SCALE)
        );
        app.adjust_field(ParamField::GuidanceSkip, -1);
        assert_eq!(
            app.generate.params.guidance_overrides.skip_step,
            Some(mold_core::Ltx2GuidanceOverrides::MAX_SKIP_STEP)
        );
    }

    #[tokio::test]
    async fn distilled_checkpoint_guidance_cannot_be_adjusted() {
        let mut app = make_settings_test_app();
        app.generate.params.model = "ltx-2.3-22b-distilled:fp8".into();
        app.config.models.insert(
            app.generate.params.model.clone(),
            mold_core::ModelConfig {
                family: Some("ltx2".into()),
                ..Default::default()
            },
        );
        app.generate.capabilities = crate::model_info::capabilities_for_model(
            "ltx2",
            &app.generate.params.model,
            None,
            None,
            None,
            None,
        );
        app.generate.params.guidance = 7.0;
        app.sync_generate_capabilities();
        app.adjust_field(ParamField::Guidance, 1);
        assert_eq!(app.generate.params.guidance, 7.0);
        let mut submitted = app.generate.params.clone();
        app.normalize_fixed_guidance_for_submit(&mut submitted);
        assert_eq!(submitted.guidance, 1.0);

        app.generate.params.model = "ltx-2.3-22b-dev:fp8".into();
        app.generate.capabilities = crate::model_info::capabilities_for_model(
            "ltx2",
            &app.generate.params.model,
            None,
            None,
            None,
            None,
        );
        app.adjust_field(ParamField::Guidance, 1);
        assert_eq!(app.generate.params.guidance, 7.5);
    }

    #[test]
    fn stg_block_input_rejects_invalid_lists_before_request_building() {
        assert_eq!(parse_stg_blocks_input(""), Ok(None));
        assert_eq!(parse_stg_blocks_input("28, 29"), Ok(Some(vec![28, 29])));
        assert!(parse_stg_blocks_input(" , ").is_err());
        assert!(parse_stg_blocks_input("28, nope").is_err());
        assert!(parse_stg_blocks_input("28, 28").is_err());
        assert!(parse_stg_blocks_input("64").is_err());
        assert!(parse_stg_blocks_input("0,1,2,3,4,5,6,7,8").is_err());
    }

    #[tokio::test]
    async fn stg_block_popup_keeps_invalid_input_and_applies_valid_input() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let mut app = make_settings_test_app();
        app.popup = Some(Popup::StgBlocksInput {
            input: "28, 28".into(),
            error: None,
        });

        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        let Some(Popup::StgBlocksInput { input, error }) = &mut app.popup else {
            panic!("invalid input must keep the block editor open");
        };
        assert_eq!(input, "28, 28");
        assert!(error
            .as_deref()
            .is_some_and(|message| message.contains("twice")));
        input.clear();
        input.push_str("28, 29");

        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        assert!(app.popup.is_none());
        assert_eq!(
            app.generate.params.guidance_overrides.stg_blocks,
            Some(vec![28, 29])
        );
    }

    // ── request advisories (x-mold-request-warning) ────────────

    /// Build a completed response carrying `request_warnings`.
    fn response_with_advisories(warnings: Vec<String>) -> GenerateResponse {
        GenerateResponse {
            mesh: None,
            request_warnings: warnings,
            audio: None,
            images: vec![mold_core::ImageData {
                data: vec![0u8; 4],
                format: OutputFormat::Png,
                width: 64,
                height: 64,
                index: 0,
            }],
            generation_time_ms: 100,
            model: "flux-dev:q4".to_string(),
            seed_used: 42,
            video: None,
            gpu: None,
        }
    }

    fn deliver_completion(app: &mut App, response: GenerateResponse) {
        let metadata_snapshot = generation_metadata_snapshot(app);
        app.bg_tx
            .send(BackgroundEvent::GenerationComplete {
                response: Box::new(response),
                from_local: false,
                metadata_snapshot,
            })
            .unwrap();
        app.process_background_events();
    }

    /// The host accepted the request and produced a print; something was
    /// dropped or adjusted along the way. That is an advisory, not a failure,
    /// so it takes the advisory slot and leaves the error slot alone.
    #[tokio::test]
    async fn a_completed_generation_surfaces_its_advisories_without_claiming_failure() {
        let mut app = make_settings_test_app();
        app.generate.prompt = TextArea::from(["a test prompt"]);
        deliver_completion(
            &mut app,
            response_with_advisories(vec![
                "tags and collection were not applied; the print was generated and saved normally"
                    .to_string(),
            ]),
        );

        let warning = app
            .generate
            .warning_message
            .as_deref()
            .expect("a dropped filing must be reported");
        assert!(
            warning.contains("tags and collection were not applied"),
            "{warning}"
        );
        assert_eq!(
            app.generate.error_message, None,
            "nothing failed \u{2014} the error slot renders a \u{2717} and would read as a failed render"
        );
    }

    /// The advisory prose contains "; " as punctuation, so it must be taken
    /// whole. Splitting on it renders two dangling half-sentences.
    #[tokio::test]
    async fn an_advisory_is_taken_whole_never_split_on_its_punctuation() {
        let advisory =
            "tags and collection were not applied; the print was generated and saved normally";
        let mut app = make_settings_test_app();
        app.generate.prompt = TextArea::from(["a test prompt"]);
        deliver_completion(
            &mut app,
            response_with_advisories(vec![advisory.to_string()]),
        );

        assert_eq!(
            app.generate.warning_message.as_deref(),
            Some(advisory),
            "one advisory must survive as exactly one sentence"
        );
    }

    /// Two advisories are joined with the TUI's own separator, which the
    /// prose cannot contain, rather than concatenated into one run-on.
    #[tokio::test]
    async fn several_advisories_are_joined_readably() {
        let mut app = make_settings_test_app();
        app.generate.prompt = TextArea::from(["a test prompt"]);
        deliver_completion(
            &mut app,
            response_with_advisories(vec!["first advisory".to_string(), "second".to_string()]),
        );

        assert_eq!(
            app.generate.warning_message.as_deref(),
            Some("first advisory \u{00b7} second")
        );
    }

    /// An ordinary generation reports nothing: the slot only ever appears
    /// when something really was dropped or adjusted.
    #[tokio::test]
    async fn an_unwarned_generation_leaves_the_advisory_slot_empty() {
        let mut app = make_settings_test_app();
        app.generate.prompt = TextArea::from(["a test prompt"]);
        deliver_completion(&mut app, response_with_advisories(Vec::new()));

        assert_eq!(app.generate.warning_message, None);
        assert_eq!(app.generate.error_message, None);
    }

    /// An advisory describes one print. Starting the next generation clears
    /// it, exactly as the error slot is cleared.
    #[tokio::test]
    async fn starting_a_generation_clears_a_previous_advisory() {
        let mut app = make_settings_test_app();
        app.generate.prompt = TextArea::from(["a cat"]);
        app.generate.warning_message = Some("a stale advisory".to_string());

        app.start_generation();

        assert_eq!(
            app.generate.warning_message, None,
            "a previous print's advisory must not describe this one"
        );
    }

    /// The advisories also land in the Timeline, which is the TUI's durable
    /// per-generation record and can hold text the one-line slot clips.
    #[tokio::test]
    async fn advisories_are_recorded_in_the_timeline() {
        let mut app = make_settings_test_app();
        app.generate.prompt = TextArea::from(["a test prompt"]);
        deliver_completion(
            &mut app,
            response_with_advisories(vec!["tags were not applied".to_string()]),
        );

        assert!(
            app.generate
                .progress
                .log
                .iter()
                .any(|entry| entry.message.contains("tags were not applied")
                    && matches!(entry.style, ProgressStyle::Warning)),
            "the advisory belongs in the per-generation record too"
        );
    }

    /// A sequence carries a filing too — stamped on the stitched print —
    /// so a host that could not apply it must say so on the same slot as a
    /// one-shot. `ChainResponse` carries the identical `request_warnings`.
    #[tokio::test]
    async fn a_completed_chain_surfaces_its_advisories_too() {
        let mut app = make_settings_test_app();
        let advisory =
            "tags and collection were not applied; the print was generated and saved normally";
        app.bg_tx
            .send(BackgroundEvent::ChainComplete {
                stage_count: 2,
                request_warnings: vec![advisory.to_string()],
            })
            .unwrap();
        app.process_background_events();

        assert_eq!(
            app.generate.warning_message.as_deref(),
            Some(advisory),
            "a stitched print's dropped filing must not be silent"
        );
        assert_eq!(app.generate.error_message, None, "the chain succeeded");
        assert!(app
            .generate
            .progress
            .log
            .iter()
            .any(|entry| entry.message.contains("were not applied")
                && matches!(entry.style, ProgressStyle::Warning)));
    }

    #[tokio::test]
    async fn an_unwarned_chain_leaves_the_advisory_slot_empty() {
        let mut app = make_settings_test_app();
        app.bg_tx
            .send(BackgroundEvent::ChainComplete {
                stage_count: 2,
                request_warnings: Vec::new(),
            })
            .unwrap();
        app.process_background_events();

        assert_eq!(app.generate.warning_message, None);
    }

    /// Submitting a sequence clears a previous print's advisory, exactly as
    /// starting a one-shot does.
    #[tokio::test]
    async fn submitting_a_chain_clears_a_previous_advisory() {
        let mut app = make_settings_test_app();
        app.generate.warning_message = Some("a stale advisory".to_string());

        app.dispatch_action(Action::ScriptSubmit);

        assert_eq!(app.generate.warning_message, None);
    }

    fn chain_response_with_advisories(warnings: Vec<String>) -> mold_core::ChainResponse {
        mold_core::ChainResponse {
            request_warnings: warnings,
            video: mold_core::VideoData {
                video_only: None,
                attention_path: None,
                int8_arm: None,
                data: vec![0u8; 4],
                format: OutputFormat::Mp4,
                width: 64,
                height: 64,
                frames: 8,
                fps: 8,
                gif_preview: Vec::new(),
                thumbnail: Vec::new(),
                pipeline: None,
                source_preprocessing: None,
                pipeline_provenance_sha256: None,
                has_audio: false,
                duration_ms: None,
                audio_sample_rate: None,
                audio_channels: None,
            },
            stage_count: 2,
            gpu: None,
            script: Default::default(),
            vram_estimate: None,
        }
    }

    // ── File under (creation-time filing) ──────────────────────

    #[test]
    fn filing_rows_read_absent_until_touched() {
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.auto_tag_title = false;
        assert_eq!(
            params.display_value(&ParamField::Title),
            "\u{27e8}untitled\u{27e9}"
        );
        assert_eq!(
            params.display_value(&ParamField::Tags),
            "\u{27e8}none\u{27e9}"
        );
        assert_eq!(
            params.display_value(&ParamField::Collection),
            "\u{27e8}none\u{27e9}"
        );

        params.title = Some("Smurf Village".into());
        params.tags = vec!["village".into(), "blue".into()];
        params.collection = Some("Blue Period".into());
        assert_eq!(params.display_value(&ParamField::Title), "Smurf Village");
        assert_eq!(params.display_value(&ParamField::Tags), "village, blue");
        assert_eq!(params.display_value(&ParamField::Collection), "Blue Period");
    }

    /// A tag the user did not type must be visible on the row it will join,
    /// before Generate \u{2014} never discovered later in the Library.
    #[test]
    fn tags_row_discloses_the_tag_derived_from_the_title() {
        let config = Config::load_or_default();
        let mut params = GenerateParams::from_config(&config);
        params.auto_tag_title = true;
        params.title = Some("Smurf Village".into());
        assert_eq!(
            params.display_value(&ParamField::Tags),
            "auto: smurf-village"
        );

        params.tags = vec!["village".into()];
        assert_eq!(
            params.display_value(&ParamField::Tags),
            "village \u{00b7} auto: smurf-village"
        );

        params.auto_tag_title = false;
        assert_eq!(params.display_value(&ParamField::Tags), "village");
    }

    /// `GenerateParams::from_config` snapshots the effective preference so
    /// the row the user reads and the request that is sent cannot disagree.
    #[test]
    fn form_snapshots_the_auto_tag_preference_from_config() {
        let mut config = Config::default();
        assert!(config.generate.auto_tag_title, "the preference defaults on");
        assert!(GenerateParams::from_config(&config).auto_tag_title);
        config.generate.auto_tag_title = false;
        assert!(!GenerateParams::from_config(&config).auto_tag_title);
    }

    #[tokio::test]
    async fn filing_editors_open_prefilled_and_write_back_on_confirm() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let mut app = make_settings_test_app();
        app.generate.params.auto_tag_title = false;
        app.generate.params.title = Some("Smurf Village".into());
        app.generate.params.tags = vec!["village".into()];
        app.generate.params.collection = Some("Blue Period".into());

        for (field, expected) in [
            (ParamField::Title, "Smurf Village"),
            (ParamField::Tags, "village"),
            (ParamField::Collection, "Blue Period"),
        ] {
            app.activate_field(field);
            let Some(Popup::FilingInput {
                field: opened,
                input,
                error,
            }) = &app.popup
            else {
                panic!("{field:?} must open its File under editor");
            };
            assert_eq!(*opened, field);
            assert_eq!(input, expected, "{field:?} opens prefilled");
            assert!(error.is_none());
            app.close_popup();
        }

        // Editing the collection writes the normalized name back.
        app.activate_field(ParamField::Collection);
        let Some(Popup::FilingInput { input, .. }) = &mut app.popup else {
            panic!("expected the collection editor");
        };
        input.clear();
        input.push_str("  Red   Period  ");
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        assert!(app.popup.is_none());
        assert_eq!(
            app.generate.params.collection.as_deref(),
            Some("Red Period")
        );
    }

    /// Invalid entry stays on screen with its reason. Nothing invalid ever
    /// reaches a generation request.
    #[tokio::test]
    async fn filing_editor_keeps_invalid_input_visible_with_its_reason() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let mut app = make_settings_test_app();
        app.popup = Some(Popup::FilingInput {
            field: ParamField::Tags,
            // 21 distinct tags: one past the request cap.
            input: (0..=mold_core::MAX_REQUEST_TAGS)
                .map(|i| format!("t{i}"))
                .collect::<Vec<_>>()
                .join(","),
            error: None,
        });
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        let Some(Popup::FilingInput { input, error, .. }) = &mut app.popup else {
            panic!("an over-cap tag list must keep the editor open");
        };
        assert!(error.is_some(), "the reason stays on screen");
        assert!(app.generate.params.tags.is_empty(), "nothing was stored");

        // Typing clears the stale reason, and a valid list commits.
        input.clear();
        input.push_str("smurfs, village");
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Char('!'),
            KeyModifiers::NONE,
        )));
        let Some(Popup::FilingInput { input, error, .. }) = &mut app.popup else {
            panic!("still editing");
        };
        assert!(error.is_none(), "typing dismisses the stale reason");
        input.pop();
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        assert!(app.popup.is_none());
        assert_eq!(
            app.generate.params.tags,
            vec!["smurfs".to_string(), "village".to_string()]
        );
    }

    /// Escape abandons the edit: the stored value is untouched.
    #[tokio::test]
    async fn filing_editor_escape_discards_the_entry() {
        use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
        let mut app = make_settings_test_app();
        app.generate.params.title = Some("Smurf Village".into());
        app.activate_field(ParamField::Title);
        let Some(Popup::FilingInput { input, .. }) = &mut app.popup else {
            panic!("expected the title editor");
        };
        input.push_str(" II");
        app.handle_crossterm_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)));
        assert!(app.popup.is_none());
        assert_eq!(app.generate.params.title.as_deref(), Some("Smurf Village"));
    }

    /// Reset Defaults is the form's explicit "start over" \u{2014} it already
    /// drops the source image, which is no more a model default than a
    /// title is, so the filing goes with it.
    #[tokio::test]
    async fn reset_to_model_defaults_clears_the_filing() {
        let mut app = make_settings_test_app();
        app.generate.params.title = Some("Smurf Village".into());
        app.generate.params.tags = vec!["village".into()];
        app.generate.params.collection = Some("Blue Period".into());

        app.reset_params_to_model_defaults();

        assert_eq!(app.generate.params.title, None);
        assert!(app.generate.params.tags.is_empty());
        assert_eq!(app.generate.params.collection, None);
    }

    /// Leaving the section \u{2014} or the accordion entirely \u{2014} keeps what
    /// was entered; only an explicit clear empties a field.
    #[tokio::test]
    async fn collapsing_the_accordion_keeps_the_filing() {
        use crate::ui::create_form::AdvSection;
        let mut app = make_settings_test_app();
        app.generate.advanced.open = true;
        app.generate.advanced.expanded = Some(AdvSection::Filing);
        app.generate.params.title = Some("Smurf Village".into());
        app.generate.params.tags = vec!["village".into()];

        app.set_advanced_expanded(Some(AdvSection::Output));
        app.dispatch_action(Action::ToggleAdvanced);

        assert_eq!(app.generate.params.title.as_deref(), Some("Smurf Village"));
        assert_eq!(app.generate.params.tags, vec!["village".to_string()]);
    }

    /// Turning the preference back on behind an already-full tag list is
    /// the one state the editors cannot see, so dispatch is the gate \u{2014}
    /// and it refuses in words that name a control the TUI has.
    #[tokio::test]
    async fn generate_refuses_a_filing_the_host_would_reject() {
        let mut app = make_settings_test_app();
        app.generate.prompt = TextArea::from(vec!["a cat".to_string()]);
        app.generate.params.tags = (0..mold_core::MAX_REQUEST_TAGS)
            .map(|i| format!("t{i}"))
            .collect();
        app.generate.params.title = Some("Smurf Village".into());
        app.generate.params.auto_tag_title = true;

        app.start_generation();

        assert!(!app.generate.generating, "nothing was queued");
        let error = app
            .generate
            .error_message
            .as_deref()
            .expect("the refusal is reported");
        assert!(error.contains("Tag by title"), "{error}");
        assert!(!error.contains("--no-auto-tag"), "{error}");
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn settings_tag_with_title_row_round_trips_through_the_db_surface() {
        // `generate.auto_tag_title` is a DB-surface key: toggling the row
        // must land in the settings table `mold run` and the other surfaces
        // read, and refresh the Create form's snapshot so its disclosure
        // and its request agree.
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            let row = find_settings_row(&app, SettingsKey::GenerateAutoTagTitle);
            match &app.build_settings_rows()[row] {
                SettingsRow::Field {
                    label, field_type, ..
                } => {
                    assert_eq!(*label, "Tag by title");
                    assert!(matches!(field_type, SettingsFieldType::Bool));
                }
                other => panic!("expected a field row, got {other:?}"),
            }
            assert_eq!(
                app.settings_display_value(&SettingsKey::GenerateAutoTagTitle),
                "on",
                "the preference defaults on"
            );

            app.generate.params.auto_tag_title = true;
            app.settings_toggle_bool(SettingsKey::GenerateAutoTagTitle);
            assert!(!app.config.generate.auto_tag_title);
            assert_eq!(
                app.settings_display_value(&SettingsKey::GenerateAutoTagTitle),
                "off"
            );
            assert!(
                !app.generate.params.auto_tag_title,
                "the Create form's snapshot follows the preference"
            );

            let db = mold_db::open_default().unwrap().expect("isolated DB");
            let mut stored = mold_core::config::GenerateSettings::default();
            let applied =
                mold_db::config_sync::hydrate_generate_settings_from_db(&db, &mut stored).unwrap();
            assert!(applied, "the DB surface holds the generate row");
            assert!(!stored.auto_tag_title);
        });
    }

    #[tokio::test]
    async fn switching_to_an_incompatible_model_clears_ltx2_video_authority() {
        use crate::ui::create_form::{AdvSection, CreateRow};
        let mut app = make_settings_test_app();
        app.generate.params.model = "flux2-klein:q8".into();
        app.generate.params.enable_audio = Some(true);
        app.generate.params.pipeline = Some(Ltx2PipelineMode::TwoStage);
        app.generate.params.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        app.generate.params.temporal_upscale = Some(Ltx2TemporalUpscale::X2);
        app.generate.params.guidance_overrides.modality_scale = Some(3.0);
        app.generate.params.guidance_overrides.stg_scale = Some(1.5);

        app.sync_generate_capabilities();

        assert!(!app.generate.capabilities.supports_audio);
        assert!(!app.generate.capabilities.supports_video_upscale);
        assert_eq!(app.generate.params.enable_audio, None);
        assert_eq!(app.generate.params.pipeline, None);
        assert_eq!(app.generate.params.spatial_upscale, None);
        assert_eq!(app.generate.params.temporal_upscale, None);
        assert!(app.generate.params.guidance_overrides.is_empty());
        assert!(!app
            .generate
            .rows
            .iter()
            .any(|row| { *row == CreateRow::SectionField(AdvSection::Video, ParamField::Audio) }));
        assert!(!app.generate.rows.iter().any(|row| {
            matches!(
                row,
                CreateRow::SectionField(
                    AdvSection::Video,
                    ParamField::SpatialUpscale | ParamField::TemporalUpscale
                )
            )
        }));
        assert!(!app.generate.rows.iter().any(|row| {
            matches!(
                row,
                CreateRow::SectionField(
                    AdvSection::Video,
                    ParamField::StgScale
                        | ParamField::StgBlocks
                        | ParamField::RescaleScale
                        | ParamField::ModalityScale
                        | ParamField::GuidanceSkip
                )
            )
        }));
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
        app.generate.error_message = Some("a stale size error".to_string());
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
        assert_eq!(app.generate.error_message, None);
    }

    #[tokio::test]
    async fn size_input_popup_applies_off_recipe_sizes_with_an_advisory() {
        // The client never blocks a custom size: an off-recipe entry is
        // applied and the recipe refusal surfaces as an advisory — the
        // server's own admission is the authority at submit time.
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
        app.generate.error_message = Some("a stale size error".to_string());
        app.dispatch_action(Action::Confirm);
        assert!(matches!(app.popup, Some(Popup::SizeInput { .. })));
        for _ in 0..12 {
            app.handle_crossterm_event(Event::Key(KeyEvent::new(
                KeyCode::Backspace,
                KeyModifiers::NONE,
            )));
        }
        // 1000x600 is off every recipe grid (not a multiple of any shipped
        // alignment), so a recipe-carrying model advises; the size applies
        // either way.
        for c in "1001x601".chars() {
            app.handle_crossterm_event(Event::Key(KeyEvent::new(
                KeyCode::Char(c),
                KeyModifiers::NONE,
            )));
        }
        app.handle_crossterm_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )));
        assert_eq!(app.generate.params.width, 1001);
        assert_eq!(app.generate.params.height, 601);
        assert_eq!(app.generate.error_message, None);
        if app.active_generation_recipe().is_some() {
            let advisory = app.generate.warning_message.as_deref().unwrap_or_default();
            assert!(advisory.contains("server may reject"), "got: {advisory}");
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
            negative_default: String::new(),
            negative_explicit_clear: false,
            params: GenerateParams::from_config(&Config::load_or_default()),
            focus: GenerateFocus::Prompt,
            param_index: 0,
            rows: vec![],
            advanced: crate::ui::create_form::AdvancedState::default(),
            param_scroll: 0,
            capabilities: capabilities_for_family("flux"),
            progress: ProgressState::default(),
            live_preview_image: None,
            live_preview_protocol: None,
            preview_image: None,
            image_state: None,
            animation: None,
            generating: true,
            batch_remaining: 3,
            last_seed: None,
            last_generation_time_ms: None,
            error_message: None,
            identity_error: None,
            warning_message: None,
            model_description: String::new(),
            last_output_path: None,
            held_batch: None,
            prompt_transform_token: 0,
            last_mesh_summary: None,
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
            negative_default: String::new(),
            negative_explicit_clear: false,
            params: GenerateParams::from_config(&Config::load_or_default()),
            focus: GenerateFocus::Prompt,
            param_index: 0,
            rows: vec![],
            advanced: crate::ui::create_form::AdvancedState::default(),
            param_scroll: 0,
            capabilities: capabilities_for_family("flux"),
            progress: ProgressState::default(),
            live_preview_image: None,
            live_preview_protocol: None,
            preview_image: None,
            image_state: None,
            animation: None,
            generating: true,
            batch_remaining: 4,
            last_seed: None,
            last_generation_time_ms: None,
            error_message: None,
            identity_error: None,
            warning_message: None,
            model_description: String::new(),
            last_output_path: None,
            held_batch: None,
            prompt_transform_token: 0,
            last_mesh_summary: None,
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
            negative_default: String::new(),
            negative_explicit_clear: false,
            params,
            focus: GenerateFocus::Prompt,
            param_index: 0,
            rows: vec![],
            advanced: crate::ui::create_form::AdvancedState::default(),
            param_scroll: 0,
            capabilities: capabilities_for_family("flux"),
            progress: ProgressState::default(),
            live_preview_image: None,
            live_preview_protocol: None,
            preview_image: None,
            image_state: None,
            animation: None,
            generating: false,
            batch_remaining: 0,
            last_seed: None,
            last_generation_time_ms: None,
            error_message: None,
            identity_error: None,
            warning_message: None,
            model_description: String::new(),
            last_output_path: None,
            held_batch: None,
            prompt_transform_token: 0,
            last_mesh_summary: None,
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
    fn framewise_picker_recovers_latest_paused_job_for_the_selected_video() {
        let job = |id: &str, filename: &str, state, updated_at_ms| mold_core::VideoUpscaleJob {
            contract_version: mold_core::VIDEO_UPSCALE_CONTRACT_VERSION,
            id: id.into(),
            state,
            source: mold_core::VideoUpscaleSource::Library {
                filename: filename.into(),
            },
            model: "real-esrgan-x4plus:fp16".into(),
            scale_factor: 4,
            tile_size: None,
            completed_frames: 0,
            total_frames: 0,
            source_facts: None,
            output_facts: None,
            output_filename: None,
            error: None,
            created_at_ms: 1,
            updated_at_ms,
            disclosure: mold_core::VIDEO_UPSCALE_DISCLOSURE.into(),
        };
        let jobs = vec![
            job(
                "older",
                "clip.mp4",
                mold_core::VideoUpscaleJobState::Running,
                2,
            ),
            job(
                "recovered",
                "clip.mp4",
                mold_core::VideoUpscaleJobState::Paused,
                3,
            ),
            job(
                "other",
                "other.mp4",
                mold_core::VideoUpscaleJobState::Paused,
                4,
            ),
        ];
        assert_eq!(
            recoverable_framewise_upscale(&jobs, "clip.mp4", "real-esrgan-x4plus:fp16")
                .unwrap()
                .id,
            "recovered"
        );
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
    fn generation_model_names_excludes_runtime_unavailable_rows() {
        // A download-only row (the pruned NVFP4 H3 partition) is on disk but
        // has no engine arm — offering it in the create-form picker would
        // only earn a 501 at submit time.
        let runnable = make_test_catalog_entry("flux-dev:q8", 20, 3.5, 1024, 1024, "");
        let unrunnable = ModelInfoExtended {
            runtime_available: Some(false),
            ..make_test_catalog_entry(
                "minimax-h3-fl2va:comfy-pruned-nvfp4",
                8,
                1.0,
                1024,
                1024,
                "",
            )
        };
        let catalog = vec![runnable, unrunnable];
        assert_eq!(
            generation_model_names(&catalog, ""),
            vec!["flux-dev:q8".to_string()]
        );
    }

    #[test]
    fn generation_model_names_keeps_runtime_available_none_as_runnable() {
        // Older servers omit the field; absence must keep meaning runnable.
        let entry = make_test_catalog_entry("flux-dev:q8", 20, 3.5, 1024, 1024, "");
        assert_eq!(
            generation_model_names(std::slice::from_ref(&entry), ""),
            vec!["flux-dev:q8".to_string()]
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn double_click_on_runtime_unavailable_model_row_does_not_open_generate() {
        // The Models-tab double-click shortcut bypasses the create-form
        // picker's `generation_model_names` filter entirely, so it needs its
        // own guard: a download-only row (no engine arm in this build) must
        // never be double-clicked into a request the server will 501.
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
        let mut app = make_settings_test_app();
        app.active_view = View::Models;
        app.layout.models_table = ratatui::layout::Rect::new(0, 0, 80, 20);
        let unrunnable = ModelInfoExtended {
            runtime_available: Some(false),
            // The server names the obstacle; the TUI must repeat that
            // sentence rather than its own layout-only guess (#1276).
            runtime_unavailable_reason: Some(
                mold_core::minimax_h3::RuntimeUnavailableReason::EngineNotBuilt
                    .message()
                    .to_string(),
            ),
            ..make_test_catalog_entry(
                "minimax-h3-fl2va:comfy-pruned-nvfp4",
                8,
                1.0,
                1024,
                1024,
                "",
            )
        };
        app.models.catalog = vec![unrunnable];
        let starting_model = app.generate.params.model.clone();

        let click_row_zero = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 5,
            row: 2, // relative_row = (2 - models_table.y=0).saturating_sub(2) = 0
            modifiers: crossterm::event::KeyModifiers::NONE,
        };
        app.handle_mouse(click_row_zero); // first click: select
        assert_eq!(app.models.selected, 0);
        assert_eq!(app.active_view, View::Models);

        app.handle_mouse(click_row_zero); // second click on the same row: "double-click"

        assert_eq!(
            app.active_view,
            View::Models,
            "double-clicking a runtime_available:false row must not jump to Create"
        );
        assert_eq!(
            app.generate.params.model, starting_model,
            "must not select the download-only model for generation either"
        );
        assert_eq!(
            app.generate.error_message.as_deref(),
            Some(mold_core::minimax_h3::RuntimeUnavailableReason::EngineNotBuilt.message()),
            "should surface the server's own reason instead of silently doing nothing"
        );

        // An older server omits the reason; the layout wording it could
        // always publish stays the fallback.
        app.generate.error_message = None;
        app.models.catalog[0].runtime_unavailable_reason = None;
        app.handle_mouse(click_row_zero);
        assert!(
            app.generate
                .error_message
                .as_deref()
                .unwrap_or_default()
                .contains("No runtime"),
            "got {:?}",
            app.generate.error_message
        );
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn double_click_on_runnable_model_row_still_opens_generate() {
        // Regression guard for the fix above: an ordinary runnable row must
        // keep working exactly as before.
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
        let mut app = make_settings_test_app();
        app.active_view = View::Models;
        app.layout.models_table = ratatui::layout::Rect::new(0, 0, 80, 20);
        app.models.catalog = vec![make_test_catalog_entry(
            "flux-dev:q8",
            20,
            3.5,
            1024,
            1024,
            "",
        )];

        let click_row_zero = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 5,
            row: 2,
            modifiers: crossterm::event::KeyModifiers::NONE,
        };
        app.handle_mouse(click_row_zero);
        app.handle_mouse(click_row_zero);

        assert_eq!(app.active_view, View::Create);
        assert_eq!(app.generate.params.model, "flux-dev:q8");
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
            runtime_available: None,
            runtime_unavailable_reason: None,
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
                ..Default::default()
            },
            downloaded: true,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
            supports_extend: None,
            supports_sequence: None,
            extend_default_overlap_frames: None,
            guidance_capabilities: None,
            source_image: None,
            generation_profile: None,
            supports_identity: None,
            supports_duration_prediction: None,
            runtime_ready: None,
            runtime_readiness_error: None,
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
            host_memory: None,
            durable_media: None,
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
                host_memory: None,
                durable_media: None,
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
            host_memory: None,
            durable_media: None,
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
            let mut entry =
                make_test_catalog_entry("flux-dev:q4", 28, 4.0, 768, 768, "Server FLUX Dev Q4");
            entry.defaults.default_frames = Some(97);
            entry.defaults.default_fps = Some(24);
            app.models.catalog = vec![entry];

            app.update_model("flux-dev:q4");

            assert_eq!(app.generate.params.steps, 28);
            assert!((app.generate.params.guidance - 4.0).abs() < f64::EPSILON);
            assert_eq!(app.generate.params.width, 768);
            assert_eq!(app.generate.params.height, 768);
            assert_eq!(app.generate.params.frames, 97);
            assert_eq!(app.generate.params.fps, 24);
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
            title: None,
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
                title: None,
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
            title: None,
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
            title: None,
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
        // Every owning machine can trash → the trash wording, which
        // deliberately never says "can't be undone".
        assert_eq!(
            delete_confirm_message("a.png", 1, RemovalKind::Trash),
            "Move a.png to the trash?"
        );
        assert_eq!(
            delete_confirm_message("a.png", 2, RemovalKind::Trash),
            "Move a.png to the trash? It exists on 2 machines."
        );
        // Some owner hard-deletes (older server, DB-less local scan,
        // unknown capabilities) → neutral copy that promises no recovery.
        assert_eq!(
            delete_confirm_message("a.png", 1, RemovalKind::Delete),
            "Remove a.png? This can't be undone."
        );
        assert_eq!(
            delete_confirm_message("a.png", 2, RemovalKind::Delete),
            "Remove a.png? It exists on 2 machines. This can't be undone."
        );
    }

    #[test]
    fn removal_kind_is_trash_only_when_every_owner_can_trash() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();
        let mut app = make_settings_test_app();
        let hal = GalleryOrigin {
            host_id: "hal9000-7680".into(),
            url: Some("http://hal9000:7680".into()),
            name: "hal9000".into(),
        };
        let mut entry = make_test_entry_with_name("shared.png");
        entry.origins = vec![GalleryOrigin::local(), hal.clone()];

        // Nothing known: the local scan was DB-less and hal's
        // capabilities were never read → hard delete wording.
        assert_eq!(app.removal_kind_for(&entry), RemovalKind::Delete);

        // Local trash available but hal still unknown → still Delete; an
        // unread host is never assumed to trash.
        app.gallery.local_trash_available = true;
        assert_eq!(app.removal_kind_for(&entry), RemovalKind::Delete);

        // hal advertises a trash → both owners can trash.
        let mut caps = mold_core::ServerCapabilities::default();
        caps.gallery.trash = Some(mold_core::GalleryTrashCapabilities {
            enabled: true,
            retention_days: 30,
        });
        app.machines
            .apply_capabilities("hal9000-7680".into(), Some(caps.clone()));
        assert_eq!(app.removal_kind_for(&entry), RemovalKind::Trash);
        assert_eq!(app.removal_kind_for(&entry).hint_label(), "Trash");

        // An older server (capabilities without `gallery.trash`) drops
        // the whole print back to the honest wording.
        caps.gallery.trash = None;
        app.machines
            .apply_capabilities("hal9000-7680".into(), Some(caps));
        assert_eq!(app.removal_kind_for(&entry), RemovalKind::Delete);

        // A local-only entry follows the local scan's DB availability.
        let local_only = make_test_entry_with_name("local.png");
        assert_eq!(app.removal_kind_for(&local_only), RemovalKind::Trash);
        app.gallery.local_trash_available = false;
        assert_eq!(app.removal_kind_for(&local_only), RemovalKind::Delete);
    }

    #[test]
    fn removal_kind_reads_the_connected_server_under_the_local_slot() {
        // The unregistered `--host` server is polled under LOCAL_HOST_ID
        // but its gallery origin is synthesized from the URL; the
        // capability lookup must bridge the two.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();
        let mut app = make_settings_test_app();
        app.server_url = Some("http://bender:7680".into());
        let mut entry = make_test_entry_with_name("remote.png");
        entry.origins = vec![GalleryOrigin::remote_from_url("http://bender:7680")];
        assert_eq!(app.removal_kind_for(&entry), RemovalKind::Delete);

        let mut caps = mold_core::ServerCapabilities::default();
        caps.gallery.trash = Some(mold_core::GalleryTrashCapabilities {
            enabled: true,
            retention_days: 0,
        });
        app.machines
            .apply_capabilities(crate::hosts::LOCAL_HOST_ID.into(), Some(caps));
        assert_eq!(app.removal_kind_for(&entry), RemovalKind::Trash);
    }

    #[test]
    fn framewise_library_action_requires_the_selected_hosts_capability() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _guard = runtime.enter();
        let mut app = make_settings_test_app();
        app.server_url = Some("http://bender:7680".into());
        let mut video = make_test_entry_with_name("clip.mp4");
        video.origins = vec![GalleryOrigin::remote_from_url("http://bender:7680")];

        assert!(!app.can_upscale_entry(&video));
        let mut caps = mold_core::ServerCapabilities::default();
        caps.video_upscale.available = true;
        app.machines
            .apply_capabilities(crate::hosts::LOCAL_HOST_ID.into(), Some(caps));
        assert!(app.can_upscale_entry(&video));

        let image = make_test_entry_with_name("still.png");
        assert!(app.can_upscale_entry(&image));
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn delete_selected_gallery_image_moves_local_file_to_trash_when_db_available() {
        // With the metadata DB available the local delete is a trash
        // move: the bytes land in `<output_dir>/.trash/` next to a
        // tombstone, the row is flagged, and the live listing no longer
        // shows the print.
        crate::test_env::with_isolated_env(|home| {
            let mut app = make_settings_test_app();
            let gallery = home.join("output");
            std::fs::create_dir_all(&gallery).unwrap();
            let path = gallery.join("mold-trash-me.png");
            let img = image::RgbaImage::from_fn(64, 64, |x, y| {
                image::Rgba([(x * 37 % 251) as u8, (y * 91 % 241) as u8, 200, 255])
            });
            image::DynamicImage::ImageRgba8(img).save(&path).unwrap();
            let db = mold_db::open_default().unwrap().expect("isolated DB");
            db.reconcile(&gallery).unwrap();

            app.gallery.entries.push(GalleryEntry {
                path: path.clone(),
                metadata: make_test_metadata(),
                generation_time_ms: None,
                timestamp: 0,
                server_url: None,
                title: None,
                origins: vec![GalleryOrigin::local()],
            });
            app.gallery.thumbnail_states.push(None);
            app.gallery.thumb_dimensions.push(None);
            app.gallery.thumb_fixed_cache.push(None);
            app.gallery.selected = 0;
            app.gallery.local_trash_available = true;

            app.delete_selected_gallery_image();

            assert!(!path.exists(), "live file moved out of the gallery");
            let trash_dir = mold_db::trash::trash_dir(&gallery);
            assert!(
                trash_dir.join("mold-trash-me.png").is_file(),
                "bytes in .trash/"
            );
            assert!(
                mold_db::trash::tombstone_path(&trash_dir, "mold-trash-me.png").is_file(),
                "tombstone written"
            );
            let row = db.get(&gallery, "mold-trash-me.png").unwrap().unwrap();
            assert!(row.trashed_at_ms.is_some(), "row flagged trashed");
            assert!(db.list_live(Some(&gallery)).unwrap().is_empty());
            assert!(
                app.gallery.entries.is_empty(),
                "tile removed from the Library"
            );
            assert!(
                app.generate.error_message.is_none(),
                "no error: {:?}",
                app.generate.error_message
            );
        });
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn delete_selected_gallery_image_keeps_the_tile_when_the_trash_move_fails() {
        // A failed trash move must not optimistically drop a print that
        // is still on disk: the tile stays, the file stays, the error
        // is surfaced.
        crate::test_env::with_isolated_env(|home| {
            let mut app = make_settings_test_app();
            let gallery = home.join("output");
            std::fs::create_dir_all(&gallery).unwrap();
            // A file that fails the gallery-format guard can't get a row,
            // so the trash move refuses it.
            let path = gallery.join("not-a-gallery-file.txt");
            std::fs::write(&path, b"plain text").unwrap();
            app.gallery.entries.push(GalleryEntry {
                path: path.clone(),
                metadata: make_test_metadata(),
                generation_time_ms: None,
                timestamp: 0,
                server_url: None,
                title: None,
                origins: vec![GalleryOrigin::local()],
            });
            app.gallery.thumbnail_states.push(None);
            app.gallery.thumb_dimensions.push(None);
            app.gallery.thumb_fixed_cache.push(None);
            app.gallery.selected = 0;
            app.gallery.local_trash_available = true;

            app.delete_selected_gallery_image();

            assert!(path.exists(), "file untouched after a failed trash move");
            assert_eq!(app.gallery.entries.len(), 1, "tile kept");
            let msg = app.generate.error_message.clone().unwrap_or_default();
            assert!(msg.starts_with("Move to trash failed"), "{msg}");
        });
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
                title: None,
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
                    local_trash_available: false,
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
            let model = "local-video:test".to_string();
            app.config.models.insert(
                model.clone(),
                mold_core::config::ModelConfig {
                    default_frames: Some(97),
                    default_fps: Some(30),
                    ..Default::default()
                },
            );
            app.update_model(&model);
            // Should succeed without panic and use local config defaults
            assert!(app.generate.params.steps > 0);
            assert!(app.generate.params.width > 0);
            assert_eq!(app.generate.params.frames, 97);
            assert_eq!(app.generate.params.fps, 30);
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
        app.generate.params.spatial_upscale = Some(Ltx2SpatialUpscale::X2);
        app.generate.params.temporal_upscale = Some(Ltx2TemporalUpscale::X2);
        app.generate.params.guidance_overrides.modality_scale = Some(3.0);

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
        assert_eq!(app.generate.params.spatial_upscale, None);
        assert_eq!(app.generate.params.temporal_upscale, None);
        assert!(app.generate.params.guidance_overrides.is_empty());
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn reset_defaults_uses_local_config_when_no_server() {
        let mut app = make_settings_test_app();
        app.server_url = None;
        app.config.models.insert(
            app.generate.params.model.clone(),
            mold_core::config::ModelConfig {
                default_frames: Some(97),
                default_fps: Some(30),
                ..Default::default()
            },
        );

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
        assert_eq!(app.generate.params.frames, 97);
        assert_eq!(app.generate.params.fps, 30);
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
            host_memory: None,
            durable_media: None,
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
            host_memory: None,
            durable_media: None,
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
                host_memory: None,
                durable_media: None,
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
            runtime_available: None,
            runtime_unavailable_reason: None,
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
                ..Default::default()
            },
            downloaded,
            disk_usage_bytes: None,
            remaining_download_bytes: None,
            display_name: None,
            kind: None,
            modality: None,
            nsfw: None,
            supports_audio: None,
            supports_extend: None,
            supports_sequence: None,
            extend_default_overlap_frames: None,
            guidance_capabilities: None,
            source_image: None,
            generation_profile: None,
            supports_identity: None,
            supports_duration_prediction: None,
            runtime_ready: None,
            runtime_readiness_error: None,
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
            connected: true,
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
    async fn machines_small_detail_reaches_device_sixty_three_and_queue() {
        let mut app = make_settings_test_app();
        app.active_view = View::Machines;
        app.machines
            .registry
            .add(machines_test_host("hal9000"))
            .unwrap();
        app.machines.select_next();
        app.machines.focus = crate::hosts::MachinesFocus::Detail;
        app.machines.apply_status(
            "hal9000".into(),
            Some(Box::new(mold_core::ServerStatus {
                version: "0.20.2".into(),
                git_sha: None,
                build_date: None,
                models_loaded: vec![],
                busy: false,
                current_generation: None,
                gpu_info: None,
                uptime_secs: 60,
                hostname: Some("hal9000".into()),
                memory_status: None,
                gpus: None,
                queue_depth: Some(1),
                queue_capacity: Some(64),
                queue_paused: Some(false),
                instance_id: Some("instance-64".into()),
                models_disk: None,
                host_memory: None,
                durable_media: None,
            })),
        );
        let devices = (0..64)
            .map(|ordinal| mold_core::DeviceInfo {
                id: format!("cuda:{ordinal:032x}"),
                backend: mold_core::GpuBackend::Cuda,
                ordinal: Some(ordinal),
                device_kind: mold_core::DeviceKind::FullGpu,
                nvml_uuid: None,
                physical_uuid: None,
                mig_uuid: None,
                mig_parent_uuid: None,
                mig_profile: None,
                name: format!("Synthetic accelerator {ordinal}"),
                pci_bus_id: None,
                compute_capability: Some("10.0".into()),
                memory: mold_core::DeviceMemoryInfo {
                    total_bytes: Some(24 * 1024_u64.pow(3)),
                    used_bytes: Some(ordinal as u64 * 1024),
                    mold_used_bytes: None,
                    other_used_bytes: None,
                },
                telemetry: mold_core::DeviceTelemetry {
                    utilization_percent: Some(ordinal as u8),
                    temperature_c: None,
                    power_w: None,
                },
                desired_enabled: true,
                restart_required: false,
                admin_state: mold_core::DeviceAdminState::Enabled,
                health: mold_core::DeviceHealth::Healthy,
                activity: mold_core::DeviceActivity::Idle,
                schedulable: true,
                unschedulable_reason: None,
                loaded_models: vec![],
                active_work_id: None,
                planned_work_ids: vec![],
            })
            .collect();
        app.machines.apply_devices(
            "hal9000".into(),
            Some(mold_core::DeviceState {
                devices,
                plan_version: 1,
            }),
        );
        app.machines.apply_queue(
            "hal9000".into(),
            Some(mold_core::QueueListingWire {
                entries: vec![mold_core::QueueJobEntryWire {
                    id: "queued-64".into(),
                    model: "queue-visible-model".into(),
                    state: "queued".into(),
                    started_at_unix_ms: 0,
                    position: 0,
                    gpu: None,
                    target_gpu: None,
                    seed_pinned: None,
                    metadata: None,
                    durable: None,
                    held_reason: None,
                    ..Default::default()
                }],
                live_only_entries: vec![],
                plan: None,
                page: None,
            }),
        );

        for _ in 0..63 {
            app.dispatch_action(Action::MachinesDeviceNext);
        }
        assert_eq!(app.machines.device_selected, 63);

        let text = render_view_to_string(&mut app, 100, 20);
        assert!(
            text.contains("Synthetic accelerator 63"),
            "selected device must be visible in a small detail pane:\n{text}"
        );
        assert!(
            text.contains("Queue") && text.contains("queue-visible-model"),
            "queue detail must remain reachable below a large device inventory:\n{text}"
        );
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
    #[serial_test::serial(mold_env)]
    async fn machines_disconnect_persists_and_resets_target_without_forgetting() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.active_view = View::Machines;
            app.machines
                .registry
                .add(machines_test_host("hal9000"))
                .unwrap();
            app.machines.select_next();
            app.target = crate::hosts::GenTarget::Host("hal9000".into());

            app.dispatch_action(Action::MachinesToggleConnection);
            let host = app.machines.registry.get("hal9000").unwrap();
            assert!(!host.connected);
            assert_eq!(app.target, crate::hosts::GenTarget::Auto);
            assert_eq!(app.machines.registry.hosts.len(), 1);

            app.dispatch_action(Action::MachinesToggleConnection);
            assert!(app.machines.registry.get("hal9000").unwrap().connected);
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
                host_memory: None,
                durable_media: None,
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
    async fn machines_cancel_job_gated_on_detail_focus_and_host_capability() {
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
                        durable: None,
                        held_reason: None,
                        ..Default::default()
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
                        durable: None,
                        held_reason: None,
                        ..Default::default()
                    },
                ],
                live_only_entries: vec![],
                plan: None,
                page: None,
            }),
        );

        let mut capabilities = mold_core::ServerCapabilities::default();
        capabilities.queue.cooperative_cancellation = true;
        app.machines
            .apply_capabilities("hal9000".into(), Some(capabilities));

        // HostList focus: `x` is a no-op.
        app.dispatch_action(Action::MachinesCancelJob);
        assert!(app.popup.is_none());

        // Current hosts expose the same cooperative running cancellation.
        app.machines.focus = crate::hosts::MachinesFocus::Detail;
        app.machines.queue_selected = 0;
        app.dispatch_action(Action::MachinesCancelJob);
        let Some(Popup::Confirm { on_confirm, .. }) = &app.popup else {
            panic!("expected the running cancel confirm popup");
        };
        assert!(matches!(
            on_confirm,
            ConfirmAction::CancelHostJob { host_id, job_id }
                if host_id == "hal9000" && job_id == "job-run"
        ));
        app.popup = None;

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

    #[test]
    fn prompt_required_unless_a_video_model_has_a_source_image() {
        let mut config = Config::default();
        config.models.insert(
            "cv:2781713".to_string(),
            mold_core::ModelConfig {
                family: Some("ltx2".to_string()),
                ..mold_core::ModelConfig::default()
            },
        );
        let mut params = GenerateParams::from_config(&config);

        // Text-to-video: still required.
        params.model = "cv:2781713".to_string();
        assert!(prompt_required_for_params(&params, &config));

        // Image-to-video: the source image carries the shot.
        params.source_image_path = Some("/tmp/first-frame.png".to_string());
        assert!(!prompt_required_for_params(&params, &config));

        // Image families keep the prompt required even with a source image.
        params.model = "flux-dev:q4".to_string();
        assert!(prompt_required_for_params(&params, &config));
    }

    #[tokio::test]
    async fn remix_with_existing_original_offers_current_source() {
        let mut app = make_settings_test_app();
        app.set_prompt_text("current edited prompt");
        app.generate.params.original_prompt = Some("original idea".into());

        app.start_prompt_transform(PromptTransformOperation::Remix);

        let Some(Popup::PromptSourceChoice {
            current_prompt,
            root_prompt,
            ..
        }) = &app.popup
        else {
            panic!("expected a source chooser");
        };
        assert_eq!(current_prompt, "current edited prompt");
        assert_eq!(root_prompt, "original idea");
    }

    #[tokio::test]
    async fn authored_prompt_retires_dormant_transform_provenance() {
        let mut app = make_settings_test_app();
        let snapshot = PromptTransformSnapshot {
            operation: PromptTransformOperation::Remix,
            model: app.generate.params.model.clone(),
            target: crate::hosts::GenTarget::Auto,
            task: app.prompt_transform_task(),
            reference_fingerprint: String::new(),
            source_prompt: "old source".into(),
            current_prompt: "old rewrite".into(),
            root_prompt: Some("old source".into()),
            source_kind: mold_core::RemixSourceKind::Original,
        };
        app.generate.params.original_prompt = Some("old source".into());
        app.generate.params.prompt_transform = Some(App::prompt_provenance(&snapshot, vec![]));
        app.generate.params.quick_transform_snapshot = Some(snapshot.clone());
        app.generate.params.prepared_prompts = vec!["prepared sibling".into()];
        app.generate.params.prepared_prompt_transforms =
            vec![App::prompt_provenance(&snapshot, vec![])];
        app.generate.params.prepared_transform_snapshot = Some(snapshot);

        app.set_authored_prompt_text("a completely new idea");

        assert_eq!(
            app.generate.prompt.lines().join("\n"),
            "a completely new idea"
        );
        assert_eq!(app.generate.params.original_prompt, None);
        assert_eq!(app.generate.params.prompt_transform, None);
        assert_eq!(app.generate.params.quick_transform_snapshot, None);
        assert!(app.generate.params.prepared_prompts.is_empty());
        assert!(app.generate.params.prepared_prompt_transforms.is_empty());
        assert_eq!(app.generate.params.prepared_transform_snapshot, None);
    }

    #[tokio::test]
    async fn moving_the_prompt_cursor_preserves_transform_provenance() {
        use crossterm::event::{Event, KeyEvent};

        let mut app = make_settings_test_app();
        app.set_prompt_text("an expanded prompt");
        app.generate.params.original_prompt = Some("the original idea".into());
        app.handle_crossterm_event(Event::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE)));

        assert_eq!(
            app.generate.params.original_prompt.as_deref(),
            Some("the original idea")
        );
    }

    /// Build a catalog entry for a wan checkpoint advertising `capability`,
    /// the way a current server's `/api/models` row arrives.
    fn wan_catalog_entry(
        name: &str,
        capability: Option<mold_core::SourceImageCapability>,
    ) -> ModelInfoExtended {
        let mut entry = make_test_catalog_entry(name, 20, 5.0, 832, 480, "wan test checkpoint");
        entry.info.family = "wan".to_string();
        entry.source_image = capability;
        entry
    }

    /// An I2V checkpoint keeps its Source row visible and unset, so nothing
    /// short of the submit gate can stop the request — and it would otherwise
    /// fail only after the queue slot, the UMT5 encode, and the expert load.
    #[tokio::test]
    async fn required_source_image_blocks_dispatch_with_the_servers_own_message() {
        let mut app = make_settings_test_app();
        app.models.catalog = vec![wan_catalog_entry(
            "wan22-i2v-a14b:q5",
            Some(mold_core::SourceImageCapability::Required),
        )];
        app.generate.params.model = "wan22-i2v-a14b:q5".into();
        app.set_prompt_text("a cat leaping a fence");
        app.sync_generate_capabilities();
        assert!(
            app.generate.capabilities.supports_source_image,
            "the row stays so the contract can be satisfied"
        );

        app.start_generation();

        assert!(!app.generate.generating);
        assert!(app
            .generate
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("needs a source image")));

        // Satisfying the contract clears the gate. Asserted through the
        // decision function rather than a second `start_generation`, which
        // would dispatch a real job from a unit test.
        app.generate.params.source_image_path = Some("/tmp/first-frame.png".into());
        assert_eq!(
            crate::model_info::source_image_contract_error(
                app.source_image_contract(&app.generate.params.model),
                app.generate.params.source_image_path.is_some(),
            ),
            None
        );
    }

    /// An advertised `unsupported` hides the row, so a path carried in from
    /// persistence or a gallery reuse has to be dropped here — leaving it
    /// would submit a request the user has no control left to fix.
    #[tokio::test]
    async fn unsupported_source_image_hides_the_row_and_drops_a_stale_path() {
        let mut app = make_settings_test_app();
        app.models.catalog = vec![wan_catalog_entry(
            // A name the older-server heuristic reads as image-to-video.
            "wan22-i2v-styled-t2v:q5",
            Some(mold_core::SourceImageCapability::Unsupported),
        )];
        app.generate.params.model = "wan22-i2v-styled-t2v:q5".into();
        app.generate.params.source_image_path = Some("/tmp/stale.png".into());

        app.sync_generate_capabilities();

        assert!(!app.generate.capabilities.supports_source_image);
        assert_eq!(app.generate.params.source_image_path, None);
    }

    /// A community fine-tune has no manifest tier, and an older server omits
    /// the field: the name heuristic is the only answer left, and it decides
    /// row visibility without ever blocking a request.
    #[tokio::test]
    async fn unknown_source_image_contract_preserves_the_name_heuristic() {
        let mut app = make_settings_test_app();
        app.models.catalog = vec![wan_catalog_entry("wan-community-i2v-finetune", None)];
        app.generate.params.model = "wan-community-i2v-finetune".into();
        app.generate.params.source_image_path = Some("/tmp/first-frame.png".into());

        app.sync_generate_capabilities();

        assert!(app.generate.capabilities.supports_source_image);
        assert_eq!(
            app.generate.params.source_image_path.as_deref(),
            Some("/tmp/first-frame.png")
        );
        assert_eq!(
            crate::model_info::source_image_contract_error(
                app.source_image_contract(&app.generate.params.model),
                false,
            ),
            None,
            "an unknown contract must not start rejecting requests"
        );
    }

    /// A manifest tier's contract is structural, so it holds against a server
    /// too old to advertise the field — where the name heuristic alone could
    /// only ever offer the row, never require it.
    #[tokio::test]
    async fn manifest_tier_supplies_the_contract_an_older_server_omits() {
        let mut app = make_settings_test_app();
        app.models.catalog = vec![wan_catalog_entry("wan22-i2v-a14b:q5", None)];
        app.generate.params.model = "wan22-i2v-a14b:q5".into();
        app.set_prompt_text("a cat leaping a fence");
        app.sync_generate_capabilities();

        app.start_generation();

        assert!(!app.generate.generating);
        assert!(app
            .generate
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("needs a source image")));
    }

    #[tokio::test]
    async fn prepared_remix_rejects_named_route_staleness_without_erasing_work() {
        let mut app = make_settings_test_app();
        app.set_prompt_text("source idea");
        let snapshot = PromptTransformSnapshot {
            operation: PromptTransformOperation::Remix,
            model: app.generate.params.model.clone(),
            target: crate::hosts::GenTarget::Auto,
            task: app.prompt_transform_task(),
            reference_fingerprint: crate::h3_references::authority_fingerprint(
                None,
                &app.generate.params.reference_paths,
            )
            .unwrap(),
            source_prompt: "source idea".into(),
            current_prompt: "source idea".into(),
            root_prompt: None,
            source_kind: mold_core::RemixSourceKind::Direct,
        };
        app.prepare_prompt_variants(
            snapshot,
            vec![
                mold_core::RemixVariant {
                    prompt: "variation one".into(),
                    dimensions: vec![mold_core::RemixDimension::Camera],
                },
                mold_core::RemixVariant {
                    prompt: "variation two".into(),
                    dimensions: vec![mold_core::RemixDimension::Lighting],
                },
            ],
        );
        app.target = crate::hosts::GenTarget::Local;

        app.start_generation();

        assert!(!app.generate.generating);
        assert_eq!(app.generate.params.prepared_prompts.len(), 2);
        assert!(app
            .generate
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("generation target changed")));
    }

    #[tokio::test]
    #[serial_test::serial(mold_env)]
    async fn transform_snapshot_names_model_prompt_and_reference_staleness() {
        crate::test_env::with_isolated_env(|_home| {
            let mut app = make_settings_test_app();
            app.machines
                .registry
                .add(machines_test_host("reference-host"))
                .unwrap();
            crate::hosts::save_api_key("reference-host", "sekrit");
            app.target = crate::hosts::GenTarget::Host("reference-host".into());
            app.set_prompt_text("frozen prompt");
            app.generate
                .params
                .reference_paths
                .push(crate::h3_references::ReferencePath {
                    kind: crate::h3_references::ReferenceKind::Image,
                    path: "/tmp/first.png".into(),
                });
            let snapshot = PromptTransformSnapshot {
                operation: PromptTransformOperation::Remix,
                model: app.generate.params.model.clone(),
                target: app.target.clone(),
                task: app.prompt_transform_task(),
                reference_fingerprint: app.reference_fingerprint_for_target(&app.target).unwrap(),
                source_prompt: "frozen prompt".into(),
                current_prompt: "frozen prompt".into(),
                root_prompt: None,
                source_kind: mold_core::RemixSourceKind::Direct,
            };
            app.generate.params.model = "different-model".into();
            assert!(app
                .prompt_transform_staleness(&snapshot)
                .is_some_and(|message| message.contains("model changed")));

            app.generate.params.model = snapshot.model.clone();
            app.set_prompt_text("edited prompt");
            assert!(app
                .prompt_transform_staleness(&snapshot)
                .is_some_and(|message| message.contains("current prompt changed")));

            app.set_prompt_text("frozen prompt");
            app.generate.params.reference_paths[0].path = "/tmp/replacement.png".into();
            assert!(app
                .prompt_transform_staleness(&snapshot)
                .is_some_and(|message| message.contains("ordered references changed")));
        });
    }

    #[tokio::test]
    async fn h3_reference_remix_fails_before_creating_an_unusable_batch() {
        let mut app = make_settings_test_app();
        app.set_prompt_text("animate the anchor");
        app.generate
            .params
            .reference_paths
            .push(crate::h3_references::ReferencePath {
                kind: crate::h3_references::ReferenceKind::Image,
                path: "/tmp/anchor.png".into(),
            });

        app.start_prompt_transform(PromptTransformOperation::Remix);

        assert!(app.popup.is_none());
        assert!(app
            .generate
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("Batch 1")));
    }

    #[tokio::test]
    async fn applied_single_remix_blocks_model_staleness_but_releases_host() {
        let mut app = make_settings_test_app();
        app.set_prompt_text("source idea");
        let snapshot = PromptTransformSnapshot {
            operation: PromptTransformOperation::Remix,
            model: app.generate.params.model.clone(),
            target: crate::hosts::GenTarget::Auto,
            task: app.prompt_transform_task(),
            reference_fingerprint: crate::h3_references::authority_fingerprint(
                None,
                &app.generate.params.reference_paths,
            )
            .unwrap(),
            source_prompt: "source idea".into(),
            current_prompt: "source idea".into(),
            root_prompt: None,
            source_kind: mold_core::RemixSourceKind::Direct,
        };
        app.apply_prompt_variant(
            snapshot,
            mold_core::RemixVariant {
                prompt: "reviewed variation".into(),
                dimensions: vec![mold_core::RemixDimension::Composition],
            },
        );

        let frozen = app
            .generate
            .params
            .quick_transform_snapshot
            .clone()
            .expect("single apply must retain its semantic snapshot");
        app.target = crate::hosts::GenTarget::Local;
        assert_eq!(
            app.quick_transform_staleness(&frozen),
            None,
            "quick Remix deliberately releases only its original host"
        );

        app.generate.params.model = "different-model".into();
        app.start_generation();

        assert!(!app.generate.generating);
        assert!(app
            .generate
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("model changed")));
        assert_eq!(
            app.generate.prompt.lines().join("\n"),
            "reviewed variation",
            "stale recovery must preserve reviewed text"
        );
    }
}
