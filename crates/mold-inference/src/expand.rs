//! Local LLM-powered prompt expansion using quantized Qwen3 GGUF models.
//!
//! Loads a small Qwen3 model (1.7B or 0.6B at Q4), generates expanded prompts,
//! then drops the weights to free VRAM for the diffusion pipeline.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use sha2::{Digest, Sha256};
use std::io::Seek;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use mold_core::expand::{ExpandConfig, ExpandResult, PromptExpander};
use mold_core::expand_prompts::{build_batch_messages, build_single_messages, format_chatml};

use crate::device::{
    discover_gpus, expand_vram_threshold, memory_status_string, preflight_memory_check,
    resolve_gpu_selection, select_expand_device_with_preference, ExpandPlacement,
};
use crate::progress::{InferenceCancellationToken, ProgressCallback, ProgressReporter};
use mold_core::types::GpuSelection;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedExpandArtifact {
    pub path: PathBuf,
    pub size_bytes: u64,
    /// A metadata identity/stat fence (canonical path, size, and filesystem
    /// identity/change-time fields), not a digest of the artifact contents.
    pub identity_fingerprint: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedExpandArtifacts {
    pub model_name: String,
    pub model: ResolvedExpandArtifact,
    pub tokenizer: ResolvedExpandArtifact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactExpandPlacement {
    Cpu,
    Device {
        backend: mold_core::GpuBackend,
        ordinal: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedExpandExecutionPlan {
    pub artifacts: ResolvedExpandArtifacts,
    pub placement: ExactExpandPlacement,
    pub predicted_vram_peak_bytes: u64,
    pub predicted_host_increment_bytes: u64,
    pub execution_fingerprint: String,
}

impl ResolvedExpandExecutionPlan {
    /// Validate both artifact identity/stat fences and every admitted execution
    /// field before any device is constructed.
    pub fn validate(&self) -> Result<()> {
        validate_frozen_artifact("expand model", &self.artifacts.model)?;
        validate_frozen_artifact("expand tokenizer", &self.artifacts.tokenizer)?;
        let (expected_vram, expected_host) =
            expand_plan_memory_floors(&self.artifacts, self.placement);
        let expected_fingerprint = expand_execution_fingerprint(
            &self.artifacts,
            self.placement,
            expected_vram,
            expected_host,
        );
        if self.predicted_vram_peak_bytes != expected_vram
            || self.predicted_host_increment_bytes != expected_host
            || self.execution_fingerprint != expected_fingerprint
        {
            bail!("expand execution plan changed after resolution");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpandSafePoint {
    BeforeLoad,
    AfterLoad,
    AfterTokenization,
    BeforePromptForward,
    TokenIteration,
    BeforeDecode,
}

fn expansion_checkpoint(progress: &ProgressReporter, _safe_point: ExpandSafePoint) -> Result<()> {
    progress.checkpoint().map_err(anyhow::Error::from)
}

fn artifact_identity_fingerprint(path: &Path, size_bytes: u64) -> Result<String> {
    let metadata = std::fs::metadata(path)?;
    let mut hash = Sha256::new();
    hash.update(path.as_os_str().as_encoded_bytes());
    hash.update(size_bytes.to_le_bytes());
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        hash.update(metadata.dev().to_le_bytes());
        hash.update(metadata.ino().to_le_bytes());
        hash.update(metadata.ctime().to_le_bytes());
        hash.update(metadata.ctime_nsec().to_le_bytes());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        hash.update(metadata.creation_time().to_le_bytes());
        hash.update(metadata.last_write_time().to_le_bytes());
        hash.update(metadata.file_size().to_le_bytes());
    }
    #[cfg(not(any(unix, windows)))]
    {
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos());
        hash.update(modified.to_le_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn freeze_artifact(path: PathBuf) -> Result<ResolvedExpandArtifact> {
    let path = std::fs::canonicalize(&path)
        .map_err(|error| anyhow::anyhow!("could not resolve {}: {error}", path.display()))?;
    let size_bytes = std::fs::metadata(&path)?.len();
    let identity_fingerprint = artifact_identity_fingerprint(&path, size_bytes)?;
    Ok(ResolvedExpandArtifact {
        path,
        size_bytes,
        identity_fingerprint,
    })
}

fn validate_frozen_artifact(label: &str, frozen: &ResolvedExpandArtifact) -> Result<()> {
    let current = freeze_artifact(frozen.path.clone())
        .map_err(|error| anyhow::anyhow!("{label} artifact changed: {error}"))?;
    if &current != frozen {
        bail!(
            "{label} artifact changed after planning: expected {} bytes identity fingerprint {}, \
             found {} bytes identity fingerprint {}",
            frozen.size_bytes,
            frozen.identity_fingerprint,
            current.size_bytes,
            current.identity_fingerprint
        );
    }
    Ok(())
}

fn expand_candidate_dirs(expand_model: &str) -> Vec<String> {
    let tag = expand_model.split(':').nth(1).unwrap_or("q8");
    if expand_model.contains("small") {
        vec![
            format!("qwen3-expand-small-{tag}"),
            format!("qwen3-expand-{tag}"),
            "qwen3-expand".to_string(),
        ]
    } else {
        vec![
            format!("qwen3-expand-{tag}"),
            format!("qwen3-expand-small-{tag}"),
            "qwen3-expand".to_string(),
        ]
    }
}

fn expand_plan_memory_floors(
    artifacts: &ResolvedExpandArtifacts,
    placement: ExactExpandPlacement,
) -> (u64, u64) {
    let model_and_tokenizer = artifacts
        .model
        .size_bytes
        .saturating_add(artifacts.tokenizer.size_bytes);
    let activation_bytes = crate::device::EXPAND_ACTIVATION_HEADROOM;
    let predicted_vram_peak_bytes = match placement {
        ExactExpandPlacement::Cpu => 0,
        ExactExpandPlacement::Device { .. } => {
            artifacts.model.size_bytes.saturating_add(activation_bytes)
        }
    };
    let predicted_host_increment_bytes = match placement {
        ExactExpandPlacement::Cpu
        | ExactExpandPlacement::Device {
            backend: mold_core::GpuBackend::Metal,
            ..
        } => model_and_tokenizer.saturating_add(activation_bytes),
        ExactExpandPlacement::Device {
            backend: mold_core::GpuBackend::Cuda,
            ..
        } => model_and_tokenizer,
    };
    (predicted_vram_peak_bytes, predicted_host_increment_bytes)
}

fn expand_execution_fingerprint(
    artifacts: &ResolvedExpandArtifacts,
    placement: ExactExpandPlacement,
    predicted_vram_peak_bytes: u64,
    predicted_host_increment_bytes: u64,
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.expand.execution.v1");
    hash.update(artifacts.model_name.as_bytes());
    for artifact in [&artifacts.model, &artifacts.tokenizer] {
        hash.update(artifact.path.as_os_str().as_encoded_bytes());
        hash.update(artifact.size_bytes.to_le_bytes());
        hash.update(artifact.identity_fingerprint.as_bytes());
    }
    match placement {
        ExactExpandPlacement::Cpu => hash.update(b"cpu"),
        ExactExpandPlacement::Device { backend, ordinal } => {
            hash.update(match backend {
                mold_core::GpuBackend::Cuda => b"cuda".as_slice(),
                mold_core::GpuBackend::Metal => b"metal".as_slice(),
            });
            hash.update(ordinal.to_le_bytes());
        }
    }
    hash.update(predicted_vram_peak_bytes.to_le_bytes());
    hash.update(predicted_host_increment_bytes.to_le_bytes());
    format!("{:x}", hash.finalize())
}

/// Resolve and freeze the exact local prompt-expansion artifacts without
/// discovering or constructing any execution device.
pub fn resolve_local_expand_artifacts(
    config: &mold_core::Config,
    expand_model: Option<&str>,
) -> Result<ResolvedExpandArtifacts> {
    let models_dir = config.resolved_models_dir();
    let expand_model = expand_model.unwrap_or(&config.expand.model);
    let candidate_dirs = expand_candidate_dirs(expand_model);
    let model_path = candidate_dirs
        .iter()
        .find_map(|dir_name| find_gguf_in_dir(&models_dir.join(dir_name), expand_model))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "local expand model '{expand_model}' not found — run: mold pull {expand_model}"
            )
        })?;
    let shared_tokenizer = models_dir.join("shared/qwen3-expand/tokenizer.json");
    let tokenizer_path = if shared_tokenizer.exists() {
        shared_tokenizer
    } else {
        candidate_dirs
            .iter()
            .find_map(|dir_name| find_tokenizer_in_dir(&models_dir.join(dir_name)))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "local expand tokenizer for '{expand_model}' was not found — \
                     re-pull the expand model"
                )
            })?
    };
    Ok(ResolvedExpandArtifacts {
        model_name: expand_model.to_string(),
        model: freeze_artifact(model_path)?,
        tokenizer: freeze_artifact(tokenizer_path)?,
    })
}

/// Resolve one immutable prompt-expansion execution plan.
///
/// CPU and Metal charge model, tokenizer, and activation/KV memory to the
/// host. CUDA conservatively charges model and tokenizer bytes as load
/// staging while reserving the model plus activation floor in VRAM.
pub fn resolve_expand_execution_plan(
    config: &mold_core::Config,
    expand_model: Option<&str>,
    placement: ExactExpandPlacement,
) -> Result<ResolvedExpandExecutionPlan> {
    let artifacts = resolve_local_expand_artifacts(config, expand_model)?;
    Ok(resolve_expand_execution_plan_from_artifacts(
        artifacts, placement,
    ))
}

/// Build another placement candidate from one already-frozen artifact set.
///
/// Server scheduling uses this to guarantee that every CPU/GPU candidate for
/// one utility child has the same model/tokenizer identity fence.
pub fn resolve_expand_execution_plan_from_artifacts(
    artifacts: ResolvedExpandArtifacts,
    placement: ExactExpandPlacement,
) -> ResolvedExpandExecutionPlan {
    let (predicted_vram_peak_bytes, predicted_host_increment_bytes) =
        expand_plan_memory_floors(&artifacts, placement);
    let execution_fingerprint = expand_execution_fingerprint(
        &artifacts,
        placement,
        predicted_vram_peak_bytes,
        predicted_host_increment_bytes,
    );
    ResolvedExpandExecutionPlan {
        artifacts,
        placement,
        predicted_vram_peak_bytes,
        predicted_host_increment_bytes,
        execution_fingerprint,
    }
}

fn create_exact_device_with<F>(
    plan: &ResolvedExpandExecutionPlan,
    progress: &ProgressReporter,
    gpu_factory: F,
) -> Result<Device>
where
    F: FnOnce(mold_core::GpuBackend, usize, &ProgressReporter) -> Result<Device>,
{
    plan.validate()?;
    expansion_checkpoint(progress, ExpandSafePoint::BeforeLoad)?;
    match plan.placement {
        ExactExpandPlacement::Cpu => {
            progress.info("Using exact CPU placement for prompt expansion");
            preflight_memory_check(
                "Expand LLM",
                plan.artifacts
                    .model
                    .size_bytes
                    .saturating_add(plan.artifacts.tokenizer.size_bytes),
                crate::device::EXPAND_ACTIVATION_HEADROOM,
            )?;
            Ok(Device::Cpu)
        }
        ExactExpandPlacement::Device { backend, ordinal } => {
            if backend == mold_core::GpuBackend::Metal {
                preflight_memory_check(
                    "Expand LLM",
                    plan.artifacts
                        .model
                        .size_bytes
                        .saturating_add(plan.artifacts.tokenizer.size_bytes),
                    crate::device::EXPAND_ACTIVATION_HEADROOM,
                )?;
            }
            gpu_factory(backend, ordinal, progress)
        }
    }
}

fn select_legacy_dynamic_expand_device(
    threshold: u64,
    model_size: u64,
    gpu_selection: &GpuSelection,
    preferred_gpu: Option<usize>,
    progress: &ProgressReporter,
) -> Result<Device> {
    let discovered = discover_gpus();
    let gpus = resolve_gpu_selection(&discovered, gpu_selection)?;
    let is_metal = candle_core::utils::metal_is_available();
    let placement = select_expand_device_with_preference(&gpus, threshold, is_metal, preferred_gpu);
    match placement {
        ExpandPlacement::Gpu(ordinal) => {
            let device = crate::device::create_device(ordinal, progress)?;
            if device.is_cpu() || device.is_metal() {
                preflight_memory_check(
                    "Expand LLM",
                    model_size,
                    crate::device::EXPAND_ACTIVATION_HEADROOM,
                )?;
            }
            Ok(device)
        }
        ExpandPlacement::Cpu => {
            if gpus.is_empty() {
                progress.info("Using CPU for prompt expansion (no GPU detected)");
            } else {
                progress.info(&format!(
                    "Using CPU for prompt expansion (needed {:.1} GB, no GPU had room)",
                    threshold as f64 / 1_000_000_000.0,
                ));
            }
            preflight_memory_check(
                "Expand LLM",
                model_size,
                crate::device::EXPAND_ACTIVATION_HEADROOM,
            )?;
            Ok(Device::Cpu)
        }
    }
}

fn report_expand_memory_status_with<F>(
    exact_plan: Option<&ResolvedExpandExecutionPlan>,
    progress: &ProgressReporter,
    probe: F,
) where
    F: FnOnce() -> Option<String>,
{
    // `memory_status_string()` probes CUDA ordinal 0 in CUDA builds. Exact
    // execution must not initialize a CUDA context for a CPU lane or touch a
    // sibling of the admitted GPU merely for diagnostics. Scheduler telemetry
    // owns per-device memory reporting for exact plans.
    if exact_plan.is_some() {
        return;
    }
    if let Some(memory_status) = probe() {
        progress.info(&memory_status);
    }
}

/// Local prompt expander using quantized Qwen3 GGUF.
pub struct LocalExpander {
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    progress: ProgressReporter,
    gpu_selection: GpuSelection,
    preferred_gpu: Option<usize>,
    exact_plan: Option<ResolvedExpandExecutionPlan>,
}

impl LocalExpander {
    /// Create a new local expander with paths to the GGUF model and tokenizer.
    pub fn new(model_path: impl Into<PathBuf>, tokenizer_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_path: tokenizer_path.into(),
            progress: ProgressReporter::default(),
            gpu_selection: GpuSelection::All,
            preferred_gpu: None,
            exact_plan: None,
        }
    }

    /// Construct an expander that consumes one immutable admitted plan.
    ///
    /// This path never discovers devices, reads `MOLD_DEVICE`, walks sibling
    /// GPUs, or silently falls back. Artifact drift and device-construction
    /// failure are returned to the caller for an explicit scheduler replan.
    pub fn from_resolved_plan(plan: ResolvedExpandExecutionPlan) -> Self {
        Self {
            model_path: plan.artifacts.model.path.clone(),
            tokenizer_path: plan.artifacts.tokenizer.path.clone(),
            progress: ProgressReporter::default(),
            gpu_selection: GpuSelection::None,
            preferred_gpu: None,
            exact_plan: Some(plan),
        }
    }

    /// Set a progress callback for reporting device selection, loading, and generation status.
    pub fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
    }

    pub fn set_cancellation_token(&mut self, token: InferenceCancellationToken) {
        self.progress.set_cancellation_token(token);
    }

    pub fn clear_cancellation_token(&mut self) {
        self.progress.clear_cancellation_token();
    }

    /// Run one attempt with cooperative cancellation installed, then clear the
    /// token on success, error, or panic so the expander is safe to reuse.
    pub fn expand_with_cancellation(
        &mut self,
        prompt: &str,
        config: &ExpandConfig,
        token: InferenceCancellationToken,
    ) -> Result<ExpandResult> {
        self.set_cancellation_token(token);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            PromptExpander::expand(self, prompt, config)
        }));
        self.clear_cancellation_token();
        match result {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Restrict local expansion to the GPU ordinals selected by the caller.
    pub fn with_gpu_selection(mut self, gpu_selection: GpuSelection) -> Self {
        self.gpu_selection = gpu_selection;
        self
    }

    /// Prefer this GPU ordinal when it is allowed and has enough free VRAM.
    pub fn with_preferred_gpu(mut self, preferred_gpu: Option<usize>) -> Self {
        self.preferred_gpu = preferred_gpu;
        self
    }

    /// Try to create a local expander by finding the model files.
    ///
    /// Searches the standard mold models directory for the expand model's
    /// GGUF and tokenizer files, checking both manifest storage paths
    /// (e.g., `qwen3-expand-q8/`) and the shared tokenizer location.
    ///
    /// `expand_model` overrides the model spec from config (e.g., from
    /// `--expand-model` or `MOLD_EXPAND_MODEL`). Pass `None` to use
    /// `config.expand.model`.
    ///
    /// Returns `None` if the model hasn't been pulled yet.
    pub fn from_config(config: &mold_core::Config, expand_model: Option<&str>) -> Option<Self> {
        Self::from_config_legacy_dynamic(config, expand_model)
    }

    /// Legacy device-discovering adapter for CLI and pre-V2 server paths.
    ///
    /// New scheduler-owned work must use [`Self::from_resolved_plan`].
    pub fn from_config_legacy_dynamic(
        config: &mold_core::Config,
        expand_model: Option<&str>,
    ) -> Option<Self> {
        let artifacts = resolve_local_expand_artifacts(config, expand_model).ok()?;
        Some(Self::new(artifacts.model.path, artifacts.tokenizer.path))
    }

    /// Load model, generate text, drop model.
    fn generate_text(&self, prompt_text: &str, config: &ExpandConfig) -> Result<String> {
        // Size the VRAM/RAM budget off the actual GGUF file (weights + 2 GB
        // activations), not a static constant — a 4 GB q8 model needs ~6 GB
        // to fit, not 2 GB.
        let model_size = self.exact_plan.as_ref().map_or_else(
            || {
                std::fs::metadata(&self.model_path)
                    .map(|metadata| metadata.len())
                    .unwrap_or(0)
            },
            |plan| plan.artifacts.model.size_bytes,
        );
        let threshold = expand_vram_threshold(model_size);

        let device = if let Some(plan) = &self.exact_plan {
            create_exact_device_with(plan, &self.progress, |backend, ordinal, progress| {
                crate::device::create_exact_gpu_device(backend, ordinal, progress)
            })?
        } else {
            expansion_checkpoint(&self.progress, ExpandSafePoint::BeforeLoad)?;
            select_legacy_dynamic_expand_device(
                threshold,
                model_size,
                &self.gpu_selection,
                self.preferred_gpu,
                &self.progress,
            )?
        };

        let device_label = if device.is_metal() {
            "Metal"
        } else if device.is_cuda() {
            "CUDA"
        } else {
            "CPU"
        };

        // Report memory status
        report_expand_memory_status_with(self.exact_plan.as_ref(), &self.progress, || {
            memory_status_string()
        });

        // Load GGUF model
        let load_start = std::time::Instant::now();
        let stage_name = format!("Loading expand model ({device_label})");
        self.progress.stage_start(&stage_name);

        let mut file = std::fs::File::open(&self.model_path)
            .map_err(|e| anyhow::anyhow!("failed to open expand model: {e}"))?;
        let ct = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("failed to read GGUF content: {e}"))?;
        file.seek(std::io::SeekFrom::Start(0))?;
        let mut model = ModelWeights::from_gguf(ct, &mut file, &device)
            .map_err(|e| anyhow::anyhow!("failed to load expand model weights: {e}"))?;
        expansion_checkpoint(&self.progress, ExpandSafePoint::AfterLoad)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&self.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load expand tokenizer: {e}"))?;

        self.progress.phase_done(
            crate::ProgressPhase::ModelLoad,
            &stage_name,
            load_start.elapsed(),
        );

        // Tokenize prompt
        let encoding = tokenizer
            .encode(prompt_text, false)
            .map_err(|e| anyhow::anyhow!("failed to tokenize expand prompt: {e}"))?;
        let input_ids = encoding.get_ids();
        expansion_checkpoint(&self.progress, ExpandSafePoint::AfterTokenization)?;

        // Generate tokens autoregressively
        let gen_start = std::time::Instant::now();
        let mut all_tokens: Vec<u32> = input_ids.to_vec();
        let mut generated_tokens: Vec<u32> = Vec::new();

        // Get stop tokens
        let eos_token = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"));
        let start_think_token = tokenizer.token_to_id("<think>");
        let end_think_token = tokenizer.token_to_id("</think>");

        let max_new_tokens = config.max_tokens as usize;
        let mut in_thinking = false;

        // Process the prompt through the model first
        expansion_checkpoint(&self.progress, ExpandSafePoint::BeforePromptForward)?;
        let prompt_offset = {
            let input = Tensor::new(input_ids, &device)?.unsqueeze(0)?;
            let _logits = model.forward(&input, 0)?;
            input_ids.len()
        };

        // Generate new tokens one at a time
        let mut last_token = *input_ids.last().unwrap_or(&0);
        for generated_offset in 0..max_new_tokens {
            expansion_checkpoint(&self.progress, ExpandSafePoint::TokenIteration)?;
            let input = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, prompt_offset + generated_offset)?;

            // Sample next token
            let next_token = sample_token(&logits, config.temperature, config.top_p)?;

            // Check for stop conditions
            if let Some(eos) = eos_token {
                if next_token == eos {
                    break;
                }
            }

            // Track thinking mode — skip <think>...</think> tokens from output
            if let Some(st) = start_think_token {
                if next_token == st {
                    in_thinking = true;
                }
            }
            if let Some(et) = end_think_token {
                if next_token == et {
                    in_thinking = false;
                    all_tokens.push(next_token);
                    last_token = next_token;
                    continue; // Don't include </think> in generated_tokens
                }
            }

            all_tokens.push(next_token);
            if !in_thinking {
                generated_tokens.push(next_token);
            }
            last_token = next_token;
        }

        // Report generation speed
        let gen_elapsed = gen_start.elapsed().as_secs_f64();
        let tok_per_sec = generated_tokens.len() as f64 / gen_elapsed.max(0.001);
        self.progress.info(&format!(
            "Generated {} tokens ({:.1} tok/s)",
            generated_tokens.len(),
            tok_per_sec
        ));

        // Decode generated tokens
        expansion_checkpoint(&self.progress, ExpandSafePoint::BeforeDecode)?;
        let output = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("failed to decode generated tokens: {e}"))?;
        self.progress.phase_done(
            crate::ProgressPhase::PromptEncode,
            "Prompt expansion",
            gen_start.elapsed(),
        );

        // Clear KV cache and drop model weights to free their allocations
        // while preserving the live context shared with the diffusion engine.
        model.clear_kv_cache();
        drop(model);
        let _ = device;

        Ok(output)
    }
}

impl PromptExpander for LocalExpander {
    fn expand(&self, prompt: &str, config: &ExpandConfig) -> Result<ExpandResult> {
        let family_override = config.family_overrides.get(&config.model_family);
        let messages = if config.variations > 1 {
            build_batch_messages(
                prompt,
                &config.model_family,
                config.variations,
                config.batch_prompt.as_deref(),
                family_override,
                config.style.as_deref(),
            )
        } else {
            build_single_messages(
                prompt,
                &config.model_family,
                config.system_prompt.as_deref(),
                family_override,
                config.style.as_deref(),
            )
        };

        let prompt_text = format_chatml(&messages, config.thinking);
        let output = self.generate_text(&prompt_text, config)?;

        let expanded = if config.variations > 1 {
            mold_core::expand::parse_variations_public(&output, config.variations)
        } else {
            vec![mold_core::expand::clean_expanded_prompt_public(&output)]
        };

        // Validate we got reasonable output
        if expanded.is_empty() || expanded.iter().all(|s| s.is_empty()) {
            bail!(
                "expand model produced empty output. The model may need re-downloading: \
                 mold pull qwen3-expand"
            );
        }

        Ok(ExpandResult {
            original: prompt.to_string(),
            expanded,
        })
    }
}

/// Sample a token from logits using temperature and top-p (nucleus) sampling.
fn sample_token(logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32> {
    let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = logits.to_vec1()?;

    if temperature <= 0.0 {
        // Greedy: pick the max
        let (max_idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        return Ok(max_idx as u32);
    }

    // Apply temperature
    let scaled: Vec<f64> = logits_vec.iter().map(|&x| x as f64 / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp.iter().sum();
    let mut probs: Vec<(usize, f64)> = exp.iter().enumerate().map(|(i, &e)| (i, e / sum)).collect();

    // Sort by probability descending for top-p
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply top-p (nucleus) sampling
    let mut cumulative = 0.0;
    let mut cutoff_idx = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }
    let candidates = &probs[..cutoff_idx];

    // Re-normalize
    let total: f64 = candidates.iter().map(|&(_, p)| p).sum();
    let r: f64 = rand::random::<f64>() * total;

    let mut acc = 0.0;
    for &(idx, p) in candidates {
        acc += p;
        if acc >= r {
            return Ok(idx as u32);
        }
    }

    // Fallback to last candidate
    Ok(candidates.last().map(|&(idx, _)| idx as u32).unwrap_or(0))
}

/// Find a GGUF file in a directory, preferring one matching the model spec's
/// quantization tag (e.g., "q8" from "qwen3-expand:q8").
fn find_gguf_in_dir(dir: &Path, model_spec: &str) -> Option<PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "gguf")
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|entry| entry.file_name());

    // Extract quantization tag from model spec (e.g., "qwen3-expand:q8" → "q8")
    let quant_tag = model_spec.split(':').nth(1).unwrap_or("q8").to_uppercase();

    // Prefer file matching the requested quantization
    for entry in &entries {
        let name = entry.file_name().to_string_lossy().to_uppercase();
        if name.contains(&quant_tag) {
            return Some(entry.path());
        }
    }

    // Fallback to any GGUF
    entries.first().map(|e| e.path())
}

/// Find a tokenizer file in a directory.
fn find_tokenizer_in_dir(dir: &Path) -> Option<PathBuf> {
    let candidates = [
        "tokenizer.json",
        "qwen3-tokenizer.json",
        "tokenizer_config.json",
    ];
    for name in &candidates {
        let path = dir.join(name);
        if path.exists() {
            return Some(path);
        }
    }
    // Search subdirectories
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut entries = entries.flatten().collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname.contains("tokenizer") && fname.ends_with(".json") {
                return Some(entry.path());
            }
        }
    }
    None
}

#[cfg(test)]
mod exact_plan_tests {
    use super::*;
    use crate::progress::{InferenceCancellationToken, InferenceCancelled, ProgressReporter};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn write_fixture(path: &Path, size: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, vec![b'x'; size]).unwrap();
    }

    fn fixture_config(root: &Path) -> mold_core::Config {
        let config = mold_core::Config {
            models_dir: root.display().to_string(),
            ..mold_core::Config::default()
        };
        write_fixture(&root.join("shared/qwen3-expand/tokenizer.json"), 128);
        config
    }

    fn fixture_plan(root: &Path, placement: ExactExpandPlacement) -> ResolvedExpandExecutionPlan {
        let config = fixture_config(root);
        write_fixture(&root.join("qwen3-expand-q4/model-Q4_K_M.gguf"), 1_024);
        resolve_expand_execution_plan(&config, Some("qwen3-expand:q4"), placement).unwrap()
    }

    #[test]
    fn artifact_resolution_and_memory_floors_follow_the_selected_quantization() {
        let root = tempfile::tempdir().unwrap();
        let config = fixture_config(root.path());
        write_fixture(
            &root.path().join("qwen3-expand-q4/model-Q4_K_M.gguf"),
            1_024,
        );
        write_fixture(&root.path().join("qwen3-expand-q8/model-Q8_0.gguf"), 4_096);

        let q4 = resolve_expand_execution_plan(
            &config,
            Some("qwen3-expand:q4"),
            ExactExpandPlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 0,
            },
        )
        .unwrap();
        let q8 = resolve_expand_execution_plan(
            &config,
            Some("qwen3-expand:q8"),
            ExactExpandPlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 0,
            },
        )
        .unwrap();

        assert_eq!(q4.artifacts.model.size_bytes, 1_024);
        assert_eq!(q8.artifacts.model.size_bytes, 4_096);
        assert_eq!(
            q4.predicted_vram_peak_bytes,
            1_024 + crate::device::EXPAND_ACTIVATION_HEADROOM
        );
        assert_eq!(
            q8.predicted_vram_peak_bytes,
            4_096 + crate::device::EXPAND_ACTIVATION_HEADROOM
        );
        assert!(q8.predicted_host_increment_bytes > q4.predicted_host_increment_bytes);
        assert_ne!(
            q4.artifacts.model.identity_fingerprint,
            q8.artifacts.model.identity_fingerprint
        );
        assert_ne!(q4.execution_fingerprint, q8.execution_fingerprint);
    }

    #[test]
    fn exact_gpu_placement_ignores_mold_device_cpu_and_preserves_ordinal() {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _env = ENV_LOCK.lock().unwrap();
        let _restore = EnvVarGuard::set("MOLD_DEVICE", "cpu");

        let root = tempfile::tempdir().unwrap();
        let plan = fixture_plan(
            root.path(),
            ExactExpandPlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 7,
            },
        );
        let reporter = ProgressReporter::default();
        let observed = std::sync::Mutex::new(None);
        let device = create_exact_device_with(&plan, &reporter, |backend, ordinal, _| {
            *observed.lock().unwrap() = Some((backend, ordinal));
            Ok(Device::Cpu)
        })
        .unwrap();

        assert!(
            device.is_cpu(),
            "test factory returns a sentinel CPU device"
        );
        assert_eq!(
            *observed.lock().unwrap(),
            Some((mold_core::GpuBackend::Cuda, 7))
        );
    }

    #[test]
    fn exact_cpu_plan_never_calls_gpu_device_factory() {
        let root = tempfile::tempdir().unwrap();
        let plan = fixture_plan(root.path(), ExactExpandPlacement::Cpu);
        let reporter = ProgressReporter::default();
        let calls = AtomicUsize::new(0);

        let device = create_exact_device_with(&plan, &reporter, |_, _, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("GPU factory must not run for CPU placement")
        })
        .unwrap();

        assert!(device.is_cpu());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn exact_reporting_never_probes_generic_gpu_memory() {
        let root = tempfile::tempdir().unwrap();
        for placement in [
            ExactExpandPlacement::Cpu,
            ExactExpandPlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 7,
            },
        ] {
            let plan = fixture_plan(root.path(), placement);
            let reporter = ProgressReporter::default();
            let calls = AtomicUsize::new(0);

            report_expand_memory_status_with(Some(&plan), &reporter, || {
                calls.fetch_add(1, Ordering::SeqCst);
                Some("must not be queried".to_string())
            });

            assert_eq!(calls.load(Ordering::SeqCst), 0, "placement {placement:?}");
        }
    }

    #[test]
    fn artifact_drift_invalidates_before_device_construction() {
        let root = tempfile::tempdir().unwrap();
        let plan = fixture_plan(
            root.path(),
            ExactExpandPlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 3,
            },
        );
        std::fs::write(&plan.artifacts.model.path, b"replacement").unwrap();
        let reporter = ProgressReporter::default();
        let calls = AtomicUsize::new(0);

        let error = create_exact_device_with(&plan, &reporter, |_, _, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Device::Cpu)
        })
        .unwrap_err()
        .to_string();

        assert!(error.contains("artifact changed"), "{error}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn execution_plan_mutation_is_rejected_before_device_construction() {
        let root = tempfile::tempdir().unwrap();
        let mut plan = fixture_plan(
            root.path(),
            ExactExpandPlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 3,
            },
        );
        plan.placement = ExactExpandPlacement::Device {
            backend: mold_core::GpuBackend::Cuda,
            ordinal: 4,
        };
        let reporter = ProgressReporter::default();
        let calls = AtomicUsize::new(0);

        let error = create_exact_device_with(&plan, &reporter, |_, _, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Device::Cpu)
        })
        .unwrap_err()
        .to_string();

        assert!(error.contains("execution plan changed"), "{error}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_gpu_factory_error_is_returned_without_cpu_fallback() {
        let root = tempfile::tempdir().unwrap();
        let plan = fixture_plan(
            root.path(),
            ExactExpandPlacement::Device {
                backend: mold_core::GpuBackend::Cuda,
                ordinal: 5,
            },
        );
        let reporter = ProgressReporter::default();
        let calls = AtomicUsize::new(0);

        let error = create_exact_device_with(&plan, &reporter, |backend, ordinal, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("failed after touching {backend:?}:{ordinal}")
        })
        .unwrap_err()
        .to_string();

        assert!(error.contains("failed after touching Cuda:5"), "{error}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn every_named_expansion_safe_point_observes_cancellation() {
        for checkpoint in [
            ExpandSafePoint::BeforeLoad,
            ExpandSafePoint::AfterLoad,
            ExpandSafePoint::AfterTokenization,
            ExpandSafePoint::BeforePromptForward,
            ExpandSafePoint::TokenIteration,
            ExpandSafePoint::BeforeDecode,
        ] {
            let token = InferenceCancellationToken::default();
            token.cancel();
            let mut reporter = ProgressReporter::default();
            reporter.set_cancellation_token(token);
            assert_eq!(
                expansion_checkpoint(&reporter, checkpoint)
                    .unwrap_err()
                    .downcast_ref::<InferenceCancelled>(),
                Some(&InferenceCancelled),
                "checkpoint {checkpoint:?} did not preserve typed cancellation"
            );
        }
    }

    #[test]
    fn cancelled_exact_attempt_clears_token_before_reuse() {
        let root = tempfile::tempdir().unwrap();
        let plan = fixture_plan(root.path(), ExactExpandPlacement::Cpu);
        let mut expander = LocalExpander::from_resolved_plan(plan);
        let token = InferenceCancellationToken::default();
        token.cancel();

        let error = expander
            .expand_with_cancellation("a cat", &ExpandConfig::default(), token)
            .unwrap_err();

        assert!(crate::is_inference_cancelled(&error));
        assert!(
            expander.progress.checkpoint().is_ok(),
            "cancelled token leaked into the next attempt"
        );
    }
}
