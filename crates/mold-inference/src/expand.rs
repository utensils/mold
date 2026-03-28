//! Local LLM-powered prompt expansion using quantized Qwen3 GGUF models.
//!
//! Loads a small Qwen3 model (1.7B or 0.6B at Q4), generates expanded prompts,
//! then drops the weights to free VRAM for the diffusion pipeline.

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use std::io::Seek;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use mold_core::expand::{ExpandConfig, ExpandResult, PromptExpander};
use mold_core::expand_prompts::{build_batch_messages, build_single_messages, format_chatml};

use crate::device::{
    create_device, free_vram_bytes, memory_status_string, preflight_memory_check, should_use_gpu,
};
use crate::progress::{ProgressCallback, ProgressReporter};

/// VRAM threshold for placing the expand LLM on GPU (Q4 1.7B ~1.3GB + headroom).
const EXPAND_LLM_VRAM_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024; // 2 GB

/// Local prompt expander using quantized Qwen3 GGUF.
pub struct LocalExpander {
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    progress: ProgressReporter,
}

impl LocalExpander {
    /// Create a new local expander with paths to the GGUF model and tokenizer.
    pub fn new(model_path: impl Into<PathBuf>, tokenizer_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tokenizer_path: tokenizer_path.into(),
            progress: ProgressReporter::default(),
        }
    }

    /// Set a progress callback for reporting device selection, loading, and generation status.
    pub fn set_on_progress(&mut self, callback: ProgressCallback) {
        self.progress.set_callback(callback);
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
        let models_dir = config.resolved_models_dir();
        let expand_model = expand_model.unwrap_or(&config.expand.model);

        // Determine the tag from the model spec (e.g., "qwen3-expand:q4" → "q4")
        let tag = expand_model.split(':').nth(1).unwrap_or("q8");

        // Search model-specific directories for GGUF files.
        // Manifest storage places transformers under <model-name>/ with colons
        // replaced by dashes, e.g., "qwen3-expand-q8/" or "qwen3-expand-small-q8/".
        // Order candidates so the explicitly requested variant is checked first —
        // otherwise if both qwen3-expand and qwen3-expand-small are installed,
        // the larger model would always win regardless of user's choice.
        let candidate_dirs = if expand_model.contains("small") {
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
        };

        let mut gguf_path = None;
        for dir_name in &candidate_dirs {
            let dir = models_dir.join(dir_name);
            if dir.exists() {
                if let Some(path) = find_gguf_in_dir(&dir, expand_model) {
                    gguf_path = Some(path);
                    break;
                }
            }
        }
        let gguf_path = gguf_path?;

        // Search for tokenizer: shared location first, then model-specific dirs
        let shared_tokenizer = models_dir.join("shared/qwen3-expand/tokenizer.json");
        let tokenizer_path = if shared_tokenizer.exists() {
            shared_tokenizer
        } else {
            // Search candidate dirs for tokenizer
            let mut found = None;
            for dir_name in &candidate_dirs {
                let dir = models_dir.join(dir_name);
                if let Some(path) = find_tokenizer_in_dir(&dir) {
                    found = Some(path);
                    break;
                }
            }
            found?
        };

        Some(Self::new(gguf_path, tokenizer_path))
    }

    /// Load model, generate text, drop model.
    fn generate_text(&self, prompt_text: &str, config: &ExpandConfig) -> Result<String> {
        // Device selection: Metal always uses GPU (unified memory), CUDA checks VRAM
        let gpu_device = create_device(&self.progress)?;
        let is_cuda = gpu_device.is_cuda();
        let is_metal = gpu_device.is_metal();
        let free_vram = free_vram_bytes().unwrap_or(0);

        let device = if should_use_gpu(is_cuda, is_metal, free_vram, EXPAND_LLM_VRAM_THRESHOLD) {
            gpu_device
        } else {
            self.progress
                .info("Using CPU for prompt expansion (insufficient GPU memory)");
            Device::Cpu
        };

        let device_label = if device.is_metal() {
            "Metal"
        } else if device.is_cuda() {
            "CUDA"
        } else {
            "CPU"
        };

        // Report memory status
        if let Some(mem_status) = memory_status_string() {
            self.progress.info(&mem_status);
        }

        // Preflight memory check (Darwin safety guard — prevents OOM from page reclamation storms)
        let model_size = std::fs::metadata(&self.model_path)
            .map(|m| m.len())
            .unwrap_or(0);
        preflight_memory_check("Expand LLM", model_size)?;

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

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&self.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load expand tokenizer: {e}"))?;

        self.progress.stage_done(&stage_name, load_start.elapsed());

        // Tokenize prompt
        let encoding = tokenizer
            .encode(prompt_text, false)
            .map_err(|e| anyhow::anyhow!("failed to tokenize expand prompt: {e}"))?;
        let input_ids = encoding.get_ids();

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
        let mut offset = {
            let input = Tensor::new(input_ids, &device)?.unsqueeze(0)?;
            let _logits = model.forward(&input, 0)?;
            input_ids.len()
        };

        // Generate new tokens one at a time
        let mut last_token = *input_ids.last().unwrap_or(&0);
        for _ in 0..max_new_tokens {
            let input = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, offset)?;
            offset += 1;

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
        let output = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("failed to decode generated tokens: {e}"))?;

        // Clear KV cache and drop model (RAII handles the rest)
        model.clear_kv_cache();
        drop(model);

        // Reclaim GPU memory if we used it
        if device.is_cuda() {
            let _ = crate::device::reclaim_gpu_memory();
        }

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
            )
        } else {
            build_single_messages(
                prompt,
                &config.model_family,
                config.system_prompt.as_deref(),
                family_override,
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
    let entries: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "gguf")
                .unwrap_or(false)
        })
        .collect();

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
        for entry in entries.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname.contains("tokenizer") && fname.ends_with(".json") {
                return Some(entry.path());
            }
        }
    }
    None
}
