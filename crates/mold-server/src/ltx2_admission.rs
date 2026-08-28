//! LTX-2 admission memory model.
//!
//! The generic streaming-transformer estimate in [`crate::memory_preflight`]
//! collapses every LTX-2 checkpoint into one flat cap. That was wrong by ~10x
//! for the 19B FP8 preset: the checkpoint carries 2.1 GB of *non-block*
//! transformer weights plus a 2.4 GB video VAE that the block-streaming path
//! never gets to offload, and the engine's adaptive residency planner then
//! keeps as many transformer blocks resident as the sampled free VRAM allows.
//!
//! This module reconstructs that plan at admission time from the checkpoint's
//! own safetensors header so the scheduler predicts the peak the engine will
//! actually reach, rejects a shape that cannot be run before two minutes of
//! loading, and can name a shape that does fit.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use mold_core::ltx2_weight_index::{Ltx2ResidentWeightForm, Ltx2TransformerWeightIndex};

/// Mirrors `mold_inference::adaptive_offload::ADAPTIVE_OFFLOAD_RUNTIME_HEADROOM`;
/// the engine reserves it inside every residency plan.
pub(crate) const LTX2_RUNTIME_HEADROOM_BYTES: u64 = 2_000_000_000;

/// Floor for the fragmentation margin. Block streaming churns 0.4–0.8 GB
/// allocations through the CUDA allocator on every denoise step, so the plan
/// must not be sized to the last byte of the sampled reading.
const LTX2_MIN_FRAGMENTATION_MARGIN_BYTES: u64 = 1_000_000_000;

/// Share of the sampled reading used as the fragmentation margin when that is
/// larger than the floor (5%).
const LTX2_FRAGMENTATION_MARGIN_PERCENT: u64 = 5;

/// Bytes per element admission prices GPU residency at. Every accelerator
/// backend runs LTX-2 in BF16 (`ltx2::backend::Ltx2Backend::compute_dtype`),
/// so a dense or narrowed tensor costs two bytes per element on the device.
const LTX2_ADMISSION_ELEMENT_BYTES: u64 = 2;

/// The same 90% safety cap `check_model_memory_budget` enforces. Admission
/// plans residency against the budget it is willing to grant, never against
/// the whole card — otherwise the predicted peak is unconditionally the size
/// of the device and every shape looks infeasible.
const LTX2_ADMISSION_BUDGET_PERCENT: u64 = 90;

// ── Activation budget ───────────────────────────────────────────────────────

/// Peak transformer activation bytes for one LTX-2 render shape.
///
/// `mold_inference::device` owns the calibrated per-token model; admission and
/// the engine must price the same shape identically, so this is a thin
/// re-export rather than a second estimate.
pub(crate) fn ltx2_activation_budget_bytes(
    width: u32,
    height: u32,
    frames: u32,
    conditioned: bool,
    adaln_dim: Option<u64>,
) -> u64 {
    mold_inference::device::ltx2_activation_budget_bytes(
        width,
        height,
        frames,
        conditioned,
        adaln_dim,
    )
}

/// Activation budget for a request shape, priced against the checkpoint's own
/// AdaLN width when its header has been read.
pub(crate) fn ltx2_activation_bytes(shape: Ltx2ShapeHint, adaln_dim: Option<u64>) -> u64 {
    ltx2_activation_budget_bytes(
        shape.width,
        shape.height,
        shape.frames,
        shape.conditioned,
        adaln_dim,
    )
}

/// Request shape that drives the LTX-2 activation budget. `ActivationHint`
/// deliberately stays as-is — it is constructed by name across the server and
/// carries no video-specific fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ltx2ShapeHint {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
    /// Source image, keyframes, or extend present.
    pub(crate) conditioned: bool,
}

impl Ltx2ShapeHint {
    pub(crate) fn from_request(req: &mold_core::GenerateRequest) -> Self {
        Self::from_request_with_projection(req, None)
    }

    pub(crate) fn from_request_with_projection(
        req: &mold_core::GenerateRequest,
        projection: Option<&crate::queue_media_store::QueueMediaProjection>,
    ) -> Self {
        let frames = req.frames.unwrap_or_else(|| {
            if matches!(
                mold_core::ltx2_preprocess::ltx2_generation(&req.model, None),
                Some(mold_core::ltx2_preprocess::Ltx2Generation::V2_5)
            ) {
                mold_core::ltx2_duration::admission_frames(req.fps.unwrap_or(24)).unwrap_or(1)
            } else {
                1
            }
        });
        Self {
            width: req.width,
            height: req.height,
            frames,
            conditioned: req.source_image.is_some()
                || req.source_video.is_some()
                || req.source_video_path.is_some()
                || req.extend_video.is_some()
                || req.extend_video_path.is_some()
                || req
                    .keyframes
                    .as_ref()
                    .is_some_and(|keyframes| !keyframes.is_empty())
                || projection.is_some_and(|projection| {
                    projection.source_image
                        || projection.source_video_inline
                        || projection.source_video_path
                        || projection.extend_video_inline
                        || projection.extend_video_path
                        || projection.keyframe_count > 0
                }),
        }
    }
}

// ── Checkpoint facts ────────────────────────────────────────────────────────

/// Per-checkpoint weight layout, read once from the safetensors header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Ltx2CheckpointFacts {
    /// Per-transformer-block byte sizes, ordered by block index.
    pub(crate) block_sizes: Vec<u64>,
    /// Non-block transformer weights (`patchify_proj`, `adaln_single.linear`,
    /// `caption_projection`, connectors, `proj_out`, norms). These are always
    /// GPU-resident — block streaming never offloads them.
    pub(crate) fixed_resident_bytes: u64,
    /// Bundled video VAE weights, resident for encode and decode while the
    /// transformer's resident blocks are still held.
    pub(crate) vae_bytes: u64,
    /// The checkpoint's `adaln_single.linear` output width, which sets the
    /// per-token AdaLN cost of a conditioned render. The 19B ships six
    /// components (24,576); LTX-2.3's 22B ships nine (36,864).
    pub(crate) adaln_dim: Option<u64>,
    /// Per-forward scratch beside the resident weights: one dequantized
    /// linear for a quantized checkpoint, zero for dense/float8. Carried from
    /// the shared index so the engine and admission reserve one figure.
    pub(crate) transient_bytes: u64,
    /// Blocks are resident packed (INT8 ConvRot on CUDA), so every forward
    /// also needs the token-scaled W8A8 workspace beside the activations.
    pub(crate) int8_packed: bool,
}

impl Ltx2CheckpointFacts {
    /// Project the shared header index onto admission's residency model.
    /// The same index feeds `mold_inference::ltx2::ltx2_transformer_weight_sizes`,
    /// and a parity test pins the two projections to each other.
    pub(crate) fn from_weight_index(index: &Ltx2TransformerWeightIndex) -> Self {
        // This estimator models the CUDA adaptive planner — the only backend
        // that pages blocks against a measured budget — so ConvRot blocks
        // are priced in the packed form CUDA actually keeps resident
        // (`LtxLinear::ConvRotPacked`). Metal admission rides the unified
        // single gate and never reaches this prefix search.
        let form = Ltx2ResidentWeightForm::for_convrot_backend(true);
        Self {
            block_sizes: index.resident_block_bytes_for(LTX2_ADMISSION_ELEMENT_BYTES, form),
            fixed_resident_bytes: index.resident_non_block_bytes(LTX2_ADMISSION_ELEMENT_BYTES),
            vae_bytes: index.vae_bytes_at_rest(),
            adaln_dim: index.adaln_dim(),
            transient_bytes: index.transient_bytes(),
            int8_packed: index.is_convrot() && form == Ltx2ResidentWeightForm::Packed,
        }
    }

    /// The activation-side reserve for one shape: the shared per-token budget
    /// plus, for packed-resident INT8 ConvRot, the W8A8 forward workspace.
    /// One method so `ltx2_shape_fits` and the preflight estimate cannot
    /// diverge on what "activation" includes.
    pub(crate) fn activation_bytes(&self, shape: Ltx2ShapeHint) -> u64 {
        let base = ltx2_activation_bytes(shape, self.adaln_dim);
        if self.int8_packed {
            base.saturating_add(
                mold_core::ltx2_weight_index::ltx2_int8_w8a8_workspace_bytes(
                    mold_inference::device::ltx2_token_count(
                        shape.width,
                        shape.height,
                        shape.frames,
                    ),
                ),
            )
        } else {
            base
        }
    }

    pub(crate) fn total_block_bytes(&self) -> u64 {
        self.block_sizes.iter().copied().sum()
    }

    fn is_usable(&self) -> bool {
        !self.block_sizes.is_empty() && self.total_block_bytes() > 0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CacheKey {
    len: u64,
    modified_ns: u128,
}

type FactsCache = RwLock<HashMap<PathBuf, (CacheKey, Arc<Ltx2CheckpointFacts>)>>;

fn facts_cache() -> &'static FactsCache {
    static CACHE: std::sync::OnceLock<FactsCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}

fn cache_key(path: &Path) -> Option<CacheKey> {
    let metadata = std::fs::metadata(path).ok()?;
    let modified_ns = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|since| since.as_nanos())
        .unwrap_or(0);
    Some(CacheKey {
        len: metadata.len(),
        modified_ns,
    })
}

/// Cache-only lookup. Never touches the filesystem body and never parses, so
/// it is safe on the scheduler coordinator thread and for
/// `equivalence_cache_only` prepared work.
pub(crate) fn checkpoint_facts_cached(path: &Path) -> Option<Arc<Ltx2CheckpointFacts>> {
    let cache = facts_cache().read().ok()?;
    cache.get(path).map(|(_, facts)| Arc::clone(facts))
}

/// Parse and cache one checkpoint's weight layout.
///
/// Blocking: this reads and JSON-parses the safetensors header. Call it from
/// `spawn_blocking` or a worker thread, never from the coordinator.
pub(crate) fn warm_checkpoint_facts(path: &Path) -> Option<Arc<Ltx2CheckpointFacts>> {
    let key = cache_key(path)?;
    if let Ok(cache) = facts_cache().read() {
        if let Some((cached_key, facts)) = cache.get(path) {
            if cached_key == &key {
                return Some(Arc::clone(facts));
            }
        }
    }

    let facts = match parse_checkpoint_facts(path) {
        Ok(facts) if facts.is_usable() => Arc::new(facts),
        Ok(_) => return None,
        Err(error) => {
            tracing::debug!(
                path = %path.display(),
                "LTX-2 admission could not read checkpoint weight layout: {error}"
            );
            return None;
        }
    };
    if let Ok(mut cache) = facts_cache().write() {
        cache.insert(path.to_path_buf(), (key, Arc::clone(&facts)));
    }
    Some(facts)
}

/// Read the checkpoint header (safetensors or GGUF) through the shared
/// weight index and project it onto per-block transformer weights,
/// always-resident transformer weights, and VAE weights. Only the header is
/// read — the tensor bodies are never touched.
pub(crate) fn parse_checkpoint_facts(path: &Path) -> anyhow::Result<Ltx2CheckpointFacts> {
    let index = Ltx2TransformerWeightIndex::read(path)?;
    Ok(Ltx2CheckpointFacts::from_weight_index(&index))
}

// ── Peak estimate ───────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Ltx2PeakEstimate {
    /// Predicted GPU peak for the plan the engine will actually run.
    pub(crate) peak_bytes: u64,
    /// Everything that is resident before a single transformer block lands.
    pub(crate) reserve_bytes: u64,
    pub(crate) resident_block_bytes: u64,
    pub(crate) streamed_block_bytes: u64,
    pub(crate) largest_streamed_block_bytes: u64,
    pub(crate) fragmentation_margin_bytes: u64,
    /// False when the shape cannot be run at an admissible residency on this
    /// budget. `peak_bytes` then reports the undegraded demand so the caller's
    /// feasibility comparison rejects it.
    pub(crate) viable: bool,
}

pub(crate) fn fragmentation_margin_bytes(available_bytes: u64) -> u64 {
    (available_bytes / 100)
        .saturating_mul(LTX2_FRAGMENTATION_MARGIN_PERCENT)
        .max(LTX2_MIN_FRAGMENTATION_MARGIN_BYTES)
}

/// Predict the LTX-2 peak for one shape against one memory budget.
///
/// This reconstructs `plan_adaptive_residency`'s choice — maximize resident
/// block bytes subject to `reserve + resident + largest_streamed_block` fitting
/// — but against the budget admission is willing to grant rather than the raw
/// sampled free reading, and with the two terms the engine never counted: the
/// non-block transformer weights and the bundled VAE.
pub(crate) fn ltx2_peak_estimate(
    facts: &Ltx2CheckpointFacts,
    activation_bytes: u64,
    available_bytes: u64,
) -> Ltx2PeakEstimate {
    let fragmentation_margin_bytes = fragmentation_margin_bytes(available_bytes);
    let reserve_bytes = facts
        .fixed_resident_bytes
        .saturating_add(facts.vae_bytes)
        .saturating_add(facts.transient_bytes)
        .saturating_add(activation_bytes)
        .saturating_add(LTX2_RUNTIME_HEADROOM_BYTES)
        .saturating_add(fragmentation_margin_bytes);
    let total_block_bytes = facts.total_block_bytes();
    let budget = available_bytes / 100 * LTX2_ADMISSION_BUDGET_PERCENT;

    let mut sorted = facts.block_sizes.clone();
    sorted.sort_unstable_by(|left, right| right.cmp(left));

    // `reserve + prefix(k) + sorted[k]` is non-decreasing in k, so the first
    // prefix that does not fit ends the search.
    let mut resident = 0u64;
    let mut chosen: Option<(u64, u64)> = None;
    for index in 0..=sorted.len() {
        let largest_streamed = sorted.get(index).copied().unwrap_or(0);
        if reserve_bytes
            .saturating_add(resident)
            .saturating_add(largest_streamed)
            > budget
        {
            break;
        }
        chosen = Some((resident, largest_streamed));
        if let Some(size) = sorted.get(index) {
            resident = resident.saturating_add(*size);
        }
    }

    let Some((resident_block_bytes, largest_streamed_block_bytes)) = chosen else {
        // Not even a fully streamed plan fits: the reserve alone is over
        // budget. Report the streaming floor, which is the smallest peak this
        // shape can possibly reach.
        let largest_block = sorted.first().copied().unwrap_or(0);
        return Ltx2PeakEstimate {
            peak_bytes: reserve_bytes.saturating_add(largest_block),
            reserve_bytes,
            resident_block_bytes: 0,
            streamed_block_bytes: total_block_bytes,
            largest_streamed_block_bytes: largest_block,
            fragmentation_margin_bytes,
            viable: false,
        };
    };

    // The prefix search only ever accepts a plan that fits the budget, so
    // reaching here means the shape is runnable. Streaming a large share of
    // the transformer is the *intended* behaviour of adaptive offload, not a
    // failure mode: it trades wall-clock for residency so a 19B checkpoint fits
    // a consumer card at all. Judging feasibility by a streamed-to-resident
    // ratio would reject exactly the workload offloading exists to serve.
    let streamed_block_bytes = total_block_bytes.saturating_sub(resident_block_bytes);
    let peak_bytes = reserve_bytes
        .saturating_add(resident_block_bytes)
        .saturating_add(largest_streamed_block_bytes);

    Ltx2PeakEstimate {
        peak_bytes,
        reserve_bytes,
        resident_block_bytes,
        streamed_block_bytes,
        largest_streamed_block_bytes,
        fragmentation_margin_bytes,
        viable: true,
    }
}

/// Whether a shape is admissible on this budget.
pub(crate) fn ltx2_shape_fits(
    facts: &Ltx2CheckpointFacts,
    shape: Ltx2ShapeHint,
    available_bytes: u64,
) -> bool {
    let estimate = ltx2_peak_estimate(facts, facts.activation_bytes(shape), available_bytes);
    estimate.viable && estimate.peak_bytes <= available_bytes / 100 * LTX2_ADMISSION_BUDGET_PERCENT
}

// ── Supported-shape search (#641 actionable rejection) ──────────────────────

/// LTX-2 frame counts live on the `8n+1` grid.
fn frame_grid(max_frames: u32) -> Vec<u32> {
    let mut frames = Vec::new();
    let mut candidate = 9u32;
    while candidate <= max_frames {
        frames.push(candidate);
        candidate += 8;
    }
    frames
}

/// Resolutions are 32-aligned; step down in 128 px increments so the
/// suggestion is a shape a user would actually pick.
fn resolution_grid(width: u32, height: u32) -> Vec<(u32, u32)> {
    let mut grid = Vec::new();
    let mut scale = width.min(height);
    while scale >= 384 {
        let ratio_w = width as f64 / width.min(height) as f64;
        let ratio_h = height as f64 / width.min(height) as f64;
        let candidate_w = (((scale as f64 * ratio_w) as u32) / 32) * 32;
        let candidate_h = (((scale as f64 * ratio_h) as u32) / 32) * 32;
        if candidate_w > 0 && candidate_h > 0 {
            grid.push((candidate_w, candidate_h));
        }
        scale = scale.saturating_sub(128);
    }
    grid
}

/// One concrete shape that fits, phrased for a user-facing message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SupportedShape {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames: u32,
}

impl std::fmt::Display for SupportedShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{} at {} frames",
            self.width, self.height, self.frames
        )
    }
}

/// Re-run the estimator over the `8n+1` frame grid and the 32-aligned
/// resolution grid and return the shapes closest to what was requested:
/// the longest clip at the requested resolution, then the largest resolution
/// at the requested frame count.
pub(crate) fn supported_shapes(
    facts: &Ltx2CheckpointFacts,
    requested: Ltx2ShapeHint,
    available_bytes: u64,
) -> Vec<SupportedShape> {
    let mut shapes: Vec<SupportedShape> = Vec::new();

    let fits = |width: u32, height: u32, frames: u32| {
        ltx2_shape_fits(
            facts,
            Ltx2ShapeHint {
                width,
                height,
                frames,
                conditioned: requested.conditioned,
            },
            available_bytes,
        )
    };

    if let Some(frames) = frame_grid(requested.frames)
        .into_iter()
        .rev()
        .find(|frames| fits(requested.width, requested.height, *frames))
    {
        shapes.push(SupportedShape {
            width: requested.width,
            height: requested.height,
            frames,
        });
    }

    if let Some((width, height)) = resolution_grid(requested.width, requested.height)
        .into_iter()
        .find(|(width, height)| {
            (*width, *height) != (requested.width, requested.height)
                && fits(*width, *height, requested.frames)
        })
    {
        shapes.push(SupportedShape {
            width,
            height,
            frames: requested.frames,
        });
    }

    shapes
}

/// Human-readable "needs ~X GB on a Y GB card; <shape> fits, or <shape>"
/// remediation used by both admission rejection and the CUDA OOM message.
pub(crate) fn supported_shape_advice(
    facts: &Ltx2CheckpointFacts,
    requested: Ltx2ShapeHint,
    available_bytes: u64,
) -> Option<String> {
    let estimate = ltx2_peak_estimate(
        facts,
        ltx2_activation_bytes(requested, facts.adaln_dim),
        available_bytes,
    );
    let shapes = supported_shapes(facts, requested, available_bytes);
    if shapes.is_empty() {
        return None;
    }
    let alternatives = shapes
        .iter()
        .map(SupportedShape::to_string)
        .collect::<Vec<_>>()
        .join(", or ");
    Some(format!(
        "needs ~{:.1} GB on a {:.1} GB card; {alternatives} fits",
        estimate.peak_bytes as f64 / 1_000_000_000.0,
        available_bytes as f64 / 1_000_000_000.0,
    ))
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    /// Measured `ltx-2-19b-distilled:fp8` layout (issue #641): 6 BF16 blocks
    /// (including block 0) at 772,284,416 B, 42 FP8 blocks at 386,408,672 B,
    /// 2,107,091,456 B of non-block `model.*` weights, and a 2,444,960,482 B
    /// video VAE.
    pub(crate) fn ltx2_19b_fp8_facts() -> Ltx2CheckpointFacts {
        let mut block_sizes = vec![772_284_416u64; 6];
        block_sizes.extend(std::iter::repeat_n(386_408_672u64, 42));
        Ltx2CheckpointFacts {
            block_sizes,
            fixed_resident_bytes: 2_107_091_456,
            vae_bytes: 2_444_960_482,
            adaln_dim: Some(24_576),
            transient_bytes: 0,
            int8_packed: false,
        }
    }

    /// Write a header-only safetensors file describing `facts`. The tensor
    /// bodies are never read by the parser, so the file stays a few KB.
    pub(crate) fn write_header_only_checkpoint(path: &Path, facts: &Ltx2CheckpointFacts) {
        let mut tensors = serde_json::Map::new();
        let mut offset = 0u64;
        let push = |tensors: &mut serde_json::Map<String, serde_json::Value>,
                    name: String,
                    bytes: u64,
                    offset: &mut u64| {
            let start = *offset;
            *offset += bytes;
            tensors.insert(
                name,
                serde_json::json!({
                    "dtype": "F8_E4M3",
                    "shape": [bytes],
                    "data_offsets": [start, *offset],
                }),
            );
        };
        for (index, bytes) in facts.block_sizes.iter().enumerate() {
            push(
                &mut tensors,
                format!("model.transformer_blocks.{index}.attn1.to_q.weight"),
                *bytes,
                &mut offset,
            );
        }
        // The AdaLN table is itself a fixed-resident tensor, so the stub
        // carries the remainder — parsing this file must reproduce exactly the
        // `fixed_resident_bytes` the fixture declares.
        let adaln_bytes = facts.adaln_dim.map_or(0, |dim| dim * 4096 * 2);
        push(
            &mut tensors,
            "model.patchify_proj.weight".to_string(),
            facts.fixed_resident_bytes.saturating_sub(adaln_bytes),
            &mut offset,
        );
        push(
            &mut tensors,
            "vae.decoder.conv_out.weight".to_string(),
            facts.vae_bytes,
            &mut offset,
        );
        if let Some(adaln_dim) = facts.adaln_dim {
            // Shape carries the real `[out, in]`, unlike the byte-sized stubs
            // above — the parser reads this width, not the tensor's size.
            let start = offset;
            offset += adaln_bytes;
            tensors.insert(
                "model.adaln_single.linear.weight".to_string(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": [adaln_dim, 4096],
                    "data_offsets": [start, offset],
                }),
            );
        }

        let header = serde_json::to_vec(&serde_json::Value::Object(tensors)).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.flush().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{ltx2_19b_fp8_facts, write_header_only_checkpoint};
    use super::*;

    const RTX_4090_AVAILABLE: u64 = 25_757_220_864;

    fn incident_shape() -> Ltx2ShapeHint {
        Ltx2ShapeHint {
            width: 1024,
            height: 1024,
            frames: 97,
            conditioned: true,
        }
    }

    #[test]
    fn projected_video_conditioning_matches_hydrated_shape() {
        let hydrated: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "video",
            "model": "ltx2:fp8",
            "width": 768,
            "height": 512,
            "frames": 97,
            "steps": 20,
            "guidance": 3.0,
            "extend_video": "dmlkZW8="
        }))
        .unwrap();
        let mut sanitized = hydrated.clone();
        sanitized.extend_video = None;
        let projection = crate::queue_media_store::QueueMediaProjection {
            extend_video_inline: true,
            ..Default::default()
        };
        assert_eq!(
            Ltx2ShapeHint::from_request_with_projection(&hydrated, None),
            Ltx2ShapeHint::from_request_with_projection(&sanitized, Some(&projection)),
        );

        let mut hydrated_path = sanitized.clone();
        hydrated_path.source_video_path = Some("/private/source.mp4".into());
        let path_projection = crate::queue_media_store::QueueMediaProjection {
            source_video_path: true,
            ..Default::default()
        };
        assert_eq!(
            Ltx2ShapeHint::from_request_with_projection(&hydrated_path, None),
            Ltx2ShapeHint::from_request_with_projection(&sanitized, Some(&path_projection)),
        );
    }

    #[test]
    fn ltx25_auto_duration_reserves_the_shared_maximum() {
        let automatic: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a slow establishing shot",
            "model": "ltx-2.5-22b-distilled:int8-conv",
            "width": 768,
            "height": 512,
            "steps": 8,
            "guidance": 0.0,
            "fps": 24
        }))
        .unwrap();
        let mut explicit = automatic.clone();
        explicit.frames = Some(97);

        assert_eq!(Ltx2ShapeHint::from_request(&automatic).frames, 473);
        assert_eq!(Ltx2ShapeHint::from_request(&explicit).frames, 97);
    }

    /// LTX-2.3's 22B ships nine AdaLN components (`[36864, 4096]`) where the
    /// 19B ships six (`[24576, 4096]`) — measured from both checkpoints'
    /// headers. Admission must price a conditioned render against the
    /// checkpoint it is actually going to run, not the 19B's width.
    #[test]
    fn checkpoint_facts_read_the_adaln_width_and_price_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut wide = ltx2_19b_fp8_facts();
        wide.adaln_dim = Some(36_864);
        let path = dir.path().join("ltx-2.3-22b-distilled-fp8.safetensors");
        write_header_only_checkpoint(&path, &wide);

        let parsed = parse_checkpoint_facts(&path).unwrap();
        assert_eq!(
            parsed.adaln_dim,
            Some(36_864),
            "the parser must read the checkpoint's own AdaLN width"
        );

        let six = ltx2_activation_bytes(incident_shape(), Some(24_576));
        let nine = ltx2_activation_bytes(incident_shape(), parsed.adaln_dim);
        assert_eq!(
            nine - six,
            327_155_712,
            "the three extra components must be reserved, not assumed away"
        );
    }

    /// The two-stage Distilled pipeline renders stage 1 at 512² (3,328 tokens)
    /// and stage 2 at the full 1024² (13,312 tokens).
    #[test]
    fn token_count_matches_the_measured_two_stage_shapes() {
        assert_eq!(
            mold_inference::device::ltx2_token_count(1024, 1024, 97),
            13_312
        );
        assert_eq!(
            mold_inference::device::ltx2_token_count(512, 512, 97),
            3_328
        );
    }

    fn golden(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mold-core/testdata/ltx25")
            .join(name)
    }

    /// Admission and the engine must price one checkpoint identically:
    /// block sizes, the fixed reserve, the AdaLN width, and the per-forward
    /// transient all come from the same header index, and this pins the two
    /// projections to each other on both golden 2.5 headers.
    #[test]
    fn admission_and_runtime_derive_identical_facts_from_one_header() {
        for name in [
            "distilled-int8-convrot.header.safetensors",
            "distilled-q4-k-m.header.gguf",
        ] {
            let path = golden(name);
            let index = Ltx2TransformerWeightIndex::read(&path).unwrap();
            let admission = parse_checkpoint_facts(&path).unwrap();
            // Admission prices for the CUDA planner, so parity is against
            // the same packed-for-CUDA form the engine selects there.
            let runtime = mold_inference::ltx2::ltx2_transformer_weight_sizes(
                &index,
                index.num_layers(),
                candle_core::DType::BF16,
                Ltx2ResidentWeightForm::for_convrot_backend(true),
            )
            .unwrap();

            assert_eq!(
                admission.block_sizes,
                runtime
                    .blocks
                    .iter()
                    .map(|bytes| *bytes as u64)
                    .collect::<Vec<_>>(),
                "{name}: block sizes"
            );
            assert_eq!(
                admission.fixed_resident_bytes, runtime.non_block_bytes,
                "{name}: fixed reserve"
            );
            assert_eq!(admission.adaln_dim, runtime.adaln_dim, "{name}: adaln");
            assert_eq!(
                admission.transient_bytes, runtime.transient_bytes,
                "{name}: transient"
            );
            assert_eq!(admission.adaln_dim, Some(36_864), "{name}");
            assert_eq!(admission.block_sizes.len(), 48, "{name}");
            assert_eq!(admission.vae_bytes, 0, "{name}: split packs bundle no VAE");
        }
    }

    /// The int8-conv pack widens every I8 linear to BF16 on the device, so
    /// Admission models the CUDA adaptive planner, and CUDA keeps ConvRot
    /// blocks packed — the widened figure is what the old rule double-charged.
    /// and admitted plans that OOMed at the first denoise step.
    #[test]
    fn checkpoint_facts_price_int8_convrot_packed_for_cuda() {
        let facts =
            parse_checkpoint_facts(&golden("distilled-int8-convrot.header.safetensors")).unwrap();
        // Blocks are priced in the packed form the CUDA planner keeps
        // resident (`LtxLinear::ConvRotPacked`): the U8 weight plus its F32
        // row scales, with dense block tensors at BF16. The widened figure
        // (773,349,760) survives as Metal's streaming transient, not as
        // residency.
        assert_eq!(facts.block_sizes[0], 387_867_008);
        assert_eq!(facts.block_sizes[47], 387_867_008);
        // Non-block weights widen on every backend — the quantized ones are
        // the prompt encoder's connectors.
        assert_eq!(facts.fixed_resident_bytes, 4_887_262_720);
        assert_eq!(facts.transient_bytes, 134_217_728);
        assert!(facts.int8_packed);
        assert!(facts.is_usable());

        let gguf = parse_checkpoint_facts(&golden("distilled-q4-k-m.header.gguf")).unwrap();
        assert_eq!(gguf.block_sizes[0], 249_164_160);
        assert_eq!(gguf.total_block_bytes(), 12_603_951_104);
        assert_eq!(gguf.transient_bytes, 402_653_184);
        assert!(gguf.is_usable());
    }

    #[test]
    fn checkpoint_facts_separate_blocks_from_resident_weights_and_vae() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ltx2-19b-distilled-fp8.safetensors");
        let expected = ltx2_19b_fp8_facts();
        write_header_only_checkpoint(&path, &expected);

        let parsed = parse_checkpoint_facts(&path).unwrap();
        assert_eq!(parsed.block_sizes.len(), 48);
        assert_eq!(parsed.total_block_bytes(), 20_862_870_720);
        assert_eq!(parsed.fixed_resident_bytes, 2_107_091_456);
        assert_eq!(parsed.vae_bytes, 2_444_960_482);
    }

    #[test]
    fn checkpoint_facts_are_cached_per_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cached.safetensors");
        write_header_only_checkpoint(&path, &ltx2_19b_fp8_facts());

        assert!(checkpoint_facts_cached(&path).is_none());
        let warmed = warm_checkpoint_facts(&path).unwrap();
        let cached = checkpoint_facts_cached(&path).unwrap();
        assert!(Arc::ptr_eq(&warmed, &cached));
    }

    #[test]
    fn peak_counts_non_block_weights_vae_and_resident_blocks() {
        let facts = ltx2_19b_fp8_facts();
        let estimate = ltx2_peak_estimate(
            &facts,
            ltx2_activation_bytes(incident_shape(), facts.adaln_dim),
            RTX_4090_AVAILABLE,
        );

        // The old flat LTX-2 arm returned 8 GB regardless of shape, and the
        // scheduler predicted 11,548,381,184 B for this exact request.
        assert!(
            estimate.peak_bytes > 20_000_000_000,
            "predicted peak {} must account for the transformer's block bytes",
            estimate.peak_bytes
        );
        assert!(estimate.reserve_bytes > facts.fixed_resident_bytes + facts.vae_bytes);
        assert!(estimate.fragmentation_margin_bytes >= 1_000_000_000);
    }

    /// Issue #641's headline requirement: the incident shape must *run* on a
    /// 24 GB card, not be rejected. Streaming more blocks is exactly how
    /// adaptive offload absorbs a shape whose weights do not fit — upstream's
    /// own answer for a consumer card is `--offload cpu`. Feasibility is
    /// therefore "does the fully-streamed floor fit", never a ratio of
    /// streamed to resident bytes.
    #[test]
    fn incident_shape_is_admissible_on_a_24gb_card_by_streaming() {
        let facts = ltx2_19b_fp8_facts();
        let estimate = ltx2_peak_estimate(
            &facts,
            ltx2_activation_bytes(incident_shape(), facts.adaln_dim),
            RTX_4090_AVAILABLE,
        );

        assert!(
            ltx2_shape_fits(&facts, incident_shape(), RTX_4090_AVAILABLE),
            "1024x1024 x 97 must be admitted on a 24 GB card, got peak {}",
            estimate.peak_bytes
        );
        assert!(
            estimate.streamed_block_bytes > 0,
            "the plan must stream to fit; resident {} / streamed {}",
            estimate.resident_block_bytes,
            estimate.streamed_block_bytes
        );
        // The predicted peak is the plan's own peak and never exceeds the
        // budget the scheduler is willing to grant.
        let budget = RTX_4090_AVAILABLE / 100 * LTX2_ADMISSION_BUDGET_PERCENT;
        assert!(
            estimate.peak_bytes <= budget,
            "peak {} must fit the {budget}-byte admission budget",
            estimate.peak_bytes
        );
    }

    /// A shape whose *reserve alone* clears the budget is genuinely
    /// unrunnable — no amount of streaming recovers it — and must be rejected
    /// before the engine spends two minutes loading.
    #[test]
    fn a_shape_whose_streaming_floor_overflows_is_rejected() {
        let facts = ltx2_19b_fp8_facts();
        let huge = Ltx2ShapeHint {
            width: 2048,
            height: 2048,
            frames: 193,
            ..incident_shape()
        };
        assert!(
            !ltx2_shape_fits(&facts, huge, RTX_4090_AVAILABLE),
            "a shape past the streaming floor must be refused at admission"
        );
    }

    #[test]
    fn the_same_checkpoint_still_fits_a_smaller_shape() {
        let facts = ltx2_19b_fp8_facts();
        let shorter = Ltx2ShapeHint {
            frames: 49,
            ..incident_shape()
        };
        assert!(
            ltx2_shape_fits(&facts, shorter, RTX_4090_AVAILABLE),
            "shortening the clip must recover a runnable plan"
        );
    }

    #[test]
    fn a_large_card_keeps_the_whole_transformer_resident() {
        let facts = ltx2_19b_fp8_facts();
        let estimate = ltx2_peak_estimate(
            &facts,
            ltx2_activation_bytes(incident_shape(), facts.adaln_dim),
            80_000_000_000,
        );
        assert!(estimate.viable);
        assert_eq!(estimate.resident_block_bytes, facts.total_block_bytes());
        assert_eq!(estimate.streamed_block_bytes, 0);
    }

    #[test]
    fn advice_names_a_concrete_supported_shape() {
        let facts = ltx2_19b_fp8_facts();
        let advice = supported_shape_advice(&facts, incident_shape(), RTX_4090_AVAILABLE)
            .expect("a 24 GB card must have some runnable LTX-2 shape");

        assert!(advice.contains(" GB card"), "got: {advice}");
        let shapes = supported_shapes(&facts, incident_shape(), RTX_4090_AVAILABLE);
        assert!(!shapes.is_empty());
        for shape in shapes {
            assert!(shape.width % 32 == 0 && shape.height % 32 == 0);
            assert_eq!(shape.frames % 8, 1);
            assert!(advice.contains(&shape.to_string()), "got: {advice}");
        }
    }

    #[test]
    fn frame_and_resolution_grids_stay_on_supported_values() {
        assert_eq!(frame_grid(25), vec![9, 17, 25]);
        assert!(frame_grid(97).iter().all(|frames| frames % 8 == 1));
        for (width, height) in resolution_grid(1024, 576) {
            assert_eq!(width % 32, 0);
            assert_eq!(height % 32, 0);
        }
    }
}
