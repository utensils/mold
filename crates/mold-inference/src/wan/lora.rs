//! LoRA support for the Wan DiT.
//!
//! The tier this exists for is lightx2v's 4-step Lightning distill, which ships
//! as a **pair** of rank-64 adapters — one per A14B expert. Their on-disk shape
//! (verified by parsing the shipped headers, see
//! `tmp/wan-research/report-layer6-formats.md`):
//!
//! ```text
//! diffusion_model.blocks.{0..39}.<leaf>.lora_down.weight   [64, 5120]
//! diffusion_model.blocks.{0..39}.<leaf>.lora_up.weight     [5120, 64]
//! diffusion_model.blocks.{0..39}.<leaf>.alpha              I64 scalar, value 8
//! ```
//!
//! with `<leaf>` in `self_attn.{q,k,v,o}`, `cross_attn.{q,k,v,o}`, `ffn.{0,2}`
//! — 400 modules, nothing outside the blocks.
//!
//! Two things make Wan cheap to support: the kohya `.lora_down`/`.lora_up`
//! suffixes are already in [`crate::flux::lora::classify_lora_key`], and after
//! the `diffusion_model.` prefix comes off, **the LoRA stem equals the
//! checkpoint path** — `blocks.0.self_attn.q` is exactly what
//! `WanAttention::load` reads. No rename table, unlike the DiT's
//! diffusers-format mapping.
//!
//! One thing makes it dangerous: the alpha is `I64`. See
//! [`crate::flux::lora`]'s `read_alpha_scalar` — before that fix these
//! adapters loaded at 8x strength with no error.
//!
//! An adapter reaches the weights two ways, decided by the checkpoint's
//! container rather than by the adapter. Against bf16 or fp8 safetensors,
//! [`WanLoraBackend`] merges `W + scale*(B @ A)` as each tensor is read, so the
//! merged tensor is the only copy that is ever resident. Against GGUF, nothing
//! merges: [`WanQuantizedLoraLinear`] hangs the same arithmetic off the
//! projection as a parallel low-rank branch, because taking a delta into a
//! quantized weight means requantizing it — minutes per expert, and most of the
//! delta rounded away. That type carries the measurements.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Tensor};
use mold_core::LoraWeight;

use crate::flux::lora::{classify_lora_key, LoraDirection};

/// Prefix every shipped Wan LoRA puts on its keys.
const WAN_LORA_PREFIX: &str = "diffusion_model.";

/// Suffixes that mark a Wan 2.1-era Lightning adapter, which carries
/// full-weight deltas this loader deliberately does not apply.
const FULL_DELTA_SUFFIXES: &[&str] = &[".diff", ".diff_b"];

/// One low-rank patch for one base tensor: `W' = W + scale * (B @ A)`.
#[derive(Debug, Clone)]
struct WanLoraPatch {
    /// `[rank, in_features]`.
    a: Tensor,
    /// `[out_features, rank]`.
    b: Tensor,
    /// `user_scale * alpha / rank`, already folded.
    scale: f64,
}

/// Every patch an adapter stack contributes, keyed by the **bare checkpoint
/// tensor name** (e.g. `blocks.0.self_attn.q.weight`).
#[derive(Debug, Default, Clone)]
pub(crate) struct WanLoraRegistry {
    patches: HashMap<String, Vec<WanLoraPatch>>,
}

impl WanLoraRegistry {
    /// Number of distinct base tensors any adapter touches.
    pub fn tensor_count(&self) -> usize {
        self.patches.len()
    }

    /// Total patches, counting a stack of adapters on one tensor separately.
    pub fn patch_count(&self) -> usize {
        self.patches.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }

    /// Fail if any patched tensor is absent from the checkpoint being loaded.
    ///
    /// The pairing check in [`Self::load`] only proves an adapter is *a* Wan
    /// adapter. A 2.2 A14B distill against the 1.3B checkpoint pairs just as
    /// cleanly — same key spelling, same module names — and then applies to the
    /// 30 blocks that exist while 10 blocks of the distill are dropped. At four
    /// steps a partially distilled model renders noise rather than a slightly
    /// worse video, so this is worth an error instead of a warning.
    ///
    /// Both weight sources call this, so the safetensors backend (which only
    /// ever sees the tensors the model asks for) and the GGUF path (which holds
    /// the whole set) cannot disagree about what counts as applied.
    pub fn ensure_targets_present(
        &self,
        contains: impl Fn(&str) -> bool,
        checkpoint: &std::path::Path,
    ) -> Result<()> {
        let mut missing: Vec<&str> = self
            .patches
            .keys()
            .filter(|name| !contains(name))
            .map(String::as_str)
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        missing.sort_unstable();
        bail!(
            "this LoRA stack patches {} tensor(s) that {} does not have, starting with `{}` — \
             the adapter was trained against a different Wan checkpoint, and applying only the \
             part that matches would leave the model half-distilled",
            missing.len(),
            checkpoint.display(),
            missing[0]
        );
    }

    /// Fold every patch for `name` into `base`. Deltas are accumulated in F32
    /// and cast back once, so a stack of adapters on a bf16 weight does not
    /// round between merges.
    ///
    /// The public name for the same operation, for weight sources outside this
    /// module that hold an already-dequantized tensor.
    pub fn merge_into(&self, name: &str, base: Tensor) -> candle_core::Result<Tensor> {
        self.apply(name, base)
    }

    fn apply(&self, name: &str, base: Tensor) -> candle_core::Result<Tensor> {
        let Some(patches) = self.patches.get(name) else {
            return Ok(base);
        };
        let dtype = base.dtype();
        let device = base.device().clone();
        let mut merged = base.to_dtype(DType::F32)?;
        for patch in patches {
            let a = patch.a.to_device(&device)?.to_dtype(DType::F32)?;
            let b = patch.b.to_device(&device)?.to_dtype(DType::F32)?;
            merged = merged.add(&b.matmul(&a)?.affine(patch.scale, 0.0)?)?;
        }
        merged.to_dtype(dtype)
    }

    /// Parse a stack of adapters into patches against the Wan checkpoint's own
    /// key space.
    pub fn load(loras: &[LoraWeight]) -> Result<Self> {
        let mut registry = Self::default();
        for lora in loras {
            registry.absorb(lora)?;
        }
        Ok(registry)
    }

    fn absorb(&mut self, lora: &LoraWeight) -> Result<()> {
        let path = std::path::Path::new(&lora.path);
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .with_context(|| format!("failed to read Wan LoRA {}", path.display()))?;

        // Refuse the 2.1-era format outright. Loading only its low-rank half
        // would silently drop hundreds of full-weight deltas and produce a
        // subtly wrong model rather than an error.
        let full_deltas = tensors
            .keys()
            .filter(|name| FULL_DELTA_SUFFIXES.iter().any(|s| name.ends_with(*s)))
            .count();
        if full_deltas > 0 {
            bail!(
                "{} is a Wan 2.1-era Lightning adapter: it carries {full_deltas} full-weight \
                 `.diff`/`.diff_b` tensors alongside its low-rank pairs, and mold would have to \
                 ignore them. Use the Wan 2.2 rank-64 distill (pure low-rank with an alpha) \
                 instead.",
                path.display()
            );
        }

        let mut down: HashMap<String, Tensor> = HashMap::new();
        let mut up: HashMap<String, Tensor> = HashMap::new();
        let mut alphas: HashMap<String, f64> = HashMap::new();
        for (name, tensor) in &tensors {
            if let Some((direction, stem)) = classify_lora_key(name) {
                let Some(stem) = canonical_stem(stem) else {
                    continue;
                };
                match direction {
                    LoraDirection::Down => down.insert(stem, tensor.clone()),
                    LoraDirection::Up => up.insert(stem, tensor.clone()),
                };
            } else if let Some(stem) = name.strip_suffix(".alpha") {
                if let Some(stem) = canonical_stem(stem) {
                    if let Some(value) = crate::flux::lora::read_alpha_scalar(tensor) {
                        alphas.insert(stem, value);
                    }
                }
            }
        }

        let mut paired = 0usize;
        for (stem, a) in down {
            let Some(b) = up.get(&stem) else { continue };
            let rank = a.dim(0)?;
            if rank == 0 {
                bail!("{}: LoRA {stem} has rank 0", path.display());
            }
            // alpha/rank is the kohya convention. The Wan 2.2 distills ship
            // alpha 8 at rank 64, so full strength is 0.125 — not 1.0.
            let scale = match alphas.get(&stem) {
                Some(alpha) => lora.scale * alpha / rank as f64,
                None => lora.scale,
            };
            self.patches
                .entry(format!("{stem}.weight"))
                .or_default()
                .push(WanLoraPatch {
                    a,
                    b: b.clone(),
                    scale,
                });
            paired += 1;
        }

        if paired == 0 {
            bail!(
                "no Wan LoRA pairs found in {} — expected `{WAN_LORA_PREFIX}blocks.N.<module>.\
                 lora_down/lora_up.weight` keys",
                path.display()
            );
        }
        Ok(())
    }
}

/// Turn a LoRA stem into the checkpoint's own key, or `None` if it names
/// nothing the Wan DiT has.
///
/// The only transformation needed is dropping the `diffusion_model.` prefix
/// (and a `model.` wrapper some exporters add on top). After that the stem is
/// already the checkpoint path.
fn canonical_stem(stem: &str) -> Option<String> {
    let stem = stem.strip_prefix("model.").unwrap_or(stem);
    let stem = stem.strip_prefix(WAN_LORA_PREFIX).unwrap_or(stem);
    // Guard against absorbing a foreign family's adapter: every Wan target is
    // a block projection or one of the handful of top-level modules.
    let recognized = stem.starts_with("blocks.")
        || stem.starts_with("patch_embedding")
        || stem.starts_with("text_embedding.")
        || stem.starts_with("time_embedding.")
        || stem.starts_with("time_projection.")
        || stem.starts_with("head.");
    recognized.then(|| stem.to_string())
}

/// One adapter's low-rank pair, resident on the execution device.
///
/// Kept separate from [`WanLoraPatch`], whose tensors stay on the host: the
/// registry is built once per request and the branch is built once per module
/// that the checkpoint actually has.
struct WanLoraBranch {
    /// `[rank, in_features]`.
    a: Tensor,
    /// `[out_features, rank]`.
    b: Tensor,
    scale: f64,
}

/// A quantized projection with its adapters applied as a parallel low-rank
/// branch: `y = QMatMul(x) + scale * ((x @ A^T) @ B^T)`.
///
/// **Why a branch and not a merge.** The other two weight sources merge
/// `W + scale*(B @ A)` into the weight as it is read, because they can: a bf16
/// tensor takes the delta losslessly. A GGUF tensor cannot. Merging there means
/// dequantize -> add -> **requantize**, and both halves of that are problems:
///
/// - *Cost.* Candle's k-quant `from_float` is scalar and single-threaded:
///   measured 57 Mparam/s for Q5_K on this machine (dequantize is 641 Mparam/s).
///   The Lightning distill targets all 400 block projections, which for A14B is
///   14.0 B of the checkpoint's 14.3 B parameters — so a merge would requantize
///   essentially the whole expert, ~250 s each, ~8 minutes for the pair, on
///   every generation. The tier exists to render in ~90 s.
/// - *Fidelity.* Q5_K resolves roughly 5 bits against each 32-value block's
///   peak. A distill delta that is a few percent of the weight is exactly the
///   magnitude that rounding discards, so the merge would pay those 8 minutes
///   to apply a fraction of the adapter.
///
/// The branch has neither problem, and it is what the reference stack for these
/// files does. ComfyUI-GGUF dequantizes each tensor per forward and applies the
/// patch to the dequantized copy (`city96/ComfyUI-GGUF ops.py:176-190`) — it
/// never requantizes either. Mold's `QMatMul` fuses dequantization into the
/// kernel, so the patch cannot be intercepted there; adding it as a parallel
/// branch is the same arithmetic in the other order.
///
/// The costs it does have, on an A14B expert with the rank-64 distill: ~1.2 GB
/// of F32 adapter weights beside a ~10 GB Q5_K expert, and `rank/in + rank/out`
/// extra FLOPs — 2.5 % of the projection's own matmul at 5120 wide.
pub(crate) struct WanQuantizedLoraLinear {
    base: mold_candle::quantized_nn::Linear,
    branches: Vec<WanLoraBranch>,
}

impl candle_core::Module for WanQuantizedLoraLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut out = self.base.forward(x)?;
        for branch in &self.branches {
            // `x` is [batch, tokens, in] and the adapter is 2-D, so the
            // broadcast form is the one that does not materialize a per-batch
            // copy of the adapter.
            let hidden = x.broadcast_matmul(&branch.a.t()?)?;
            out = out.add(
                &hidden
                    .broadcast_matmul(&branch.b.t()?)?
                    .affine(branch.scale, 0.0)?,
            )?;
        }
        Ok(out)
    }
}

/// Build the linear for `key`, attaching a low-rank branch per patch.
///
/// Returns the bare quantized linear untouched when nothing patches `key`,
/// which is the case for every tensor outside the 400 the distills target.
pub(crate) fn attach_quantized_branches(
    registry: &WanLoraRegistry,
    key: &str,
    base: mold_candle::quantized_nn::Linear,
    device: &Device,
) -> candle_core::Result<std::sync::Arc<dyn candle_core::Module + Send + Sync>> {
    let Some(patches) = registry.patches.get(key) else {
        return Ok(std::sync::Arc::new(base));
    };
    let mut branches = Vec::with_capacity(patches.len());
    for patch in patches {
        branches.push(WanLoraBranch {
            // The GGUF path computes in F32 (`QMatMul` dequantizes into it),
            // so the adapter is staged at F32 once rather than cast per step.
            a: patch.a.to_device(device)?.to_dtype(DType::F32)?,
            b: patch.b.to_device(device)?.to_dtype(DType::F32)?,
            scale: patch.scale,
        });
    }
    Ok(std::sync::Arc::new(WanQuantizedLoraLinear {
        base,
        branches,
    }))
}

/// A `SimpleBackend` over the base checkpoint that merges LoRA deltas as each
/// tensor is read.
///
/// This is the flux `LoraBackend` shape, minus the fused-slice machinery: Wan
/// keeps q/k/v as separate projections, so every patch is a whole-tensor
/// `Direct` merge and there is no slice bookkeeping to get wrong.
pub(crate) struct WanLoraBackend {
    st: MmapedSafetensors,
    /// Prefix the checkpoint puts on its keys (`model.diffusion_model.` for the
    /// shipped Comfy-Org repacks, empty for bare exports).
    prefix: String,
    registry: WanLoraRegistry,
}

impl WanLoraBackend {
    /// # Safety
    /// Memory-maps `paths`; the files must not be mutated while loaded.
    pub unsafe fn new(
        paths: &[std::path::PathBuf],
        prefix: String,
        registry: WanLoraRegistry,
    ) -> candle_core::Result<Self> {
        let st = unsafe { MmapedSafetensors::multi(paths) }?;
        Ok(Self {
            st,
            prefix,
            registry,
        })
    }
}

impl candle_nn::var_builder::SimpleBackend for WanLoraBackend {
    fn get(
        &self,
        _shape: candle_core::Shape,
        name: &str,
        _init: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        self.get_unchecked(name, dtype, device)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let raw = format!("{}{name}", self.prefix);
        let tensor = self.st.load(&raw, device)?;
        let tensor = if tensor.dtype() == dtype {
            tensor
        } else {
            tensor.to_dtype(dtype)?
        };
        self.registry.apply(name, tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.st.get(&format!("{}{name}", self.prefix)).is_ok()
    }
}

impl WanLoraBackend {
    /// Refuse an adapter that names tensors this checkpoint does not carry.
    pub fn ensure_lora_targets_present(&self, checkpoint: &std::path::Path) -> Result<()> {
        self.registry.ensure_targets_present(
            |name| candle_nn::var_builder::SimpleBackend::contains_tensor(self, name),
            checkpoint,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap as StdHashMap;

    fn write_tensors(path: &std::path::Path, entries: &[(&str, SafeDtype, Vec<usize>, Vec<u8>)]) {
        let mut map: StdHashMap<String, TensorView<'_>> = StdHashMap::new();
        for (name, dtype, shape, data) in entries {
            map.insert(
                (*name).to_string(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            );
        }
        serialize_to_file(&map, &None, path).unwrap();
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A minimal rank-2 Wan adapter with an **I64** alpha, exactly the shape
    /// the shipped Lightning distills use.
    fn write_wan_lora(path: &std::path::Path, alpha: Option<(SafeDtype, Vec<u8>)>) {
        let mut entries: Vec<(&str, SafeDtype, Vec<usize>, Vec<u8>)> = vec![
            (
                "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
                SafeDtype::F32,
                vec![2, 4],
                f32_bytes(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            ),
            (
                "diffusion_model.blocks.0.self_attn.q.lora_up.weight",
                SafeDtype::F32,
                vec![4, 2],
                f32_bytes(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ),
        ];
        if let Some((dtype, data)) = &alpha {
            entries.push((
                "diffusion_model.blocks.0.self_attn.q.alpha",
                *dtype,
                vec![],
                data.clone(),
            ));
        }
        write_tensors(path, &entries);
    }

    fn lora(path: &std::path::Path, scale: f64) -> LoraWeight {
        LoraWeight {
            path: path.to_string_lossy().to_string(),
            scale,
        }
    }

    /// The regression this whole part exists for: an `I64` alpha must be read,
    /// not silently dropped. alpha 8 at rank 2 means scale 4.0 at strength 1.
    #[test]
    fn i64_alpha_is_read_not_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wan_i64_alpha.safetensors");
        write_wan_lora(&path, Some((SafeDtype::I64, 8i64.to_le_bytes().to_vec())));

        let registry = WanLoraRegistry::load(&[lora(&path, 1.0)]).unwrap();
        assert_eq!(registry.tensor_count(), 1);
        let patch = &registry.patches["blocks.0.self_attn.q.weight"][0];
        assert_eq!(patch.scale, 4.0, "alpha 8 / rank 2 = 4.0");
    }

    /// The same file with an F32 alpha must behave identically — the fix is
    /// dtype-agnostic, not I64-specific.
    #[test]
    fn f32_alpha_still_works() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wan_f32_alpha.safetensors");
        write_wan_lora(&path, Some((SafeDtype::F32, f32_bytes(&[8.0]))));
        let registry = WanLoraRegistry::load(&[lora(&path, 1.0)]).unwrap();
        assert_eq!(
            registry.patches["blocks.0.self_attn.q.weight"][0].scale,
            4.0
        );
    }

    /// Without an alpha the user scale passes through unmodified.
    #[test]
    fn missing_alpha_falls_back_to_the_user_scale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wan_no_alpha.safetensors");
        write_wan_lora(&path, None);
        let registry = WanLoraRegistry::load(&[lora(&path, 0.75)]).unwrap();
        assert_eq!(
            registry.patches["blocks.0.self_attn.q.weight"][0].scale,
            0.75
        );
    }

    /// The shipped ratio: alpha 8 at rank 64 is 0.125 at full strength. A
    /// loader that dropped the alpha would apply 1.0 — 8x too much.
    #[test]
    fn shipped_lightning_ratio_is_one_eighth() {
        let rank = 64.0;
        let alpha = 8.0;
        assert_eq!(1.0 * alpha / rank, 0.125);
    }

    #[test]
    fn diffusion_model_prefix_is_stripped_and_stems_match_checkpoint_paths() {
        for (input, want) in [
            (
                "diffusion_model.blocks.0.self_attn.q",
                Some("blocks.0.self_attn.q"),
            ),
            ("diffusion_model.blocks.39.ffn.2", Some("blocks.39.ffn.2")),
            (
                "model.diffusion_model.blocks.7.cross_attn.v",
                Some("blocks.7.cross_attn.v"),
            ),
            // Already bare.
            ("blocks.3.self_attn.o", Some("blocks.3.self_attn.o")),
            ("diffusion_model.head.head", Some("head.head")),
            ("diffusion_model.text_embedding.0", Some("text_embedding.0")),
            // Another family's adapter must not be absorbed.
            ("diffusion_model.transformer_blocks.0.attn.to_q", None),
            ("lora_unet_double_blocks_0_img_attn_qkv", None),
            ("diffusion_model.single_transformer_blocks.0.proj_out", None),
        ] {
            assert_eq!(canonical_stem(input).as_deref(), want, "{input}");
        }
    }

    /// 2.1-era adapters must be refused by name, not partially applied.
    #[test]
    fn wan21_full_delta_adapters_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wan21_style.safetensors");
        write_tensors(
            &path,
            &[
                (
                    "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
                    SafeDtype::F32,
                    vec![2, 4],
                    f32_bytes(&[0.0; 8]),
                ),
                (
                    "diffusion_model.blocks.0.self_attn.q.lora_up.weight",
                    SafeDtype::F32,
                    vec![4, 2],
                    f32_bytes(&[0.0; 8]),
                ),
                (
                    "diffusion_model.blocks.0.self_attn.norm_q.diff",
                    SafeDtype::F32,
                    vec![4],
                    f32_bytes(&[0.0; 4]),
                ),
                (
                    "diffusion_model.blocks.0.self_attn.q.diff_b",
                    SafeDtype::F32,
                    vec![4],
                    f32_bytes(&[0.0; 4]),
                ),
            ],
        );
        let error = WanLoraRegistry::load(&[lora(&path, 1.0)])
            .unwrap_err()
            .to_string();
        assert!(error.contains("2.1-era"), "{error}");
        assert!(error.contains("full-weight"), "{error}");
        assert!(error.contains('2'), "the count must be named: {error}");
    }

    #[test]
    fn an_adapter_naming_nothing_wan_has_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("foreign.safetensors");
        write_tensors(
            &path,
            &[
                (
                    "diffusion_model.transformer_blocks.0.attn.to_q.lora_down.weight",
                    SafeDtype::F32,
                    vec![2, 4],
                    f32_bytes(&[0.0; 8]),
                ),
                (
                    "diffusion_model.transformer_blocks.0.attn.to_q.lora_up.weight",
                    SafeDtype::F32,
                    vec![4, 2],
                    f32_bytes(&[0.0; 8]),
                ),
            ],
        );
        let error = WanLoraRegistry::load(&[lora(&path, 1.0)])
            .unwrap_err()
            .to_string();
        assert!(error.contains("no Wan LoRA pairs"), "{error}");
    }

    /// The merge must equal the hand-computed `W + scale * (B @ A)`.
    #[test]
    fn merged_weight_equals_the_manual_delta() {
        let device = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("delta.safetensors");
        // A = [[1,0,0,0],[0,1,0,0]] (rank 2, in 4)
        // B = [[1,0],[0,1],[0,0],[0,0]] (out 4, rank 2)
        // B@A = diag-ish: rows 0,1 pick inputs 0,1; rows 2,3 zero.
        write_wan_lora(&path, Some((SafeDtype::I64, 8i64.to_le_bytes().to_vec())));
        let registry = WanLoraRegistry::load(&[lora(&path, 0.5)]).unwrap();
        // scale = 0.5 * 8 / 2 = 2.0
        let scale = 2.0f32;

        let base = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let merged = registry
            .apply("blocks.0.self_attn.q.weight", base.clone())
            .unwrap();
        let got: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();

        let mut want = vec![0f32; 16];
        want[0] = scale; // row 0, col 0
        want[5] = scale; // row 1, col 1
        assert_eq!(got, want);

        // An untouched tensor passes through byte-identical.
        let other = registry
            .apply("blocks.0.self_attn.k.weight", base.clone())
            .unwrap();
        assert_eq!(
            other.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0f32; 16]
        );
    }

    /// Two adapters on one tensor accumulate additively.
    #[test]
    fn stacked_adapters_accumulate() {
        let device = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("a.safetensors");
        let second = dir.path().join("b.safetensors");
        write_wan_lora(&first, Some((SafeDtype::I64, 8i64.to_le_bytes().to_vec())));
        write_wan_lora(&second, Some((SafeDtype::I64, 8i64.to_le_bytes().to_vec())));

        let registry = WanLoraRegistry::load(&[lora(&first, 0.5), lora(&second, 0.25)]).unwrap();
        assert_eq!(registry.tensor_count(), 1);
        assert_eq!(registry.patch_count(), 2, "both adapters must be retained");

        let base = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let merged = registry.apply("blocks.0.self_attn.q.weight", base).unwrap();
        let got: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
        // (0.5 + 0.25) * 8 / 2 = 3.0
        assert_eq!(got[0], 3.0);
        assert_eq!(got[5], 3.0);
    }

    /// End to end through the real loader: build a tiny checkpoint on disk,
    /// load it twice — once bare, once with an adapter — and check the merged
    /// weight equals `W + scale * (B @ A)` computed by hand.
    ///
    /// This is the test that proves the `SimpleBackend` interception actually
    /// reaches the transformer's weights, which the registry-level tests
    /// cannot show.
    #[test]
    fn transformer_load_merges_the_adapter_into_the_weight() {
        use crate::wan::model::transformer::{WanTransformer, WanTransformerConfig};
        use candle_nn::{VarBuilder, VarMap};

        let device = Device::Cpu;
        let dtype = DType::F32;
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("tiny_wan.safetensors");

        // Materialize a tiny checkpoint through the model's own loader so the
        // key set is exactly what it reads back.
        let config = WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let varmap = VarMap::new();
        WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            config.clone(),
        )
        .unwrap();
        varmap.save(&checkpoint).unwrap();

        let target = "blocks.0.self_attn.q.weight";
        let base = {
            let data = varmap.data().lock().unwrap();
            data[target].as_tensor().clone()
        };
        let (out_features, in_features) = base.dims2().unwrap();

        // rank-2 adapter: A picks the first two inputs, B routes them to the
        // first two outputs, so the delta is a 2x2 identity block.
        let rank = 2usize;
        let mut a = vec![0f32; rank * in_features];
        a[0] = 1.0;
        a[in_features + 1] = 1.0;
        let mut b = vec![0f32; out_features * rank];
        b[0] = 1.0;
        b[rank + 1] = 1.0;

        let lora_path = dir.path().join("adapter.safetensors");
        write_tensors(
            &lora_path,
            &[
                (
                    "diffusion_model.blocks.0.self_attn.q.lora_down.weight",
                    SafeDtype::F32,
                    vec![rank, in_features],
                    f32_bytes(&a),
                ),
                (
                    "diffusion_model.blocks.0.self_attn.q.lora_up.weight",
                    SafeDtype::F32,
                    vec![out_features, rank],
                    f32_bytes(&b),
                ),
                (
                    "diffusion_model.blocks.0.self_attn.q.alpha",
                    SafeDtype::I64,
                    vec![],
                    8i64.to_le_bytes().to_vec(),
                ),
            ],
        );

        // scale = user 0.5 * alpha 8 / rank 2 = 2.0
        let registry = WanLoraRegistry::load(&[lora(&lora_path, 0.5)]).unwrap();
        assert_eq!(registry.tensor_count(), 1);

        let bare = WanTransformer::from_safetensors_with_loras(
            std::slice::from_ref(&checkpoint),
            config.clone(),
            &device,
            dtype,
            &WanLoraRegistry::default(),
        )
        .unwrap();
        let merged = WanTransformer::from_safetensors_with_loras(
            &[checkpoint],
            config.clone(),
            &device,
            dtype,
            &registry,
        )
        .unwrap();

        // Both models must run; the merged one must differ.
        let x = Tensor::zeros((1, config.in_dim, 1, 4, 4), dtype, &device).unwrap();
        let timestep = Tensor::from_vec(vec![500f32], 1, &device).unwrap();
        let context = Tensor::zeros((1, 4, config.text_dim), dtype, &device).unwrap();
        let bare_out = bare.forward(&x, &timestep, &context).unwrap();
        let merged_out = merged.forward(&x, &timestep, &context).unwrap();
        assert_eq!(bare_out.dims(), merged_out.dims());

        // The hand-computed expectation for the one patched tensor.
        let expected = registry.apply(target, base.clone()).unwrap();
        let base_v: Vec<f32> = base.flatten_all().unwrap().to_vec1().unwrap();
        let expected_v: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(expected_v[0], base_v[0] + 2.0, "row 0 col 0 gains scale");
        assert_eq!(
            expected_v[in_features + 1],
            base_v[in_features + 1] + 2.0,
            "row 1 col 1 gains scale"
        );
        assert_eq!(
            expected_v[2], base_v[2],
            "untouched elements must be unchanged"
        );
    }

    // -----------------------------------------------------------------------
    // GGUF: adapters as parallel low-rank branches
    // -----------------------------------------------------------------------

    use crate::wan::model::transformer::{WanTransformer, WanTransformerConfig};
    use candle_core::quantized::GgmlDType;
    use candle_nn::{VarBuilder, VarMap};

    /// Deterministic fill, so a failure is reproducible and the two checkpoints
    /// under comparison hold identical numbers.
    fn synth(seed: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| 0.05 * (0.7 * (i + seed * 13) as f32).sin())
            .collect()
    }

    /// Every projection the shipped distills target, for one block.
    fn block_projections(
        config: &WanTransformerConfig,
        block: usize,
    ) -> Vec<(String, usize, usize)> {
        let dim = config.dim;
        let mut modules = Vec::new();
        for attn in ["self_attn", "cross_attn"] {
            for leaf in ["q", "k", "v", "o"] {
                modules.push((format!("blocks.{block}.{attn}.{leaf}"), dim, dim));
            }
        }
        modules.push((format!("blocks.{block}.ffn.0"), config.ffn_dim, dim));
        modules.push((format!("blocks.{block}.ffn.2"), dim, config.ffn_dim));
        modules
    }

    /// Write a rank-`rank` adapter over `modules`, in the shipped 2.2 layout:
    /// `diffusion_model.` prefix, `lora_down`/`lora_up`, I64 alpha.
    fn write_block_lora(
        path: &std::path::Path,
        modules: &[(String, usize, usize)],
        rank: usize,
        alpha: i64,
    ) {
        let mut owned: Vec<(String, SafeDtype, Vec<usize>, Vec<u8>)> = Vec::new();
        for (index, (stem, out_features, in_features)) in modules.iter().enumerate() {
            owned.push((
                format!("diffusion_model.{stem}.lora_down.weight"),
                SafeDtype::F32,
                vec![rank, *in_features],
                f32_bytes(&synth(index * 2 + 1, rank * in_features)),
            ));
            owned.push((
                format!("diffusion_model.{stem}.lora_up.weight"),
                SafeDtype::F32,
                vec![*out_features, rank],
                f32_bytes(&synth(index * 2 + 2, out_features * rank)),
            ));
            owned.push((
                format!("diffusion_model.{stem}.alpha"),
                SafeDtype::I64,
                vec![],
                alpha.to_le_bytes().to_vec(),
            ));
        }
        let borrowed: Vec<(&str, SafeDtype, Vec<usize>, Vec<u8>)> = owned
            .iter()
            .map(|(name, dtype, shape, data)| (name.as_str(), *dtype, shape.clone(), data.clone()))
            .collect();
        write_tensors(path, &borrowed);
    }

    /// Write every tensor in `varmap` to a GGUF, quantizing each with
    /// `dtype_for`.
    fn write_gguf(path: &std::path::Path, varmap: &VarMap, dtype_for: impl Fn(&str) -> GgmlDType) {
        use candle_core::quantized::{gguf_file, QTensor};

        let quantized: Vec<(String, QTensor)> = {
            let data = varmap.data().lock().unwrap();
            data.iter()
                .map(|(name, var)| {
                    (
                        name.clone(),
                        QTensor::quantize(var.as_tensor(), dtype_for(name)).unwrap(),
                    )
                })
                .collect()
        };
        let refs: Vec<(&str, &QTensor)> = quantized
            .iter()
            .map(|(name, q)| (name.as_str(), q))
            .collect();
        let arch = gguf_file::Value::String("wan".to_string());
        let mut file = std::fs::File::create(path).unwrap();
        gguf_file::write(&mut file, &[("general.architecture", &arch)], &refs).unwrap();
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// The GGUF path applies the adapter as `y = Wx + scale*(B@(A@x))` while
    /// the safetensors path applies it as `y = (W + scale*(B@A))x`. Those are
    /// the same function, and at a lossless F32 GGUF quantization the two
    /// loaders must agree to f32 round-off — no quantization tolerance.
    #[test]
    fn a_gguf_low_rank_branch_equals_the_merged_safetensors_forward() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let dir = tempfile::tempdir().unwrap();

        let config = WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let varmap = VarMap::new();
        WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            config.clone(),
        )
        .unwrap();

        let safetensors_path = dir.path().join("tiny.safetensors");
        varmap.save(&safetensors_path).unwrap();
        let gguf_path = dir.path().join("tiny.gguf");
        write_gguf(&gguf_path, &varmap, |_| GgmlDType::F32);

        // All twenty projections of both blocks, exactly the set the shipped
        // distills touch.
        let mut modules = block_projections(&config, 0);
        modules.extend(block_projections(&config, 1));
        let adapter = dir.path().join("adapter.safetensors");
        write_block_lora(&adapter, &modules, 4, 8);
        let registry = WanLoraRegistry::load(&[lora(&adapter, 1.0)]).unwrap();
        assert_eq!(registry.tensor_count(), 20);

        let x = Tensor::randn(0f32, 1f32, (1, config.in_dim, 1, 4, 4), &device).unwrap();
        let timestep = Tensor::from_vec(vec![500f32], 1, &device).unwrap();
        let context = Tensor::randn(0f32, 1f32, (1, 4, config.text_dim), &device).unwrap();
        let run = |model: &WanTransformer| model.forward(&x, &timestep, &context).unwrap();

        let merged = run(&WanTransformer::from_safetensors_with_loras(
            &[safetensors_path],
            config.clone(),
            &device,
            dtype,
            &registry,
        )
        .unwrap());
        let branched = run(&WanTransformer::from_gguf_with_loras(
            &gguf_path,
            config.clone(),
            &device,
            &registry,
        )
        .unwrap());
        let bare = run(&WanTransformer::from_gguf(&gguf_path, config, &device).unwrap());

        let err = max_abs_diff(&branched, &merged);
        assert!(
            err < 1e-4,
            "the branch and the merge must be the same function, diverged by {err:e}"
        );
        // Otherwise the agreement above would also hold for a loader that
        // ignored the adapter entirely.
        assert!(
            max_abs_diff(&branched, &bare) > 1e-3,
            "the adapter must actually reach the GGUF forward"
        );
    }

    /// The same equivalence on a genuinely lossy checkpoint, quantized the way
    /// the QuantStack A14B files are: Q5_K for q/k/o and `ffn.0`, Q6_K for v
    /// and `ffn.2`, everything else unquantized. Here the two paths cannot
    /// agree exactly — the bases differ — but the *adapter's contribution*
    /// must be the same to within the base's own error.
    #[test]
    fn a_k_quantized_gguf_receives_the_adapter_at_full_precision() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let dir = tempfile::tempdir().unwrap();

        // 256 wide so the k-quant super-block (256 values) divides every
        // projection's inner dimension; `patch_embedding` stays F32 because its
        // 5-D shape ends in the 2-wide patch axis, which is also why the real
        // files leave it alone.
        let config = WanTransformerConfig::tiny(256, 2, 1);
        let varmap = VarMap::new();
        WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            config.clone(),
        )
        .unwrap();

        let gguf_path = dir.path().join("kquant.gguf");
        write_gguf(&gguf_path, &varmap, |name| {
            if !name.starts_with("blocks.") || !name.ends_with(".weight") || name.contains("norm") {
                return GgmlDType::F32;
            }
            if name.ends_with(".v.weight") || name.ends_with("ffn.2.weight") {
                GgmlDType::Q6K
            } else {
                GgmlDType::Q5K
            }
        });

        let modules = block_projections(&config, 0);
        let adapter = dir.path().join("adapter.safetensors");
        write_block_lora(&adapter, &modules, 8, 8);
        let registry = WanLoraRegistry::load(&[lora(&adapter, 1.0)]).unwrap();
        assert_eq!(registry.tensor_count(), 10);

        // Deterministic inputs: this test bounds a ratio between two error
        // magnitudes, and a random draw would make that bound flaky rather than
        // strict.
        let latent_len = config.in_dim * 16;
        let x = Tensor::from_vec(synth(101, latent_len), (1, config.in_dim, 1, 4, 4), &device)
            .unwrap()
            .affine(20.0, 0.0)
            .unwrap();
        let timestep = Tensor::from_vec(vec![500f32], 1, &device).unwrap();
        let context = Tensor::from_vec(
            synth(202, 4 * config.text_dim),
            (1, 4, config.text_dim),
            &device,
        )
        .unwrap()
        .affine(20.0, 0.0)
        .unwrap();
        let run = |model: &WanTransformer| model.forward(&x, &timestep, &context).unwrap();

        let quantized_bare =
            run(&WanTransformer::from_gguf(&gguf_path, config.clone(), &device).unwrap());
        let quantized_branched = run(&WanTransformer::from_gguf_with_loras(
            &gguf_path,
            config.clone(),
            &device,
            &registry,
        )
        .unwrap());

        // The plain-weight pair, for the contribution the adapter is supposed
        // to make.
        let plain_path = dir.path().join("plain.safetensors");
        varmap.save(&plain_path).unwrap();
        let plain_bare = run(&WanTransformer::from_safetensors(
            std::slice::from_ref(&plain_path),
            config.clone(),
            &device,
            dtype,
        )
        .unwrap());
        let plain_merged = run(&WanTransformer::from_safetensors_with_loras(
            &[plain_path],
            config,
            &device,
            dtype,
            &registry,
        )
        .unwrap());

        let peak = |t: &Tensor| {
            t.abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        let quantized_effect = peak(&(quantized_branched - &quantized_bare).unwrap());
        let plain_effect = peak(&(plain_merged - &plain_bare).unwrap());
        let base_error = max_abs_diff(&quantized_bare, &plain_bare);

        assert!(
            plain_effect > 1e-3,
            "the fixture's adapter must move the output; it moved it by {plain_effect:e}"
        );
        // The sharp assertion. The adapter's contribution rides on a quantized
        // base and travels through the block's non-linearities, so it cannot be
        // compared exactly — but it must arrive at full strength. An attenuated
        // contribution is precisely what merge-and-requantize produces, and
        // because the base's own quantization error (3.09e-1 here) is larger
        // than the adapter's effect, an absolute-error bound would pass on a
        // loader that dropped the adapter outright — it did, until this was
        // mutation-tested. Measured on this fixture: plain 1.73e-1, quantized
        // 2.24e-1, ratio 1.30; the spread over 1.0 is the base error and the
        // block's non-linearities, not attenuation.
        let ratio = quantized_effect / plain_effect;
        assert!(
            (0.5..2.0).contains(&ratio),
            "the adapter moved the quantized forward by {quantized_effect:e} against \
             {plain_effect:e} on the plain one (ratio {ratio}) — it is arriving attenuated \
             or not at all, with the base's own error at {base_error:e}"
        );
    }

    /// An adapter naming tensors the checkpoint lacks — a 2.2 A14B distill
    /// pointed at a 30-block checkpoint, say — must fail on both weight
    /// sources, not apply to the blocks that happen to exist.
    #[test]
    fn an_adapter_naming_absent_tensors_is_refused_by_both_weight_sources() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let dir = tempfile::tempdir().unwrap();

        let config = WanTransformerConfig {
            ffn_dim: 32,
            text_dim: 32,
            freq_dim: 16,
            ..WanTransformerConfig::tiny(16, 2, 2)
        };
        let varmap = VarMap::new();
        WanTransformer::from_var_builder(
            VarBuilder::from_varmap(&varmap, dtype, &device),
            config.clone(),
        )
        .unwrap();
        let safetensors_path = dir.path().join("two_blocks.safetensors");
        varmap.save(&safetensors_path).unwrap();
        let gguf_path = dir.path().join("two_blocks.gguf");
        write_gguf(&gguf_path, &varmap, |_| GgmlDType::F32);

        // Block 0 exists; block 7 does not.
        let mut modules = block_projections(&config, 0);
        modules.extend(block_projections(&config, 7));
        let adapter = dir.path().join("too_deep.safetensors");
        write_block_lora(&adapter, &modules, 4, 8);
        let registry = WanLoraRegistry::load(&[lora(&adapter, 1.0)]).unwrap();

        for error in [
            WanTransformer::from_safetensors_with_loras(
                &[safetensors_path],
                config.clone(),
                &device,
                dtype,
                &registry,
            )
            .err(),
            WanTransformer::from_gguf_with_loras(&gguf_path, config, &device, &registry).err(),
        ] {
            let error = error
                .expect("an adapter for a deeper checkpoint must not load")
                .to_string();
            assert!(error.contains("10 tensor(s)"), "{error}");
            assert!(error.contains("blocks.7."), "{error}");
        }
    }

    #[test]
    fn an_empty_stack_produces_an_empty_registry() {
        let registry = WanLoraRegistry::load(&[]).unwrap();
        assert!(registry.is_empty());
        assert_eq!(registry.patch_count(), 0);
    }
}
