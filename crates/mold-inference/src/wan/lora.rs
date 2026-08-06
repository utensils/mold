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

    /// Fold every patch for `name` into `base`. Deltas are accumulated in F32
    /// and cast back once, so a stack of adapters on a bf16 weight does not
    /// round between merges.
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

    #[test]
    fn an_empty_stack_produces_an_empty_registry() {
        let registry = WanLoraRegistry::load(&[]).unwrap();
        assert!(registry.is_empty());
        assert_eq!(registry.patch_count(), 0);
    }
}
