//! PuLID-FLUX identity cross-attention adapter.
//!
//! PuLID conditions a FLUX render on a face by inserting a stack of small
//! cross-attention modules between the transformer's blocks. Each module takes
//! the current image tokens as the query and a 32-token identity embedding as
//! key/value, and adds its scaled output back onto the image stream:
//!
//! ```text
//! img = img + id_weight * ca[ca_idx](id_embeds, img)
//! ```
//!
//! Ported from `ToTheBeginning/PuLID`:
//! `pulid/encoders_transformer.py:29-72` (`PerceiverAttentionCA`) and
//! `flux/model.py:83-85, 116-147` (interval constants and the two injection
//! sites). The executable oracle is stable-diffusion.cpp —
//! `src/model/adapter/pulid.hpp` for the module and
//! `src/model/diffusion/flux.hpp:993-1004` (module count) and `:1120-1161`
//! (injection) for the wiring — which consumes the same
//! `pulid_flux_v0.9.1.safetensors` weights mold downloads.
//!
//! Three properties are load-bearing and every consumer depends on them:
//!
//! * **The adapter is always resident.** It is ~1.14 GB of fp16 next to a
//!   12–24 GB transformer, and every one of its 20 modules runs inside a
//!   single forward pass. Streaming it with the offloaded blocks would pay a
//!   host↔device copy 20 times per step to save a rounding error of VRAM. The
//!   engine loads it lazily on the first identity request and drops it when a
//!   request does not condition on a face, which is the same drop-and-reload
//!   discipline the text encoders follow.
//! * **Zero identity work happens when identity is off.** The gate lives in
//!   [`PulidRuntime::hook_for_step`]: before `id_start_step`, or at an
//!   effective `id_weight` of 0, it yields `None` and the denoise loop calls
//!   the variant's ordinary `forward`. Bit-identity with a vanilla FLUX render
//!   is therefore a property of the control flow, not of an arithmetic
//!   coincidence — see `tmp/sdcpp/docs/pulid.md`'s three-way SHA check, which
//!   is the same falsification test.
//! * **The single-stream injection touches only the image slice.** After the
//!   streams are concatenated, the first `txt_len` tokens are text and must
//!   come out of the hook unchanged (`flux/model.py:141-146`).
//!
//! Attention runs through [`crate::attention`] so the Metal auto-chunking and
//! `MOLD_ATTN` rules that bound every other FLUX-family score matrix apply
//! here too. One deliberate deviation from upstream: PyTorch softmaxes the
//! score matrix in f32 and casts back (`encoders_transformer.py:67`), while
//! mold's shared helper softmaxes in the working dtype. The adapter is loaded
//! at the transformer's own compute dtype (f32 on the quantized paths, bf16 on
//! the dense one), so the f32 paths match upstream exactly and the bf16 path
//! matches every other bf16 attention in the engine.

use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};
use candle_transformers::models::flux::BlockHook;

/// Identity tokens in a PuLID embedding.
pub const ID_TOKENS: usize = 32;
/// Width of one identity token.
pub const ID_TOKEN_DIM: usize = 2048;

/// Every double-stream block whose index is a multiple of this receives an
/// injection (`flux/model.py:84`).
pub const DOUBLE_INTERVAL: usize = 2;
/// Same, for single-stream blocks (`flux/model.py:85`).
pub const SINGLE_INTERVAL: usize = 4;

/// Geometry of one `PerceiverAttentionCA`.
///
/// The defaults are PuLID-FLUX's trained shape
/// (`pulid/encoders_transformer.py:30`). They are a struct rather than
/// constants only so a test can build a transformer small enough to run on a
/// CPU in milliseconds; production never overrides them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PulidAdapterConfig {
    /// FLUX hidden size — the width of the image stream.
    pub dim: usize,
    /// Width of one attention head.
    pub dim_head: usize,
    /// Attention heads.
    pub heads: usize,
    /// Width of one identity token.
    pub kv_dim: usize,
}

impl Default for PulidAdapterConfig {
    fn default() -> Self {
        Self {
            dim: 3072,
            dim_head: 128,
            heads: 16,
            kv_dim: ID_TOKEN_DIM,
        }
    }
}

impl PulidAdapterConfig {
    fn inner_dim(&self) -> usize {
        self.dim_head * self.heads
    }
}

/// How many cross-attention modules a transformer of this shape needs.
///
/// `ceil(depth / 2) + ceil(depth_single / 4)` — 10 + 10 = 20 for FLUX.1's 19
/// double and 38 single blocks, matching `flux.hpp:994-996`. The two halves are
/// returned separately because the single-stream modules continue the same
/// index space the double-stream ones started (`ca_idx` never resets).
pub fn injection_counts(depth: usize, depth_single_blocks: usize) -> (usize, usize) {
    (
        depth.div_ceil(DOUBLE_INTERVAL),
        depth_single_blocks.div_ceil(SINGLE_INTERVAL),
    )
}

/// One PuLID cross-attention module.
///
/// `norm1` normalizes the identity tokens and `norm2` the image tokens — note
/// they are sized differently (`kv_dim` and `dim`), which is why the two cannot
/// be collapsed.
#[derive(Debug)]
pub struct PerceiverAttentionCA {
    norm1: LayerNorm,
    norm2: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    config: PulidAdapterConfig,
}

impl PerceiverAttentionCA {
    /// Load one module from `vb`, which must already be scoped to a
    /// `pulid_ca.{i}` prefix.
    pub fn load(config: PulidAdapterConfig, vb: VarBuilder) -> Result<Self> {
        let inner_dim = config.inner_dim();
        Ok(Self {
            norm1: candle_nn::layer_norm(
                config.kv_dim,
                LayerNormConfig::default(),
                vb.pp("norm1"),
            )?,
            norm2: candle_nn::layer_norm(config.dim, LayerNormConfig::default(), vb.pp("norm2"))?,
            to_q: candle_nn::linear_no_bias(config.dim, inner_dim, vb.pp("to_q"))?,
            to_kv: candle_nn::linear_no_bias(config.kv_dim, inner_dim * 2, vb.pp("to_kv"))?,
            to_out: candle_nn::linear_no_bias(inner_dim, config.dim, vb.pp("to_out"))?,
            config,
        })
    }

    /// `[b, n, heads * dim_head]` → `[b, heads, n, dim_head]`, the BHND layout
    /// [`crate::attention`] expects. Mirrors upstream `reshape_tensor`
    /// (`pulid/encoders_transformer.py:18-26`).
    fn split_heads(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        xs.reshape((b, n, self.config.heads, self.config.dim_head))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Cross-attend `image_tokens` (query) against `id_embeds` (key/value).
    ///
    /// Returns the module's raw output — the caller applies `id_weight` and the
    /// residual add, because the two injection sites scale identically but
    /// splice differently.
    pub fn forward(
        &self,
        id_embeds: &Tensor,
        image_tokens: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let x = self.norm1.forward(id_embeds)?;
        let latents = self.norm2.forward(image_tokens)?;
        let (b, seq_len, _) = latents.dims3()?;

        let q = self.to_q.forward(&latents)?;
        let kv = self.to_kv.forward(&x)?;
        let inner_dim = self.config.inner_dim();
        // Upstream `chunk(2, dim=-1)`: the first half is K, the second V.
        let k = kv.narrow(D::Minus1, 0, inner_dim)?;
        let v = kv.narrow(D::Minus1, inner_dim, inner_dim)?;

        let q = self.split_heads(&q)?;
        let k = self.split_heads(&k)?;
        let v = self.split_heads(&v)?;

        // Upstream pre-scales q and k by `dim_head^-0.25` each for fp16
        // stability, which is the same total `dim_head^-0.5`
        // (`encoders_transformer.py:65-66`).
        let scale = (self.config.dim_head as f64).powf(-0.5) as f32;
        let attn = crate::attention::attention(&q, &k, &v, scale)?;

        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, inner_dim))?;
        self.to_out.forward(&attn)
    }
}

/// The full stack of PuLID cross-attention modules for one transformer.
#[derive(Debug)]
pub struct PulidAdapter {
    ca: Vec<PerceiverAttentionCA>,
    /// Modules consumed by the double-stream loop. The single-stream loop
    /// continues at this index.
    double_injections: usize,
    config: PulidAdapterConfig,
    dtype: DType,
}

impl PulidAdapter {
    /// Load the adapter from PuLID's `pulid_flux_v0.9.x.safetensors`.
    ///
    /// The file also carries the IDFormer encoder weights, which this ignores:
    /// only the `pulid_ca.*` prefix is read. `dtype` is the transformer's
    /// working dtype — the file ships fp16 and the `VarBuilder` casts on read.
    pub fn load(
        path: &Path,
        depth: usize,
        depth_single_blocks: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // SAFETY: the same mmap contract every other mold safetensors loader
        // relies on — the file must not be mutated while the engine holds it.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&path.to_path_buf()),
                dtype,
                device,
            )
            .with_context(|| format!("reading PuLID adapter {}", path.display()))?
        };
        Self::from_var_builder(
            vb,
            PulidAdapterConfig::default(),
            depth,
            depth_single_blocks,
        )
        .with_context(|| format!("loading PuLID adapter {}", path.display()))
    }

    /// Build the stack from an already-open `VarBuilder` rooted at the file's
    /// top level (module `i` is at `pulid_ca.{i}`).
    ///
    /// The module count the transformer shape implies is checked against the
    /// count the file actually carries before anything is read, so a v1.1
    /// checkpoint — which renames the prefix to `id_adapter_attn_layers.*` —
    /// is refused by name rather than silently loading zero modules and
    /// rendering an unconditioned image.
    pub fn from_var_builder(
        vb: VarBuilder,
        config: PulidAdapterConfig,
        depth: usize,
        depth_single_blocks: usize,
    ) -> Result<Self> {
        let (double_injections, single_injections) = injection_counts(depth, depth_single_blocks);
        let expected = double_injections + single_injections;

        let mut present = 0usize;
        while vb.contains_tensor(&format!("pulid_ca.{present}.to_q.weight")) {
            present += 1;
        }
        if present != expected {
            bail!(
                "PuLID adapter carries {present} cross-attention modules but a transformer with \
                 {depth} double and {depth_single_blocks} single blocks needs {expected} \
                 ({double_injections} double + {single_injections} single); this is not a \
                 pulid_flux_v0.9.x checkpoint"
            );
        }

        let dtype = vb.dtype();
        let mut ca = Vec::with_capacity(expected);
        for index in 0..expected {
            ca.push(
                PerceiverAttentionCA::load(config, vb.pp(format!("pulid_ca.{index}")))
                    .with_context(|| format!("loading pulid_ca.{index}"))?,
            );
        }
        Ok(Self {
            ca,
            double_injections,
            config,
            dtype,
        })
    }

    /// Number of cross-attention modules.
    pub fn len(&self) -> usize {
        self.ca.len()
    }

    /// Always false in practice — an adapter with no modules is refused at
    /// load — but clippy asks for it beside `len`.
    pub fn is_empty(&self) -> bool {
        self.ca.is_empty()
    }

    /// Modules the double-stream loop consumes; also the first single-stream
    /// module's index.
    pub fn double_injections(&self) -> usize {
        self.double_injections
    }

    /// The dtype the modules were loaded at.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Geometry the modules were built with.
    pub fn config(&self) -> PulidAdapterConfig {
        self.config
    }

    /// One module, by its `ca_idx`. Exposed so a parity test can drive a
    /// single module against the upstream golden without a transformer.
    pub fn module(&self, index: usize) -> Option<&PerceiverAttentionCA> {
        self.ca.get(index)
    }

    /// `img + id_weight * ca[index](id_embeds, img)`.
    fn inject(
        &self,
        index: usize,
        id_embeds: &Tensor,
        image_tokens: &Tensor,
        id_weight: f32,
    ) -> candle_core::Result<Tensor> {
        let module = self.ca.get(index).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "PuLID cross-attention module {index} is out of range ({} loaded)",
                self.ca.len()
            ))
        })?;
        let delta = module.forward(id_embeds, image_tokens)?;
        image_tokens + (delta * f64::from(id_weight))?
    }
}

/// The identity signal a render conditions on.
///
/// A newtype rather than a bare `Tensor` so the `[1, 32, 2048]` contract is
/// checked once, at the boundary, instead of being re-asserted at each of the
/// twenty injection sites. This is the seam the face extractor plugs into:
/// #1223 replaces the loaders below with the real
/// detector → ArcFace → EVA-CLIP → IDFormer stack and everything downstream is
/// unchanged.
#[derive(Debug, Clone)]
pub struct IdentityEmbedding {
    tensor: Tensor,
}

impl IdentityEmbedding {
    /// Wrap a tensor, accepting `[32, 2048]` or `[1, 32, 2048]`.
    pub fn new(tensor: Tensor) -> Result<Self> {
        let tensor = match tensor.dims() {
            [ID_TOKENS, ID_TOKEN_DIM] => tensor.unsqueeze(0)?,
            [1, ID_TOKENS, ID_TOKEN_DIM] => tensor,
            other => bail!(
                "PuLID identity embedding must be [{ID_TOKENS}, {ID_TOKEN_DIM}] or \
                 [1, {ID_TOKENS}, {ID_TOKEN_DIM}], got {other:?}"
            ),
        };
        Ok(Self {
            tensor: tensor.to_dtype(DType::F32)?,
        })
    }

    /// Read the embedding from a safetensors file.
    ///
    /// `name` defaults to `pulid_id`, the tensor name stable-diffusion.cpp's
    /// `.pulidembd` container uses (`tmp/sdcpp/docs/pulid.md`), so the same
    /// identity can be pushed through both implementations for a seed-matched
    /// comparison.
    pub fn from_safetensors(path: &Path, name: Option<&str>) -> Result<Self> {
        let name = name.unwrap_or("pulid_id");
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .with_context(|| format!("reading identity embedding {}", path.display()))?;
        let tensor = tensors.get(name).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "identity embedding {} has no tensor '{name}' (found: {:?})",
                path.display(),
                tensors.keys().collect::<Vec<_>>()
            )
        })?;
        Self::new(tensor)
    }

    /// Read the embedding from stable-diffusion.cpp's `.pulidembd` gguf
    /// container, whose single `pulid_id` tensor is stored in ggml axis order.
    ///
    /// Both orientations are accepted because "ggml order" is a property of
    /// the writer, not of the format: a `[2048, 32]` read is transposed back
    /// to token-major rather than being refused.
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("opening identity embedding {}", path.display()))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("reading gguf {}", path.display()))?;
        let tensor = content
            .tensor(&mut file, "pulid_id", &Device::Cpu)
            .with_context(|| format!("gguf {} has no 'pulid_id' tensor", path.display()))?
            .dequantize(&Device::Cpu)?;
        let tensor = if tensor.dims() == [ID_TOKEN_DIM, ID_TOKENS] {
            tensor.t()?.contiguous()?
        } else {
            tensor
        };
        Self::new(tensor)
    }

    /// The `[1, 32, 2048]` tensor, on the CPU in f32.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
}

/// Identity conditioning frozen for one render.
///
/// `id_weight` and `start_step` come from
/// `mold_core::identity::effective_id_weight` / `effective_id_start_step`, so
/// the value applied here is the value the request contract advertised.
#[derive(Debug, Clone)]
pub struct PulidContext {
    /// `[1, 32, 2048]` on the transformer's device, in its working dtype.
    pub id_embeds: Tensor,
    pub id_weight: f32,
    pub start_step: usize,
}

impl PulidContext {
    /// Move an embedding onto the transformer's device and dtype.
    pub fn new(
        embedding: &IdentityEmbedding,
        id_weight: f32,
        start_step: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            id_embeds: embedding.tensor().to_device(device)?.to_dtype(dtype)?,
            id_weight,
            start_step,
        })
    }

    /// Whether this context contributes anything at `step`.
    ///
    /// The two ways it does not are the falsification cases from
    /// `tmp/sdcpp/docs/pulid.md`: an explicit zero weight, and a delayed start.
    pub fn active_at_step(&self, step: usize) -> bool {
        self.id_weight != 0.0 && step >= self.start_step
    }
}

/// The adapter and the context, paired for the length of one denoise loop.
///
/// Kept separate from [`PulidContext`] because the adapter belongs to the
/// loaded transformer and the context belongs to the request; the denoise loop
/// is the only place both are in scope.
#[derive(Debug, Clone, Copy)]
pub struct PulidRuntime<'a> {
    adapter: &'a PulidAdapter,
    context: &'a PulidContext,
}

impl<'a> PulidRuntime<'a> {
    pub fn new(adapter: &'a PulidAdapter, context: &'a PulidContext) -> Self {
        Self { adapter, context }
    }

    /// The hook for `step`, or `None` when identity contributes nothing.
    ///
    /// `None` is the whole gate. The denoise loop answers it by calling the
    /// variant's ordinary `forward`, so a gated step executes the exact code a
    /// build with no identity request executes — which is what makes
    /// bit-identity structural rather than numerical.
    pub fn hook_for_step(&self, step: usize) -> Option<PulidBlockHook<'a>> {
        self.context.active_at_step(step).then_some(PulidBlockHook {
            adapter: self.adapter,
            context: self.context,
        })
    }

    pub fn adapter(&self) -> &'a PulidAdapter {
        self.adapter
    }

    pub fn context(&self) -> &'a PulidContext {
        self.context
    }
}

/// Injects identity features at the trained block intervals.
///
/// Implements the candle fork's [`BlockHook`] so the dense and quantized
/// upstream transformers can drive it through `forward_with_hook`; mold's own
/// offloaded and bypass transformers call the same two methods directly, so
/// all four variants share one injection policy.
#[derive(Debug, Clone, Copy)]
pub struct PulidBlockHook<'a> {
    adapter: &'a PulidAdapter,
    context: &'a PulidContext,
}

impl PulidBlockHook<'_> {
    /// Which module a double-stream block at `index` uses, if any.
    ///
    /// `flux/model.py:122` — every `DOUBLE_INTERVAL`-th block, counting from
    /// zero, and `ca_idx` advances only on an injection.
    pub fn double_ca_index(&self, index: usize) -> Option<usize> {
        index
            .is_multiple_of(DOUBLE_INTERVAL)
            .then_some(index / DOUBLE_INTERVAL)
            .filter(|ca| *ca < self.adapter.double_injections)
    }

    /// Which module a single-stream block at `index` uses, if any.
    ///
    /// `flux/model.py:143`. The index continues where the double-stream loop
    /// left off — for FLUX.1 that is modules 0–9 for double blocks and 10–19
    /// for single ones.
    pub fn single_ca_index(&self, index: usize) -> Option<usize> {
        index
            .is_multiple_of(SINGLE_INTERVAL)
            .then(|| self.adapter.double_injections + index / SINGLE_INTERVAL)
            .filter(|ca| *ca < self.adapter.len())
    }
}

impl BlockHook for PulidBlockHook<'_> {
    fn after_double_block(
        &self,
        index: usize,
        img: &Tensor,
        _txt: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let Some(ca) = self.double_ca_index(index) else {
            return Ok(None);
        };
        self.adapter
            .inject(ca, &self.context.id_embeds, img, self.context.id_weight)
            .map(Some)
    }

    fn after_single_block(
        &self,
        index: usize,
        txt_len: usize,
        xs: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let Some(ca) = self.single_ca_index(index) else {
            return Ok(None);
        };
        // Only the image slice is conditioned; the text prefix is spliced back
        // untouched (`flux/model.py:141-146`, `flux.hpp:1141-1159`).
        let txt = xs.i((.., ..txt_len))?;
        let img = xs.i((.., txt_len..))?.contiguous()?;
        let img = self
            .adapter
            .inject(ca, &self.context.id_embeds, &img, self.context.id_weight)?;
        Tensor::cat(&[&txt, &img], 1).map(Some)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};
    use std::collections::HashMap;

    /// FLUX.1's block counts (`candle_transformers::models::flux::Config::dev`).
    const FLUX1_DEPTH: usize = 19;
    const FLUX1_DEPTH_SINGLE: usize = 38;

    pub(crate) fn tiny_config() -> PulidAdapterConfig {
        PulidAdapterConfig {
            dim: 32,
            dim_head: 8,
            heads: 2,
            kv_dim: 16,
        }
    }

    /// A deterministic adapter over `config`, with the module count `depth` /
    /// `depth_single_blocks` implies. Values are a fixed ramp so two builds in
    /// one process are identical.
    pub(crate) fn synthetic_adapter(
        config: PulidAdapterConfig,
        depth: usize,
        depth_single_blocks: usize,
        dtype: DType,
        device: &Device,
    ) -> PulidAdapter {
        let (double, single) = injection_counts(depth, depth_single_blocks);
        let inner = config.dim_head * config.heads;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let mut seed = 1u64;
        let mut ramp = |shape: &[usize]| -> Tensor {
            let count: usize = shape.iter().product();
            let values: Vec<f32> = (0..count)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect();
            Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
        };
        for index in 0..(double + single) {
            let prefix = format!("pulid_ca.{index}");
            tensors.insert(format!("{prefix}.norm1.weight"), ramp(&[config.kv_dim]));
            tensors.insert(format!("{prefix}.norm1.bias"), ramp(&[config.kv_dim]));
            tensors.insert(format!("{prefix}.norm2.weight"), ramp(&[config.dim]));
            tensors.insert(format!("{prefix}.norm2.bias"), ramp(&[config.dim]));
            tensors.insert(format!("{prefix}.to_q.weight"), ramp(&[inner, config.dim]));
            tensors.insert(
                format!("{prefix}.to_kv.weight"),
                ramp(&[inner * 2, config.kv_dim]),
            );
            tensors.insert(
                format!("{prefix}.to_out.weight"),
                ramp(&[config.dim, inner]),
            );
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        PulidAdapter::from_var_builder(vb, config, depth, depth_single_blocks).unwrap()
    }

    pub(crate) fn synthetic_context(
        config: PulidAdapterConfig,
        id_weight: f32,
        start_step: usize,
        dtype: DType,
        device: &Device,
    ) -> PulidContext {
        let count = ID_TOKENS * config.kv_dim;
        let values: Vec<f32> = (0..count).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect();
        let tensor = Tensor::from_vec(values, (1, ID_TOKENS, config.kv_dim), device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        PulidContext {
            id_embeds: tensor,
            id_weight,
            start_step,
        }
    }

    #[test]
    fn flux1_needs_twenty_modules_split_ten_and_ten() {
        assert_eq!(injection_counts(FLUX1_DEPTH, FLUX1_DEPTH_SINGLE), (10, 10));
    }

    #[test]
    fn injection_counts_round_up_like_the_oracle() {
        // `flux.hpp:994-996` uses `(depth + interval - 1) / interval`.
        for depth in 1..40usize {
            for single in 1..40usize {
                assert_eq!(
                    injection_counts(depth, single),
                    // `flux.hpp:994-996` spells the same rounding as
                    // `(depth + interval - 1) / interval`; clippy asks for the
                    // std spelling, which is the identical arithmetic.
                    (
                        depth.div_ceil(DOUBLE_INTERVAL),
                        single.div_ceil(SINGLE_INTERVAL)
                    ),
                    "depth={depth} single={single}"
                );
            }
        }
    }

    #[test]
    fn a_module_count_that_disagrees_with_the_transformer_is_refused() {
        let config = tiny_config();
        // Build for a 4/8 transformer (2 + 2 modules) then ask for FLUX.1's shape.
        let (double, single) = injection_counts(4, 8);
        let inner = config.dim_head * config.heads;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for index in 0..(double + single) {
            let prefix = format!("pulid_ca.{index}");
            for (name, shape) in [
                ("norm1.weight", vec![config.kv_dim]),
                ("norm1.bias", vec![config.kv_dim]),
                ("norm2.weight", vec![config.dim]),
                ("norm2.bias", vec![config.dim]),
                ("to_q.weight", vec![inner, config.dim]),
                ("to_kv.weight", vec![inner * 2, config.kv_dim]),
                ("to_out.weight", vec![config.dim, inner]),
            ] {
                tensors.insert(
                    format!("{prefix}.{name}"),
                    Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap(),
                );
            }
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let error = PulidAdapter::from_var_builder(vb, config, FLUX1_DEPTH, FLUX1_DEPTH_SINGLE)
            .expect_err("a 4-module adapter cannot drive a 19/38 transformer");
        let message = error.to_string();
        assert!(message.contains("carries 4"), "{message}");
        assert!(message.contains("needs 20"), "{message}");
    }

    #[test]
    fn ca_indices_run_double_zero_to_nine_then_single_ten_to_nineteen() {
        let device = Device::Cpu;
        let adapter = synthetic_adapter(
            tiny_config(),
            FLUX1_DEPTH,
            FLUX1_DEPTH_SINGLE,
            DType::F32,
            &device,
        );
        let context = synthetic_context(tiny_config(), 1.0, 0, DType::F32, &device);
        let runtime = PulidRuntime::new(&adapter, &context);
        let hook = runtime.hook_for_step(0).expect("active");

        let doubles: Vec<usize> = (0..FLUX1_DEPTH)
            .filter_map(|i| hook.double_ca_index(i))
            .collect();
        assert_eq!(doubles, (0..10).collect::<Vec<_>>());
        assert_eq!(
            (0..FLUX1_DEPTH)
                .filter(|i| hook.double_ca_index(*i).is_some())
                .count(),
            10
        );

        let singles: Vec<usize> = (0..FLUX1_DEPTH_SINGLE)
            .filter_map(|i| hook.single_ca_index(i))
            .collect();
        assert_eq!(singles, (10..20).collect::<Vec<_>>());

        // Every module is used exactly once across the whole forward.
        let mut all = doubles;
        all.extend(singles);
        assert_eq!(all, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn odd_double_and_non_multiple_single_blocks_never_index_past_the_stack() {
        let device = Device::Cpu;
        let adapter = synthetic_adapter(tiny_config(), 5, 6, DType::F32, &device);
        assert_eq!(adapter.len(), 3 + 2);
        let context = synthetic_context(tiny_config(), 1.0, 0, DType::F32, &device);
        let runtime = PulidRuntime::new(&adapter, &context);
        let hook = runtime.hook_for_step(0).expect("active");
        let doubles: Vec<usize> = (0..5).filter_map(|i| hook.double_ca_index(i)).collect();
        let singles: Vec<usize> = (0..6).filter_map(|i| hook.single_ca_index(i)).collect();
        assert_eq!(doubles, vec![0, 1, 2]);
        assert_eq!(singles, vec![3, 4]);
    }

    #[test]
    fn zero_weight_and_a_delayed_start_yield_no_hook_at_all() {
        let device = Device::Cpu;
        let adapter = synthetic_adapter(tiny_config(), 4, 8, DType::F32, &device);

        let zero = synthetic_context(tiny_config(), 0.0, 0, DType::F32, &device);
        assert!(PulidRuntime::new(&adapter, &zero)
            .hook_for_step(0)
            .is_none());
        assert!(PulidRuntime::new(&adapter, &zero)
            .hook_for_step(9)
            .is_none());

        let delayed = synthetic_context(tiny_config(), 1.0, 3, DType::F32, &device);
        let runtime = PulidRuntime::new(&adapter, &delayed);
        for step in 0..3 {
            assert!(runtime.hook_for_step(step).is_none(), "step {step}");
        }
        for step in 3..6 {
            assert!(runtime.hook_for_step(step).is_some(), "step {step}");
        }
    }

    #[test]
    fn the_single_stream_hook_leaves_the_text_prefix_untouched() {
        let device = Device::Cpu;
        let config = tiny_config();
        let adapter = synthetic_adapter(config, 4, 8, DType::F32, &device);
        let context = synthetic_context(config, 1.0, 0, DType::F32, &device);
        let hook = PulidRuntime::new(&adapter, &context)
            .hook_for_step(0)
            .expect("active");

        let txt_len = 5usize;
        let img_len = 7usize;
        let total = txt_len + img_len;
        let values: Vec<f32> = (0..total * config.dim).map(|i| i as f32 / 10.0).collect();
        let xs = Tensor::from_vec(values, (1, total, config.dim), &device).unwrap();

        let out = hook
            .after_single_block(0, txt_len, &xs)
            .unwrap()
            .expect("block 0 is an injection site");
        assert_eq!(out.dims(), xs.dims());

        let before_txt = xs.i((.., ..txt_len)).unwrap().flatten_all().unwrap();
        let after_txt = out.i((.., ..txt_len)).unwrap().flatten_all().unwrap();
        assert_eq!(
            before_txt.to_vec1::<f32>().unwrap(),
            after_txt.to_vec1::<f32>().unwrap(),
            "the text prefix must be spliced back bit-for-bit"
        );

        let before_img = xs.i((.., txt_len..)).unwrap().flatten_all().unwrap();
        let after_img = out.i((.., txt_len..)).unwrap().flatten_all().unwrap();
        assert_ne!(
            before_img.to_vec1::<f32>().unwrap(),
            after_img.to_vec1::<f32>().unwrap(),
            "the image slice must actually be conditioned"
        );

        // A non-injection index changes nothing at all.
        assert!(hook.after_single_block(1, txt_len, &xs).unwrap().is_none());
        assert!(hook.after_single_block(3, txt_len, &xs).unwrap().is_none());
    }

    #[test]
    fn a_zero_weight_injection_is_arithmetically_the_identity() {
        // The gate makes this unreachable in the denoise loop, but the
        // arithmetic must agree with the gate: `img + 0 * ca(...) == img`.
        let device = Device::Cpu;
        let config = tiny_config();
        let adapter = synthetic_adapter(config, 4, 8, DType::F32, &device);
        let context = synthetic_context(config, 0.0, 0, DType::F32, &device);
        let hook = PulidBlockHook {
            adapter: &adapter,
            context: &context,
        };
        let values: Vec<f32> = (0..6 * config.dim).map(|i| (i % 13) as f32).collect();
        let img = Tensor::from_vec(values, (1, 6, config.dim), &device).unwrap();
        let out = hook
            .after_double_block(0, &img, &img)
            .unwrap()
            .expect("block 0 is an injection site");
        assert_eq!(
            out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            img.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn identity_embedding_accepts_both_ranks_and_refuses_anything_else() {
        let device = Device::Cpu;
        let flat = Tensor::zeros((ID_TOKENS, ID_TOKEN_DIM), DType::F32, &device).unwrap();
        assert_eq!(
            IdentityEmbedding::new(flat).unwrap().tensor().dims(),
            [1, ID_TOKENS, ID_TOKEN_DIM]
        );
        let batched = Tensor::zeros((1, ID_TOKENS, ID_TOKEN_DIM), DType::F32, &device).unwrap();
        assert_eq!(
            IdentityEmbedding::new(batched).unwrap().tensor().dims(),
            [1, ID_TOKENS, ID_TOKEN_DIM]
        );
        let wrong = Tensor::zeros((2, ID_TOKENS, ID_TOKEN_DIM), DType::F32, &device).unwrap();
        assert!(IdentityEmbedding::new(wrong).is_err());
        let transposed = Tensor::zeros((ID_TOKEN_DIM, ID_TOKENS), DType::F32, &device).unwrap();
        assert!(
            IdentityEmbedding::new(transposed).is_err(),
            "a transposed embedding is a real mistake, not a layout to guess at"
        );
    }

    /// A fixed ramp, so a round trip through a container is checkable
    /// element-by-element rather than only by shape.
    fn embedding_ramp(device: &Device) -> (Vec<f32>, Tensor) {
        let values: Vec<f32> = (0..ID_TOKENS * ID_TOKEN_DIM)
            .map(|i| (i % 251) as f32 / 251.0 - 0.5)
            .collect();
        let tensor = Tensor::from_vec(values.clone(), (ID_TOKENS, ID_TOKEN_DIM), device).unwrap();
        (values, tensor)
    }

    #[test]
    fn an_embedding_round_trips_through_safetensors() {
        let device = Device::Cpu;
        let (values, tensor) = embedding_ramp(&device);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.safetensors");
        candle_core::safetensors::save(
            &std::collections::HashMap::from([("pulid_id".to_string(), tensor)]),
            &path,
        )
        .unwrap();

        let loaded = IdentityEmbedding::from_safetensors(&path, None).unwrap();
        assert_eq!(loaded.tensor().dims(), [1, ID_TOKENS, ID_TOKEN_DIM]);
        assert_eq!(
            loaded
                .tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            values
        );

        let missing = IdentityEmbedding::from_safetensors(&path, Some("not_there"))
            .expect_err("a wrong tensor name is an error, not an empty identity");
        assert!(missing.to_string().contains("not_there"), "{missing}");
    }

    /// stable-diffusion.cpp's `.pulidembd` container: one `pulid_id` tensor in
    /// a gguf, written by `scripts/pulid_extract_id.py` through
    /// `gguf.GGUFWriter(arch="pulid")`. Reading it is what lets the identical
    /// identity be pushed through both implementations for a seed-matched
    /// comparison, so both axis orders a writer might produce are covered.
    #[test]
    fn an_embedding_round_trips_through_the_sdcpp_gguf_container() {
        let device = Device::Cpu;
        let (values, tensor) = embedding_ramp(&device);
        let dir = tempfile::tempdir().unwrap();

        for (label, stored) in [
            ("token-major", tensor.clone()),
            ("ggml-order", tensor.t().unwrap().contiguous().unwrap()),
        ] {
            let path = dir.path().join(format!("{label}.pulidembd"));
            let quantized = mold_candle::quantized::quantize_onto(
                &stored,
                candle_core::quantized::GgmlDType::F32,
                &device,
            )
            .unwrap();
            let mut file = std::fs::File::create(&path).unwrap();
            candle_core::quantized::gguf_file::write(
                &mut file,
                &[(
                    "general.architecture",
                    &candle_core::quantized::gguf_file::Value::String("pulid".to_string()),
                )],
                &[("pulid_id", &quantized)],
            )
            .unwrap();
            drop(file);

            let loaded = IdentityEmbedding::from_gguf(&path).unwrap();
            assert_eq!(
                loaded.tensor().dims(),
                [1, ID_TOKENS, ID_TOKEN_DIM],
                "{label}"
            );
            assert_eq!(
                loaded
                    .tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                values,
                "{label}: a transposed store must come back token-major"
            );
        }
    }

    #[test]
    fn a_var_map_backed_adapter_reports_its_geometry() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // `from_varmap` materializes on demand, so `contains_tensor` is false
        // until something asks — which is exactly the "no modules present"
        // case the count check must refuse rather than silently accept.
        let error = PulidAdapter::from_var_builder(vb, tiny_config(), 4, 8)
            .expect_err("an empty VarMap carries no pulid_ca modules");
        assert!(error.to_string().contains("carries 0"), "{error}");

        let adapter = synthetic_adapter(tiny_config(), 4, 8, DType::F32, &device);
        assert_eq!(adapter.len(), 4);
        assert_eq!(adapter.double_injections(), 2);
        assert_eq!(adapter.dtype(), DType::F32);
        assert_eq!(adapter.config(), tiny_config());
        assert!(!adapter.is_empty());
    }
}
