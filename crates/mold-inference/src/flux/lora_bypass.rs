//! Bypass-mode LoRA primitive for FLUX.
//!
//! Mirrors the LTX-2 [`crate::ltx2::lora::LinearLoraAdapter`] /
//! [`crate::ltx2::model::video_transformer::LtxLinear`] pattern and
//! ComfyUI's `comfy/weight_adapter/bypass.py`: never modify base weights,
//! apply the LoRA delta at forward time on top of the base matmul output.
//!
//! ```text
//! Linear.forward(x)        = x @ W.T + b
//! LoraLinear.forward(x)    = x @ W.T + b + Σᵢ scaleᵢ · (x @ Aᵢ.T) @ Bᵢ.T
//! ```
//!
//! For fused-QKV targets (e.g. FLUX `img_attn.qkv` packs Q‖K‖V into one
//! linear), an adapter that targets only Q writes its delta into the
//! corresponding row slice of the output via `narrow` + `cat`.
//!
//! This module is the building block that makes "swap LoRA" cheap — no
//! base-weight rebuilds, no dequant→merge→requant on quantized models —
//! at the cost of a small per-step matmul (the adapters are tiny: rank
//! is typically 4–32).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use candle_transformers::quantized_nn::Linear as QuantizedLinear;
use std::collections::HashMap;

/// Slice a [`LinearLoraAdapter`]'s contribution into a fused output.
///
/// `offset` is the row index (in output / "out_features" axis) where the
/// adapter's delta starts; `length` is the number of rows it owns. For
/// QKV-fused linears (FLUX `img_attn.qkv`), Q lives at `[0, h)`, K at
/// `[h, 2h)`, V at `[2h, 3h)`. The adapter's `up` matrix has `length`
/// rows, not `out_features`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FusedSlice {
    pub offset: usize,
    pub length: usize,
}

/// A LoRA delta applied at forward time, never merged into base weights.
///
/// `down` is the (rank, in_features) matrix usually called `lora_A` /
/// `lora_down`. `up` is (out_features_or_slice_length, rank), usually
/// called `lora_B` / `lora_up`. `scale` already includes the alpha/rank
/// rescale, so `apply()` just multiplies through.
///
/// Tensors live wherever you put them — typically GPU. They're tiny
/// (rank-4 to rank-32 LoRAs are a few MB total per adapter), so it's fine
/// to keep them GPU-resident even when the host transformer is offloading
/// blocks back to CPU between steps.
#[derive(Clone, Debug)]
pub struct LinearLoraAdapter {
    pub down: Tensor,
    pub up: Tensor,
    pub scale: f32,
    pub fused_slice: Option<FusedSlice>,
}

impl LinearLoraAdapter {
    /// Compute `out += scale · (x @ down.T) @ up.T` in `out`'s dtype.
    ///
    /// For a fused target, the contribution lands in
    /// `out[..., offset..offset+length]` — rows outside that range are
    /// left untouched. Returns the new tensor (candle is functional).
    pub fn apply(&self, x: &Tensor, out: &Tensor) -> Result<Tensor> {
        // Nothing to do for zero-scale adapters; skip the matmul so a
        // user-disabled-but-still-attached adapter is genuinely free.
        if self.scale == 0.0 {
            return Ok(out.clone());
        }

        let dtype = out.dtype();
        let device = out.device();

        let down = adapter_to_runtime(&self.down, device, dtype)?;
        let up = adapter_to_runtime(&self.up, device, dtype)?;

        // delta = (x @ down.T) @ up.T, shape [..., adapter_out_rows]
        let delta = matmul_through_lora(x, &down, &up)?;

        // scale = adapter scale; apply via affine for one fused kernel.
        let delta = delta.affine(self.scale as f64, 0.0)?;

        match self.fused_slice {
            None => Ok(out.broadcast_add(&delta)?),
            Some(slice) => add_into_slice(out, &delta, slice),
        }
    }
}

/// A `Linear` (BF16 / F32) or `quantized_nn::Linear` (GGUF) that may
/// carry zero or more LoRA adapters.
///
/// `Plain` / `Quantized` are bit-identical to the underlying inner
/// linear. `WithAdapters*` runs the inner forward, then layers each
/// adapter on top. The split exists so callers don't pay any forward
/// overhead when no LoRA is active — the common case.
///
/// The Plain↔Quantized split lets the offload path keep using the BF16
/// `candle_nn::Linear` (it streams base weights CPU↔GPU each step) while
/// the GGUF path stores `quantized_nn::Linear` permanently on GPU. Both
/// share the same `LinearLoraAdapter` math because adapter weights are
/// always small dense BF16/F32 tensors regardless of the base.
#[derive(Clone, Debug)]
pub enum LoraLinear {
    Plain(Linear),
    WithAdapters {
        inner: Linear,
        adapters: Vec<LinearLoraAdapter>,
    },
    Quantized(QuantizedLinear),
    WithAdaptersQuantized {
        inner: QuantizedLinear,
        adapters: Vec<LinearLoraAdapter>,
    },
}

#[allow(dead_code)]
impl LoraLinear {
    /// Wrap a `Linear` in the no-adapter variant. Forward is identical.
    pub fn plain(inner: Linear) -> Self {
        Self::Plain(inner)
    }

    /// Wrap a `quantized_nn::Linear` in the no-adapter variant.
    pub fn quantized(inner: QuantizedLinear) -> Self {
        Self::Quantized(inner)
    }

    /// Read-only access to the underlying BF16/F32 `Linear`. Panics if
    /// called on a quantized variant — those use [`Self::inner_quantized`]
    /// instead. Legacy callers (e.g. the offload path's `to_device` copy)
    /// only ever hold the BF16 variant so this contract preserves their
    /// existing return-type ergonomics without `.unwrap()` noise at every
    /// call site.
    pub fn inner(&self) -> &Linear {
        match self {
            Self::Plain(l) => l,
            Self::WithAdapters { inner, .. } => inner,
            Self::Quantized(_) | Self::WithAdaptersQuantized { .. } => {
                panic!("LoraLinear::inner() called on a Quantized variant — use inner_quantized()")
            }
        }
    }

    /// Read-only access to the underlying quantized linear, or `None`
    /// for BF16/F32 variants.
    pub fn inner_quantized(&self) -> Option<&QuantizedLinear> {
        match self {
            Self::Quantized(q) => Some(q),
            Self::WithAdaptersQuantized { inner, .. } => Some(inner),
            Self::Plain(_) | Self::WithAdapters { .. } => None,
        }
    }

    /// Replace the adapter stack. Empty `adapters` collapses to the
    /// no-adapter variant so future forward calls skip the per-adapter
    /// loop entirely. Preserves the Plain/Quantized kind of the inner.
    pub fn set_adapters(&mut self, adapters: Vec<LinearLoraAdapter>) {
        let is_quantized = matches!(
            self,
            Self::Quantized(_) | Self::WithAdaptersQuantized { .. }
        );
        if is_quantized {
            let inner = self.inner_quantized().unwrap().clone();
            if adapters.is_empty() {
                *self = Self::Quantized(inner);
            } else {
                *self = Self::WithAdaptersQuantized { inner, adapters };
            }
        } else {
            let inner = self.inner().clone();
            if adapters.is_empty() {
                *self = Self::Plain(inner);
            } else {
                *self = Self::WithAdapters { inner, adapters };
            }
        }
    }

    /// Drop all adapters — `set_adapters(vec![])` shorthand.
    pub fn clear_adapters(&mut self) {
        self.set_adapters(Vec::new());
    }

    /// `inner.forward(x) + Σ adapter.apply(x, out)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(l) => Ok(<Linear as candle_core::Module>::forward(l, x)?),
            Self::WithAdapters { inner, adapters } => {
                let mut out = <Linear as candle_core::Module>::forward(inner, x)?;
                for adapter in adapters {
                    out = adapter.apply(x, &out)?;
                }
                Ok(out)
            }
            Self::Quantized(q) => Ok(<QuantizedLinear as candle_core::Module>::forward(q, x)?),
            Self::WithAdaptersQuantized { inner, adapters } => {
                let mut out = <QuantizedLinear as candle_core::Module>::forward(inner, x)?;
                for adapter in adapters {
                    out = adapter.apply(x, &out)?;
                }
                Ok(out)
            }
        }
    }
}

impl candle_core::Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Surface anyhow errors through candle_core::Error to keep the
        // `Module` blanket impls (e.g. `Tensor::apply`) usable.
        Self::forward(self, x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

// ─── helpers ──────────────────────────────────────────────────────────

/// Move + cast an adapter tensor onto the runtime device/dtype if needed.
/// LoRA tensors are typically loaded on CPU as F32; runtime is GPU/BF16.
fn adapter_to_runtime(t: &Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
    let t = if t.device().same_device(device) {
        t.clone()
    } else {
        t.to_device(device)?
    };
    if t.dtype() != dtype {
        Ok(t.to_dtype(dtype)?)
    } else {
        Ok(t)
    }
}

/// Run `(x @ down.T) @ up.T` reshaping to a 2-D matmul where possible.
/// Mirrors the LTX-2 helper so we share the perf characteristics.
fn matmul_through_lora(x: &Tensor, down: &Tensor, up: &Tensor) -> Result<Tensor> {
    let down_t = down.t()?;
    let up_t = up.t()?;
    Ok(match *x.dims() {
        [b0, b1, t, h] => x
            .reshape((b0 * b1 * t, h))?
            .matmul(&down_t)?
            .matmul(&up_t)?
            .reshape((b0, b1, t, ()))?,
        [b, t, h] => x
            .reshape((b * t, h))?
            .matmul(&down_t)?
            .matmul(&up_t)?
            .reshape((b, t, ()))?,
        _ => x.matmul(&down_t)?.matmul(&up_t)?,
    })
}

/// Per-Linear adapter store. Keys are FLUX candle tensor names (e.g.
/// `double_blocks.0.img_attn.qkv.weight`). Values are the bypass
/// adapters that should fire each time that Linear runs forward.
///
/// All `down`/`up` tensors live on the runtime device — typically GPU —
/// because they're tiny (rank-32 LoRAs are a few MB total per adapter)
/// and we don't want a CPU↔GPU round-trip in the per-step path.
#[derive(Clone, Default, Debug)]
pub(crate) struct LoraRegistry {
    by_key: HashMap<String, Vec<LinearLoraAdapter>>,
}

impl LoraRegistry {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Number of (key, stack) entries — useful for progress logging.
    pub(crate) fn len(&self) -> usize {
        self.by_key.len()
    }

    /// True when no adapters are installed for any tensor.
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    /// All adapters that fire for `key`, or an empty slice. Cheap enough
    /// to call from each block's `to_device` path.
    pub(crate) fn adapters_for(&self, key: &str) -> &[LinearLoraAdapter] {
        self.by_key.get(key).map(Vec::as_slice).unwrap_or(&[])
    }

    fn push(&mut self, key: String, adapter: LinearLoraAdapter) {
        self.by_key.entry(key).or_default().push(adapter);
    }
}

/// Build a [`LoraRegistry`] from an ordered LoRA stack.
///
/// Reuses [`super::lora::map_lora_key`] / [`super::lora::fused_slice_range`]
/// so we accept the same diffusers/Kohya naming as the merge-time path —
/// no second lookup table to keep in sync. `device`/`dtype` say where
/// the adapters should live; pass GPU+BF16 for the standard FLUX flow.
///
/// `linear_out_dims` maps candle tensor key → fused tensor's full row
/// count (e.g. 21504 for FLUX single-block `linear1.weight`). It's
/// needed only for [`crate::flux::lora::LoraTarget::FusedSlice`] cases
/// to compute the absolute slice offset; pass an empty map if you only
/// expect Direct targets.
pub(crate) fn build_registry(
    specs: &[super::lora::LoraSpec<'_>],
    linear_out_dims: &HashMap<String, usize>,
    device: &Device,
    dtype: DType,
) -> Result<LoraRegistry> {
    let mut registry = LoraRegistry::new();
    for spec in specs {
        for (diffusers_key, lora_layer) in &spec.adapter.layers {
            let target = match super::lora::map_lora_key(diffusers_key) {
                Some(t) => t,
                None => continue,
            };

            // Effective scale folds in alpha/rank just like merge mode.
            let layer_rank = lora_layer.a.dims()[0] as f64;
            let effective_scale = match lora_layer.alpha {
                Some(alpha) => spec.scale * alpha / layer_rank,
                None => spec.scale,
            };

            let down = lora_layer.a.to_device(device)?.to_dtype(dtype)?;
            let up = lora_layer.b.to_device(device)?.to_dtype(dtype)?;

            let (candle_key, fused_slice) = match target {
                super::lora::LoraTarget::Direct { candle_key } => (candle_key, None),
                super::lora::LoraTarget::FusedSlice {
                    candle_key,
                    component,
                    num_components,
                } => {
                    let base_rows = match linear_out_dims.get(&candle_key) {
                        Some(n) => *n,
                        None => {
                            tracing::warn!(
                                key = candle_key.as_str(),
                                "fused-slice target unknown to bypass registry, skipping"
                            );
                            continue;
                        }
                    };
                    let lora_out_dim = up.dim(0)?;
                    let (offset, length) = super::lora::fused_slice_range(
                        base_rows,
                        lora_out_dim,
                        component,
                        num_components,
                    );
                    if offset + length > base_rows {
                        tracing::warn!(
                            key = candle_key.as_str(),
                            offset,
                            length,
                            base_rows,
                            "fused slice out of bounds, skipping"
                        );
                        continue;
                    }
                    (candle_key, Some(FusedSlice { offset, length }))
                }
            };

            registry.push(
                candle_key,
                LinearLoraAdapter {
                    down,
                    up,
                    scale: effective_scale as f32,
                    fused_slice,
                },
            );
        }
    }
    Ok(registry)
}

/// Add `delta` (shape `[..., length]`) into the `[offset, offset+length)`
/// slice of `out`'s last dim, leaving the rest untouched. Implemented via
/// `narrow` + `cat` because candle has no in-place `slice_assign`.
fn add_into_slice(out: &Tensor, delta: &Tensor, slice: FusedSlice) -> Result<Tensor> {
    let last = out.rank().saturating_sub(1);
    let total = out.dim(last)?;
    if slice.offset + slice.length > total {
        anyhow::bail!(
            "fused-slice [{o}, {o}+{l}) out of bounds for output dim {total}",
            o = slice.offset,
            l = slice.length,
        );
    }
    let middle = out
        .narrow(last, slice.offset, slice.length)?
        .broadcast_add(delta)?;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if slice.offset > 0 {
        parts.push(out.narrow(last, 0, slice.offset)?);
    }
    parts.push(middle);
    let after = slice.offset + slice.length;
    if after < total {
        parts.push(out.narrow(last, after, total - after)?);
    }
    Ok(Tensor::cat(&parts, last)?.contiguous()?)
}

// ─── tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "cuda", feature = "metal"))]
    use candle_core::DType;
    use candle_core::{Device, Module, Tensor};

    /// Build a deterministic Linear with explicit weight/bias on CPU/F32.
    fn make_linear(out_dim: usize, in_dim: usize, with_bias: bool) -> Linear {
        let device = Device::Cpu;
        let weight: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let weight = Tensor::from_vec(weight, (out_dim, in_dim), &device).unwrap();
        let bias = if with_bias {
            let b: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.01).collect();
            Some(Tensor::from_vec(b, (out_dim,), &device).unwrap())
        } else {
            None
        };
        Linear::new(weight, bias)
    }

    fn make_input(batch: usize, tokens: usize, in_dim: usize) -> Tensor {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..batch * tokens * in_dim)
            .map(|i| ((i as f32) * 0.017).cos())
            .collect();
        Tensor::from_vec(data, (batch, tokens, in_dim), &device).unwrap()
    }

    fn make_lora_pair(out_dim: usize, rank: usize, in_dim: usize, salt: f32) -> (Tensor, Tensor) {
        let device = Device::Cpu;
        let down: Vec<f32> = (0..rank * in_dim)
            .map(|i| ((i as f32 + salt) * 0.011).sin())
            .collect();
        let up: Vec<f32> = (0..out_dim * rank)
            .map(|i| ((i as f32 + salt) * 0.019).cos())
            .collect();
        let down = Tensor::from_vec(down, (rank, in_dim), &device).unwrap();
        let up = Tensor::from_vec(up, (out_dim, rank), &device).unwrap();
        (down, up)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let diff = (a - b).unwrap().abs().unwrap();
        diff.flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn test_plain_linear_forward_unchanged() {
        // Bypass-mode wrapper without adapters must be bit-equal to the
        // raw Linear — otherwise the no-LoRA hot path silently regresses.
        let inner = make_linear(8, 4, true);
        let x = make_input(2, 3, 4);
        let baseline = inner.forward(&x).unwrap();
        let wrapped = LoraLinear::plain(inner.clone()).forward(&x).unwrap();
        let max = max_abs_diff(&baseline, &wrapped);
        assert!(max < 1e-7, "plain-wrap diverged: max abs diff {max}");
    }

    #[test]
    fn test_single_adapter_matches_merged() {
        // The whole point: bypass forward (W·x + s·BA·x) must equal
        // merged-weight forward ((W + s·BA)·x). A single adapter at
        // a non-trivial scale is enough to catch sign/dtype/shape bugs.
        let out_dim = 8;
        let in_dim = 6;
        let rank = 3;
        let inner = make_linear(out_dim, in_dim, true);
        let (down, up) = make_lora_pair(out_dim, rank, in_dim, 1.0);
        let scale = 0.7f32;

        // Build merged Linear: W' = W + scale * (up @ down)
        let merged_delta = up.matmul(&down).unwrap().affine(scale as f64, 0.0).unwrap();
        let merged_w = (inner.weight() + &merged_delta).unwrap();
        let merged = Linear::new(merged_w, inner.bias().cloned());

        let lora = LoraLinear::WithAdapters {
            inner,
            adapters: vec![LinearLoraAdapter {
                down,
                up,
                scale,
                fused_slice: None,
            }],
        };

        let x = make_input(2, 5, in_dim);
        let bypass_out = lora.forward(&x).unwrap();
        let merged_out = merged.forward(&x).unwrap();
        let max = max_abs_diff(&bypass_out, &merged_out);
        assert!(max < 1e-5, "bypass != merged for f32: max abs diff {max}");
    }

    #[test]
    fn test_two_adapters_compose() {
        // Multi-LoRA stacking: two independent A/B pairs must compose
        // additively, exactly as if their deltas had been pre-summed
        // into one merged weight.
        let out_dim = 6;
        let in_dim = 4;
        let rank = 2;
        let inner = make_linear(out_dim, in_dim, false);
        let (d1, u1) = make_lora_pair(out_dim, rank, in_dim, 2.0);
        let (d2, u2) = make_lora_pair(out_dim, rank, in_dim, 7.0);
        let s1 = 0.4f32;
        let s2 = -0.3f32;

        let delta_1 = u1.matmul(&d1).unwrap().affine(s1 as f64, 0.0).unwrap();
        let delta_2 = u2.matmul(&d2).unwrap().affine(s2 as f64, 0.0).unwrap();
        let merged_w = ((inner.weight() + &delta_1).unwrap() + &delta_2).unwrap();
        let merged = Linear::new(merged_w, None);

        let lora = LoraLinear::WithAdapters {
            inner,
            adapters: vec![
                LinearLoraAdapter {
                    down: d1,
                    up: u1,
                    scale: s1,
                    fused_slice: None,
                },
                LinearLoraAdapter {
                    down: d2,
                    up: u2,
                    scale: s2,
                    fused_slice: None,
                },
            ],
        };

        let x = make_input(1, 4, in_dim);
        let bypass_out = lora.forward(&x).unwrap();
        let merged_out = merged.forward(&x).unwrap();
        let max = max_abs_diff(&bypass_out, &merged_out);
        assert!(
            max < 1e-5,
            "two-adapter compose != merged: max abs diff {max}"
        );
    }

    #[test]
    fn test_fused_slice_only_writes_target_slice() {
        // Output dim 12, adapter targets rows [4, 8). Bypass output
        // outside that slice must equal the un-adapted Linear forward
        // exactly; inside the slice it must equal merged forward.
        let out_dim = 12;
        let in_dim = 4;
        let rank = 2;
        let slice = FusedSlice {
            offset: 4,
            length: 4,
        };
        let inner = make_linear(out_dim, in_dim, true);
        let (down, up) = make_lora_pair(slice.length, rank, in_dim, 3.0);
        let scale = 0.5f32;

        let lora = LoraLinear::WithAdapters {
            inner: inner.clone(),
            adapters: vec![LinearLoraAdapter {
                down: down.clone(),
                up: up.clone(),
                scale,
                fused_slice: Some(slice),
            }],
        };

        let x = make_input(1, 3, in_dim);
        let plain_out = inner.forward(&x).unwrap();
        let bypass_out = lora.forward(&x).unwrap();

        // Outside-the-slice rows untouched.
        let before = bypass_out.narrow(2, 0, slice.offset).unwrap();
        let before_ref = plain_out.narrow(2, 0, slice.offset).unwrap();
        let max_before = max_abs_diff(&before, &before_ref);
        assert!(
            max_before < 1e-7,
            "rows < {}: drifted by {max_before}",
            slice.offset
        );

        let after_start = slice.offset + slice.length;
        let after = bypass_out
            .narrow(2, after_start, out_dim - after_start)
            .unwrap();
        let after_ref = plain_out
            .narrow(2, after_start, out_dim - after_start)
            .unwrap();
        let max_after = max_abs_diff(&after, &after_ref);
        assert!(
            max_after < 1e-7,
            "rows >= {after_start}: drifted by {max_after}"
        );

        // Inside-the-slice rows = plain + delta.
        let delta_full = up.matmul(&down).unwrap().affine(scale as f64, 0.0).unwrap();
        // Apply x @ delta_full.T over batched input.
        let expected_inside = {
            let dt = delta_full.t().unwrap();
            let (b, t, _h) = x.dims3().unwrap();
            x.reshape((b * t, in_dim))
                .unwrap()
                .matmul(&dt)
                .unwrap()
                .reshape((b, t, slice.length))
                .unwrap()
        };
        let inside_plain = plain_out.narrow(2, slice.offset, slice.length).unwrap();
        let inside_expected = (inside_plain + expected_inside).unwrap();
        let inside_actual = bypass_out.narrow(2, slice.offset, slice.length).unwrap();
        let max_inside = max_abs_diff(&inside_actual, &inside_expected);
        assert!(max_inside < 1e-5, "slice rows: max abs diff {max_inside}");
    }

    #[test]
    fn test_clear_adapters_returns_to_plain_behavior() {
        // Lifecycle test: install, then clear, then forward must equal
        // the original Linear forward — guarantees `clear_adapters()`
        // is a true reset for engine reuse across requests.
        let inner = make_linear(8, 4, true);
        let (down, up) = make_lora_pair(8, 2, 4, 5.0);
        let mut lora = LoraLinear::WithAdapters {
            inner: inner.clone(),
            adapters: vec![LinearLoraAdapter {
                down,
                up,
                scale: 0.6,
                fused_slice: None,
            }],
        };
        lora.clear_adapters();
        match &lora {
            LoraLinear::Plain(_) => {}
            _ => panic!("clear_adapters must collapse to Plain"),
        }
        let x = make_input(1, 2, 4);
        let max = max_abs_diff(&inner.forward(&x).unwrap(), &lora.forward(&x).unwrap());
        assert!(max < 1e-7, "post-clear diverged: max abs diff {max}");
    }

    #[test]
    fn test_zero_scale_adapter_is_identity() {
        // A scale=0 adapter must be a no-op even though it's still
        // "installed" — important when a multi-LoRA UI lets users
        // toggle individual scales to 0 without removing entries.
        let inner = make_linear(6, 4, false);
        let (down, up) = make_lora_pair(6, 2, 4, 9.0);
        let lora = LoraLinear::WithAdapters {
            inner: inner.clone(),
            adapters: vec![LinearLoraAdapter {
                down,
                up,
                scale: 0.0,
                fused_slice: None,
            }],
        };
        let x = make_input(1, 3, 4);
        let max = max_abs_diff(&inner.forward(&x).unwrap(), &lora.forward(&x).unwrap());
        assert!(max < 1e-7, "zero-scale adapter changed output: {max}");
    }

    #[test]
    fn test_set_adapters_then_replace() {
        // Going from one adapter stack to another via `set_adapters`
        // must equal building the second stack fresh — guards against
        // stale state being retained between LoRA swaps.
        let inner = make_linear(5, 3, true);
        let (d1, u1) = make_lora_pair(5, 2, 3, 11.0);
        let (d2, u2) = make_lora_pair(5, 2, 3, 13.0);
        let mut lora = LoraLinear::WithAdapters {
            inner: inner.clone(),
            adapters: vec![LinearLoraAdapter {
                down: d1,
                up: u1,
                scale: 0.4,
                fused_slice: None,
            }],
        };
        lora.set_adapters(vec![LinearLoraAdapter {
            down: d2.clone(),
            up: u2.clone(),
            scale: 0.55,
            fused_slice: None,
        }]);
        let fresh = LoraLinear::WithAdapters {
            inner,
            adapters: vec![LinearLoraAdapter {
                down: d2,
                up: u2,
                scale: 0.55,
                fused_slice: None,
            }],
        };
        let x = make_input(2, 2, 3);
        let max = max_abs_diff(&lora.forward(&x).unwrap(), &fresh.forward(&x).unwrap());
        assert!(max < 1e-7, "swap-via-set_adapters drifted: {max}");
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    fn test_bf16_tolerance() {
        // BF16 has ~7-bit mantissa; merged-vs-bypass should still match
        // within ~1e-2 because the only divergence is two extra rounds
        // in the bypass path (matmul-down, matmul-up, then add). CPU
        // candle has no BF16 matmul kernel, so this test is gated on
        // a real GPU build — when run on cargo test --features metal
        // (or cuda) it picks the available accelerator.
        let device = if candle_core::Device::cuda_if_available(0).is_ok() {
            candle_core::Device::cuda_if_available(0).unwrap()
        } else if let Ok(m) = candle_core::Device::new_metal(0) {
            m
        } else {
            // Build feature-gated this test, but the runner may still
            // not have a usable accelerator; skip silently then.
            return;
        };
        let out_dim = 8;
        let in_dim = 6;
        let rank = 3;
        let inner_cpu = make_linear(out_dim, in_dim, true);
        let (down_cpu, up_cpu) = make_lora_pair(out_dim, rank, in_dim, 1.0);
        let scale = 0.7f32;

        let to_bf16 = |t: &Tensor| t.to_device(&device).unwrap().to_dtype(DType::BF16).unwrap();
        let inner = Linear::new(
            to_bf16(inner_cpu.weight()),
            inner_cpu.bias().map(|b| to_bf16(b)),
        );
        let down = to_bf16(&down_cpu);
        let up = to_bf16(&up_cpu);
        let merged_delta = up.matmul(&down).unwrap().affine(scale as f64, 0.0).unwrap();
        let merged = Linear::new(
            (inner.weight() + &merged_delta).unwrap(),
            inner.bias().cloned(),
        );

        let lora = LoraLinear::WithAdapters {
            inner,
            adapters: vec![LinearLoraAdapter {
                down,
                up,
                scale,
                fused_slice: None,
            }],
        };
        let x = to_bf16(&make_input(1, 4, in_dim));
        let a = lora.forward(&x).unwrap().to_dtype(DType::F32).unwrap();
        let b = merged.forward(&x).unwrap().to_dtype(DType::F32).unwrap();
        let max = max_abs_diff(
            &a.to_device(&candle_core::Device::Cpu).unwrap(),
            &b.to_device(&candle_core::Device::Cpu).unwrap(),
        );
        assert!(max < 1e-2, "bf16 bypass vs merged: {max}");
    }

    #[test]
    fn test_build_registry_double_block_qkv_into_fused_slice() {
        use crate::flux::lora::{LoraAdapter, LoraLayer, LoraSpec};
        use std::collections::HashMap as HM;
        // Synthesize a diffusers-shaped LoRA touching the double-block
        // image-attention Q projection. After build_registry the result
        // must land under the fused `img_attn.qkv.weight` key with a
        // FusedSlice covering rows [0, h) (Q is component 0).
        let h = 16; // pretend hidden_size for this test
        let device = Device::Cpu;
        let a = Tensor::zeros((4, h), DType::F32, &device).unwrap();
        let b = Tensor::zeros((h, 4), DType::F32, &device).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "transformer.transformer_blocks.0.attn.to_q".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 4 };
        let specs = [LoraSpec {
            adapter: &adapter,
            scale: 0.5,
            path_hash: 0xAB,
        }];
        let mut linear_out_dims = HM::new();
        linear_out_dims.insert("double_blocks.0.img_attn.qkv.weight".to_string(), 3 * h);
        let registry = build_registry(&specs, &linear_out_dims, &device, DType::F32).unwrap();
        let stack = registry.adapters_for("double_blocks.0.img_attn.qkv.weight");
        assert_eq!(stack.len(), 1, "registry must record the Q-only adapter");
        let slice = stack[0].fused_slice.expect("fused slice present");
        assert_eq!(slice.offset, 0, "Q is component 0 → row offset 0");
        assert_eq!(slice.length, h, "Q slice spans hidden_size rows");
        assert!((stack[0].scale - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_build_registry_single_block_mlp_lands_after_qkv() {
        use crate::flux::lora::{LoraAdapter, LoraLayer, LoraSpec};
        use std::collections::HashMap as HM;
        // single_blocks.linear1 fuses [Q, K, V, MLP]. Verify that an
        // MLP-targeting LoRA gets a FusedSlice that starts at 3*h and
        // has length = mlp dim (= LoRA's `b` dim 0). This is the case
        // most likely to silently miscompute via the equal-split path.
        let h = 16;
        let mlp = 64;
        let device = Device::Cpu;
        let a = Tensor::zeros((4, h), DType::F32, &device).unwrap();
        let b = Tensor::zeros((mlp, 4), DType::F32, &device).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "transformer.single_transformer_blocks.0.proj_mlp".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 4 };
        let specs = [LoraSpec {
            adapter: &adapter,
            scale: 1.0,
            path_hash: 0xCD,
        }];
        let mut linear_out_dims = HM::new();
        linear_out_dims.insert("single_blocks.0.linear1.weight".to_string(), 3 * h + mlp);
        let registry = build_registry(&specs, &linear_out_dims, &device, DType::F32).unwrap();
        let stack = registry.adapters_for("single_blocks.0.linear1.weight");
        assert_eq!(stack.len(), 1);
        let slice = stack[0].fused_slice.expect("fused slice present");
        assert_eq!(slice.offset, 3 * h, "MLP starts after Q,K,V");
        assert_eq!(slice.length, mlp, "MLP slice spans mlp dim");
    }

    #[test]
    fn test_build_registry_direct_target_no_fused_slice() {
        use crate::flux::lora::{LoraAdapter, LoraLayer, LoraSpec};
        use std::collections::HashMap as HM;
        // FF MLP `ff.net.0.proj` is a Direct target — adapter must
        // attach without a fused slice and the registry must still
        // expose it under the right candle key.
        let h = 16;
        let mlp = 64;
        let device = Device::Cpu;
        let a = Tensor::zeros((4, h), DType::F32, &device).unwrap();
        let b = Tensor::zeros((mlp, 4), DType::F32, &device).unwrap();
        let mut layers = HashMap::new();
        layers.insert(
            "transformer.transformer_blocks.0.ff.net.0.proj".to_string(),
            LoraLayer { a, b, alpha: None },
        );
        let adapter = LoraAdapter { layers, rank: 4 };
        let specs = [LoraSpec {
            adapter: &adapter,
            scale: 0.25,
            path_hash: 0xEF,
        }];
        let registry = build_registry(&specs, &HM::new(), &device, DType::F32).unwrap();
        let stack = registry.adapters_for("double_blocks.0.img_mlp.0.weight");
        assert_eq!(stack.len(), 1);
        assert!(stack[0].fused_slice.is_none(), "Direct target = no slice");
    }

    #[test]
    fn test_fused_slice_offset_zero_and_end() {
        // Edge cases for the slice-cat plumbing: offset=0 (nothing
        // before) and offset+length = total (nothing after). Both
        // branches of `add_into_slice` must work without an extra
        // empty-tensor concat.
        let inner = make_linear(9, 3, false);
        let (down, up) = make_lora_pair(3, 2, 3, 17.0);
        let scale = 0.5f32;

        // Slice [0, 3): "before" branch absent.
        let lora_front = LoraLinear::WithAdapters {
            inner: inner.clone(),
            adapters: vec![LinearLoraAdapter {
                down: down.clone(),
                up: up.clone(),
                scale,
                fused_slice: Some(FusedSlice {
                    offset: 0,
                    length: 3,
                }),
            }],
        };
        // Slice [6, 9): "after" branch absent.
        let lora_back = LoraLinear::WithAdapters {
            inner: inner.clone(),
            adapters: vec![LinearLoraAdapter {
                down,
                up,
                scale,
                fused_slice: Some(FusedSlice {
                    offset: 6,
                    length: 3,
                }),
            }],
        };
        let x = make_input(1, 2, 3);
        // Both must be finite and match the plain output outside the slice.
        let plain = inner.forward(&x).unwrap();
        let front = lora_front.forward(&x).unwrap();
        let back = lora_back.forward(&x).unwrap();
        let front_after = front.narrow(2, 3, 6).unwrap();
        let plain_after = plain.narrow(2, 3, 6).unwrap();
        assert!(max_abs_diff(&front_after, &plain_after) < 1e-7);
        let back_before = back.narrow(2, 0, 6).unwrap();
        let plain_before = plain.narrow(2, 0, 6).unwrap();
        assert!(max_abs_diff(&back_before, &plain_before) < 1e-7);
    }
}
