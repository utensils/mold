//! Custom `SimpleBackend` for Civitai single-file checkpoints.
//!
//! Translates each diffusers `vb.get(name)` call (issued by candle's
//! `stable_diffusion::{unet_2d::UNet2DConditionModel, vae::AutoEncoderKL,
//! clip::ClipTextTransformer}` constructors) into an mmap'd read of the
//! corresponding A1111 source tensor.
//!
//! Two projection rules:
//!
//! 1. **Direct** — 1:1 lookup; the diffusers key resolves to one A1111
//!    source tensor returned whole. Used for every UNet / VAE / CLIP-L
//!    tensor and most CLIP-G tensors.
//! 2. **Slice** — the diffusers key resolves to a row-wise slice of a
//!    fused source tensor. Used only by SDXL's CLIP-G OpenCLIP
//!    `attn.in_proj_{weight,bias}` slabs, which split into the diffusers
//!    `self_attn.{q,k,v}_proj.{weight,bias}` triple.
//!
//! Modeled on `crates/mold-inference/src/flux/lora.rs::LoraBackend` —
//! same `MmapedSafetensors` + `SimpleBackend` shape, but the rule table
//! is built from a fresh `Sd15Remap` / `SdxlRemap` instead of LoRA
//! patches, and slicing returns *just* the slice (versus LoRA's
//! "modify-and-return-whole").

use crate::flux2::Flux2Config;
use crate::loader::{RenameOutput, Sd15Remap, SdxlRemap};
use anyhow::{anyhow, Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Tensor};
use candle_nn::var_builder::SimpleBackend;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Verify the safetensors file at `path` is not truncated relative to its
/// declared header. Returns a clear, actionable error when the on-disk file
/// is shorter than the header demands.
///
/// Without this check, an interrupted Civitai download (e.g. cv:2739091:
/// 11.2 GB on disk vs. 18.16 GB declared) bubbles up from
/// `MmapedSafetensors::new` as an opaque safetensors `InvalidData` —
/// users see only the outer `with_context` wrapper and have no way to
/// know that a re-download is the fix.
///
/// Touches only the JSON header; tensor data is never read.
fn check_safetensors_not_truncated(path: &Path) -> Result<()> {
    let file_size = std::fs::metadata(path)
        .with_context(|| format!("stat {} for size check", path.display()))?
        .len();

    let mut f =
        File::open(path).with_context(|| format!("open {} for size check", path.display()))?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).with_context(|| {
        format!(
            "read safetensors header length at {} (file is only {} bytes — likely truncated)",
            path.display(),
            file_size,
        )
    })?;
    let header_len = u64::from_le_bytes(len_buf);

    let header_end = 8u64.saturating_add(header_len);
    if header_end > file_size {
        return Err(anyhow!(
            "checkpoint at {} is truncated: file is {} bytes but the safetensors header alone \
             needs {} bytes (8-byte length prefix + {} declared header length). \
             Re-download the model — the file is incomplete.",
            path.display(),
            file_size,
            header_end,
            header_len,
        ));
    }

    let mut header_buf = vec![0u8; header_len as usize];
    f.read_exact(&mut header_buf)
        .with_context(|| format!("read safetensors header at {}", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&header_buf)
        .with_context(|| format!("parse safetensors header JSON at {}", path.display()))?;
    let obj = header.as_object().ok_or_else(|| {
        anyhow!(
            "safetensors header at {} is not a JSON object",
            path.display(),
        )
    })?;

    let mut max_end: u64 = 0;
    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        if let Some(end) = v
            .get("data_offsets")
            .and_then(|x| x.as_array())
            .filter(|a| a.len() == 2)
            .and_then(|a| a[1].as_u64())
        {
            max_end = max_end.max(end);
        }
    }

    let expected_total = header_end.saturating_add(max_end);
    if expected_total > file_size {
        let missing = expected_total - file_size;
        return Err(anyhow!(
            "checkpoint at {} is truncated: file is {} bytes but the safetensors header declares \
             tensor data ending at {} bytes ({} bytes missing). \
             The download is incomplete — re-fetch the model.",
            path.display(),
            file_size,
            expected_total,
            missing,
        ));
    }

    Ok(())
}

/// One NVFP4 sub-component routed through a sub-key on the diffusers side.
///
/// Streaming dequant emits THREE sub-keys per NVFP4 layer
/// (`weight.nvfp4_packed`, `weight.nvfp4_block_scales`, `weight.nvfp4_tensor_scale`)
/// instead of fusing them into a single `weight` lookup. `Flux2Linear::load_with_bias`
/// detects NVFP4 by probing `vb.contains_tensor("weight.nvfp4_packed")` and
/// loads each component separately, deferring the FP4 → BF16 dequant to first
/// forward (and caching the BF16 weight on CPU so subsequent forwards only DMA
/// to GPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nvfp4Component {
    /// U8 `[N, K/2]` packed FP4 nibbles — read `{base}.weight`.
    Packed,
    /// F8E4M3 `[N, K/16]` per-block scales — read `{base}.weight_scale`.
    BlockScales,
    /// F32 scalar per-tensor scale — read `{base}.weight_scale_2`.
    TensorScale,
    /// U32 `[3]` slice metadata — synthesized at lookup time as
    /// `[axis, component, num_components]`. Only present for fused QKV slabs
    /// shared across `to_q`/`to_k`/`to_v`.
    SliceMeta {
        axis: u32,
        component: u32,
        num_components: u32,
    },
}

/// One projection rule per diffusers key.
#[derive(Debug, Clone)]
enum BackendEntry {
    /// 1:1 lookup — return the whole source tensor.
    Direct { source_key: String },
    /// Row-wise (or other-axis) slice of a fused source tensor.
    /// Components are equal-sized: `stride = base.dim(axis) / num_components`,
    /// `offset = component * stride`.
    Slice {
        source_key: String,
        axis: usize,
        component: usize,
        num_components: usize,
    },
    /// One NVFP4 sub-component lookup. `source_base` is the BFL key with
    /// `.weight` stripped (e.g.
    /// `model.diffusion_model.double_blocks.0.img_attn.proj`). The
    /// `component` selects which on-disk sidecar tensor to read (or, in the
    /// `SliceMeta` case, synthesizes a metadata tensor without touching the
    /// file). All three sub-component entries for one NVFP4 layer share the
    /// same `source_base`; the streaming `Flux2Linear` recombines them at
    /// load time and caches the BF16 dequant on first forward.
    Nvfp4Component {
        source_base: String,
        component: Nvfp4Component,
    },
    /// Load `source_key` and swap its two equal halves along `axis`.
    ///
    /// Used to convert BFL-native weight ordering to diffusers ordering.
    /// Specifically, BFL's `final_layer.adaLN_modulation.1.weight` stores
    /// `(shift, scale)` along the output dimension (dim 0), but diffusers'
    /// `AdaLayerNormContinuous` convention — and `LastLayer::forward` —
    /// expect `(scale, shift)`. Swapping at load time means the forward
    /// path never needs to know which checkpoint format was used.
    SwapHalves { source_key: String, axis: usize },
}

/// `SimpleBackend` over an mmap'd Civitai single-file checkpoint.
///
/// One per engine — built from the engine's `Sd15Remap` (SD1.5) or
/// `SdxlRemap` (SDXL). Fed to candle's `VarBuilder::from_backend(...)`,
/// then handed to `UNet2DConditionModel::new` / `AutoEncoderKL::new` /
/// `ClipTextTransformer::new` exactly like the diffusers-layout path.
pub struct SingleFileBackend {
    st: MmapedSafetensors,
    /// `diffusers_key → projection rule into the mmap'd source tensors`.
    entries: BTreeMap<String, BackendEntry>,
    /// Hand `F8_E4M3` weights back at their on-disk dtype instead of casting
    /// them to the VarBuilder's dtype. Set only for FP8-scaled Flux.2
    /// checkpoints, whose `Flux2Linear::new` selects its FP8 arm by inspecting
    /// `weight.dtype()` — a cast here would silently take the BF16 arm, which
    /// applies no `scale_weight` and renders noise. Every other consumer keeps
    /// the historical casting behaviour.
    preserve_fp8: bool,
}

impl SingleFileBackend {
    /// Mmap the checkpoint and wrap it in a backend with the given entries.
    /// Internal helper — used by every per-component factory.
    fn from_entries(checkpoint: &Path, entries: BTreeMap<String, BackendEntry>) -> Result<Self> {
        check_safetensors_not_truncated(checkpoint).with_context(|| {
            format!(
                "validate single-file checkpoint at {}",
                checkpoint.display(),
            )
        })?;
        let st = unsafe { MmapedSafetensors::new(checkpoint) }
            .with_context(|| format!("mmap single-file checkpoint at {}", checkpoint.display()))?;
        Ok(Self {
            st,
            entries,
            preserve_fp8: false,
        })
    }

    /// Cast a looked-up tensor to the VarBuilder's dtype, except for the FP8
    /// weights of an FP8-scaled checkpoint, which are handed back at their
    /// on-disk dtype so `Flux2Linear::new` can recognize and dequantize them
    /// with their `scale_weight`.
    fn cast(&self, t: Tensor, dtype: DType) -> candle_core::Result<Tensor> {
        if t.dtype() == dtype {
            return Ok(t);
        }
        if self.preserve_fp8 && matches!(t.dtype(), DType::F8E4M3) {
            return Ok(t);
        }
        t.to_dtype(dtype)
    }

    /// Direct-only entries from a `BTreeMap<diffusers, a1111>` slice — used by
    /// the UNet, VAE, and SD1.5 / SDXL CLIP-L factories.
    fn direct_entries(remap_slice: &BTreeMap<String, String>) -> BTreeMap<String, BackendEntry> {
        let mut entries: BTreeMap<String, BackendEntry> = BTreeMap::new();
        for (diffusers, a1111) in remap_slice {
            entries.insert(
                diffusers.clone(),
                BackendEntry::Direct {
                    source_key: a1111.clone(),
                },
            );
        }
        entries
    }

    /// SDXL CLIP-G entries (Direct or FusedSlice) from `remap.clip_g`.
    fn clip_g_entries(
        clip_g_remap: &BTreeMap<String, (String, RenameOutput)>,
    ) -> BTreeMap<String, BackendEntry> {
        let mut entries: BTreeMap<String, BackendEntry> = BTreeMap::new();
        for (diffusers, (a1111_key, output)) in clip_g_remap {
            let entry = match output {
                RenameOutput::Direct(_) => BackendEntry::Direct {
                    source_key: a1111_key.clone(),
                },
                RenameOutput::FusedSlice {
                    axis,
                    component,
                    num_components,
                    ..
                } => BackendEntry::Slice {
                    source_key: a1111_key.clone(),
                    axis: *axis,
                    component: *component,
                    num_components: *num_components,
                },
            };
            entries.insert(diffusers.clone(), entry);
        }
        entries
    }

    /// Construct from an SD1.5 remap. Every entry is `Direct` since
    /// SD1.5 has no fused QKV slabs (CLIP-L is HF layout, not OpenCLIP).
    ///
    /// Carries UNet + VAE + CLIP-L in one entries map. SD1.5's three
    /// component keyspaces are disjoint (UNet under `down_blocks/up_blocks/...`,
    /// VAE under `encoder/decoder/...`, CLIP-L under `text_model.X`) so
    /// no key collides.
    pub fn from_sd15_remap(checkpoint: &Path, remap: &Sd15Remap) -> Result<Self> {
        let mut entries: BTreeMap<String, BackendEntry> = BTreeMap::new();
        for (diffusers, a1111) in remap
            .unet
            .iter()
            .chain(remap.vae.iter())
            .chain(remap.clip_l.iter())
        {
            entries.insert(
                diffusers.clone(),
                BackendEntry::Direct {
                    source_key: a1111.clone(),
                },
            );
        }
        Self::from_entries(checkpoint, entries)
    }

    /// SD1.5 UNet-scoped backend — only `remap.unet` entries. Used by the
    /// per-component construction helpers in `mold-inference::sd15`.
    pub fn from_sd15_unet(checkpoint: &Path, remap: &Sd15Remap) -> Result<Self> {
        Self::from_entries(checkpoint, Self::direct_entries(&remap.unet))
    }

    /// SD1.5 VAE-scoped backend.
    pub fn from_sd15_vae(checkpoint: &Path, remap: &Sd15Remap) -> Result<Self> {
        Self::from_entries(checkpoint, Self::direct_entries(&remap.vae))
    }

    /// SD1.5 CLIP-L-scoped backend.
    pub fn from_sd15_clip_l(checkpoint: &Path, remap: &Sd15Remap) -> Result<Self> {
        Self::from_entries(checkpoint, Self::direct_entries(&remap.clip_l))
    }

    /// Construct from an SDXL remap. UNet / VAE / CLIP-L are `Direct`;
    /// CLIP-G threads `RenameOutput` through — `Direct(_)` becomes a
    /// `Direct` entry, `FusedSlice {axis, component, num_components, …}`
    /// becomes a `Slice` entry.
    ///
    /// **Collision-prone** when the consumer wraps the backend in a
    /// CLIP-L-or-CLIP-G `VarBuilder`: SDXL CLIP-L's renamed diffusers
    /// keys (`text_model.embeddings.token_embedding.weight`, every
    /// encoder layer's `self_attn.{q,k,v,out}_proj.weight`,
    /// `final_layer_norm.weight`, `position_embedding.weight`) collide
    /// with CLIP-G's renamed keys for the same diffusers paths. The
    /// `clip_g` insertion happens after `clip_l`, so CLIP-G overwrites
    /// CLIP-L on collision — when CLIP-L's `ClipTextTransformer` then
    /// requests `text_model.embeddings.token_embedding.weight`, it
    /// receives CLIP-G's `[vocab, 1280]` weight (instead of its own
    /// `[vocab, 768]`), and the next `Embedding::forward` reshape
    /// fails with `shape mismatch in reshape, lhs: [77, 1280],
    /// rhs: [1, 77, 768]`.
    ///
    /// Production code should prefer the per-component
    /// `from_sdxl_{unet,vae,clip_l,clip_g}` factories. This method is
    /// kept for the existing CLIP-G-only `sdxl_backend_slices_clip_g_*`
    /// tests (which assert the slice semantics in isolation) and for
    /// callers that explicitly want the all-in-one entries map.
    pub fn from_sdxl_remap(checkpoint: &Path, remap: &SdxlRemap) -> Result<Self> {
        let mut entries: BTreeMap<String, BackendEntry> = BTreeMap::new();
        for (diffusers, a1111) in remap
            .unet
            .iter()
            .chain(remap.vae.iter())
            .chain(remap.clip_l.iter())
        {
            entries.insert(
                diffusers.clone(),
                BackendEntry::Direct {
                    source_key: a1111.clone(),
                },
            );
        }
        for (diffusers, entry) in Self::clip_g_entries(&remap.clip_g) {
            entries.insert(diffusers, entry);
        }
        Self::from_entries(checkpoint, entries)
    }

    /// SDXL UNet-scoped backend — only `remap.unet` entries.
    pub fn from_sdxl_unet(checkpoint: &Path, remap: &SdxlRemap) -> Result<Self> {
        Self::from_entries(checkpoint, Self::direct_entries(&remap.unet))
    }

    /// SDXL VAE-scoped backend.
    pub fn from_sdxl_vae(checkpoint: &Path, remap: &SdxlRemap) -> Result<Self> {
        Self::from_entries(checkpoint, Self::direct_entries(&remap.vae))
    }

    /// SDXL CLIP-L-scoped backend — only `remap.clip_l` entries. Avoids
    /// collisions with CLIP-G that would otherwise materialise CLIP-L's
    /// `ClipTextTransformer` with CLIP-G's `[vocab, 1280]` weights when
    /// CLIP-L expects `[vocab, 768]`.
    pub fn from_sdxl_clip_l(checkpoint: &Path, remap: &SdxlRemap) -> Result<Self> {
        Self::from_entries(checkpoint, Self::direct_entries(&remap.clip_l))
    }

    /// SDXL CLIP-G-scoped backend — only `remap.clip_g` entries (Direct
    /// for layer norms / embeddings / projections / fc / out_proj, FusedSlice
    /// for the `attn.in_proj_*` slabs).
    pub fn from_sdxl_clip_g(checkpoint: &Path, remap: &SdxlRemap) -> Result<Self> {
        Self::from_entries(checkpoint, Self::clip_g_entries(&remap.clip_g))
    }

    /// Construct from a Civitai / ComfyUI single-file Flux.2 checkpoint
    /// (BFL-native naming, every key prefixed `model.diffusion_model.`).
    ///
    /// Builds the diffusers→BFL-native rename table from `cfg.depth` and
    /// `cfg.depth_single_blocks` so it transparently covers Klein-4B
    /// (depth=5, depth_single_blocks=20) and Klein-9B (depth=8,
    /// depth_single_blocks=24). The conditional `vector_in` / `guidance_in`
    /// entries are added only when their corresponding `Flux2Config` flags
    /// are set — Klein has neither.
    ///
    /// **Detection contract**: header-detected `Nvfp4` checkpoints route
    /// through synthetic NVFP4 subkeys (`weight.nvfp4_packed`,
    /// `weight.nvfp4_block_scales`, `weight.nvfp4_tensor_scale`). The
    /// consuming `Flux2Linear::Nvfp4Streaming` lazily dequantizes
    /// FP4 × FP8-block × F32-tensor-scale to a CPU BF16 cache at first
    /// forward. Non-BFL-native (root-level diffusers) layouts fail with
    /// a clear message.
    ///
    /// Fused-QKV slabs in `double_blocks.{i}.{img,txt}_attn.qkv.weight`
    /// are split into the diffusers `to_q/to_k/to_v` (and
    /// `add_q_proj/add_k_proj/add_v_proj`) triple via `BackendEntry::Slice`
    /// (or `Nvfp4Slice` in the NVFP4 case). Single blocks already use a
    /// single fused `linear1` on both sides — no slicing needed.
    pub fn from_flux2_singlefile(checkpoint: &Path, cfg: &Flux2Config) -> Result<Self> {
        let format = crate::flux2::detect_format(checkpoint).with_context(|| {
            format!("peek single-file Flux.2 header at {}", checkpoint.display(),)
        })?;
        let (prefix, quant) = match format {
            crate::flux2::Flux2SingleFileFormat::Nvfp4 => ("model.diffusion_model.", Quant::Nvfp4),
            crate::flux2::Flux2SingleFileFormat::BflNative => {
                ("model.diffusion_model.", Quant::None)
            }
            crate::flux2::Flux2SingleFileFormat::BflNativeRoot => ("", Quant::None),
            crate::flux2::Flux2SingleFileFormat::Diffusers
            | crate::flux2::Flux2SingleFileFormat::Unknown => {
                return Err(anyhow!(
                    "checkpoint does not look like a BFL-native single-file \
                     Flux.2 (no model.diffusion_model.* or root-level BFL keys \
                     found — expected Civitai/ComfyUI export)",
                ));
            }
        };

        // Per-tensor NVFP4 routing: in cv:2759597 (and similar Civitai
        // exports), the producer typically leaves input/output projections
        // and small MLPs as BF16 — so NOT every `*.weight` is NVFP4 even in
        // an NVFP4-flagged file. Header-peek once and build the set of
        // bases that actually have NVFP4 sidecars; routing falls back to
        // Direct for any base not in that set.
        let nvfp4_bases: std::collections::BTreeSet<String> = if quant == Quant::Nvfp4 {
            collect_nvfp4_bases(checkpoint).with_context(|| {
                format!(
                    "enumerate NVFP4 bases in {} (header peek)",
                    checkpoint.display(),
                )
            })?
        } else {
            std::collections::BTreeSet::new()
        };

        let rms_suffix = detect_rms_norm_suffix(checkpoint, prefix).with_context(|| {
            format!(
                "probe RMSNorm tensor suffix in {} (header peek)",
                checkpoint.display(),
            )
        })?;

        // FP8-scaled layers are orthogonal to NVFP4 and just as mixed: BFL's
        // klein FP8 quantizes every attention and MLP slab, Comfy-Org's dev
        // `fp8mixed` quantizes only the MLPs. Collect them per tensor so the
        // BF16 half of a mixed file keeps its ordinary Direct route.
        let fp8_scaled = if quant == Quant::Nvfp4 {
            BTreeMap::new()
        } else {
            collect_fp8_scaled_bases(checkpoint).with_context(|| {
                format!(
                    "enumerate FP8-scaled bases in {} (header peek)",
                    checkpoint.display(),
                )
            })?
        };

        let entries =
            build_flux2_entries(cfg, prefix, quant, &nvfp4_bases, rms_suffix, &fp8_scaled);
        let mut backend = Self::from_entries(checkpoint, entries)?;
        backend.preserve_fp8 = !fp8_scaled.is_empty();
        Ok(backend)
    }

    /// Resolve `diffusers_key` to a tensor on `dev` per the projection rule.
    /// Direct entries return the whole source tensor; Slice entries narrow
    /// along `axis` to one of `num_components` equal-sized chunks.
    fn lookup(&self, diffusers_key: &str, dev: &Device) -> candle_core::Result<Tensor> {
        let entry = self.entries.get(diffusers_key).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "single-file backend: no rename rule for diffusers key '{diffusers_key}'"
            ))
        })?;

        match entry {
            BackendEntry::Direct { source_key } => self.st.load(source_key, dev),
            BackendEntry::Slice {
                source_key,
                axis,
                component,
                num_components,
            } => {
                let full = self.st.load(source_key, dev)?;
                let total = full.dim(*axis)?;
                if *num_components == 0 || total % num_components != 0 {
                    return Err(candle_core::Error::Msg(format!(
                        "single-file backend: source tensor '{source_key}' axis {axis} dim {total} is not divisible by num_components {num_components}",
                    )));
                }
                let stride = total / num_components;
                let offset = component * stride;
                full.narrow(*axis, offset, stride)
            }
            BackendEntry::Nvfp4Component {
                source_base,
                component,
            } => self.load_nvfp4_component(source_base, *component, dev),
            BackendEntry::SwapHalves { source_key, axis } => {
                let t = self.st.load(source_key, dev)?;
                let total = t.dim(*axis)?;
                if total % 2 != 0 {
                    return Err(candle_core::Error::Msg(format!(
                        "single-file backend: SwapHalves source '{source_key}' axis {axis} dim {total} is odd",
                    )));
                }
                let half = total / 2;
                let first = t.narrow(*axis, 0, half)?;
                let second = t.narrow(*axis, half, half)?;
                Tensor::cat(&[&second, &first], *axis)
            }
        }
    }

    /// Read one NVFP4 sub-component for `source_base` from the mmap. NVFP4
    /// sidecars must always live on CPU regardless of the requested device —
    /// the streaming `Flux2Linear` defers the actual BF16 dequant to first
    /// forward and caches the result on CPU. Returning the packed tensors
    /// straight to GPU would (a) blow a 9 GB / 24 GB GPU budget on a Klein-9B
    /// load and (b) waste a DMA round-trip because the dequant must run on CPU.
    fn load_nvfp4_component(
        &self,
        source_base: &str,
        component: Nvfp4Component,
        _requested_dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let cpu = Device::Cpu;
        match component {
            Nvfp4Component::Packed => {
                let weight_key = format!("{source_base}.weight");
                let t = self.st.load(&weight_key, &cpu)?;
                if t.dtype() != DType::U8 {
                    return Err(candle_core::Error::Msg(format!(
                        "NVFP4: expected '{weight_key}' to be U8 packed FP4, got {:?}",
                        t.dtype()
                    )));
                }
                Ok(t)
            }
            Nvfp4Component::BlockScales => {
                let scale_key = format!("{source_base}.weight_scale");
                let t = self.st.load(&scale_key, &cpu)?;
                if t.dtype() != DType::F8E4M3 {
                    return Err(candle_core::Error::Msg(format!(
                        "NVFP4: expected '{scale_key}' to be F8E4M3 block scales, got {:?}",
                        t.dtype()
                    )));
                }
                Ok(t)
            }
            Nvfp4Component::TensorScale => {
                let scale2_key = format!("{source_base}.weight_scale_2");
                let t = self.st.load(&scale2_key, &cpu)?;
                t.to_dtype(DType::F32)
            }
            Nvfp4Component::SliceMeta {
                axis,
                component,
                num_components,
            } => Tensor::from_vec(vec![axis, component, num_components], 3, &cpu),
        }
    }
}

// ---------------------------------------------------------------------------
// Flux.2 BFL-native → diffusers rename table.
// ---------------------------------------------------------------------------

/// Quantization of a Flux.2 single-file checkpoint. Drives whether the
/// rename table emits plain `Direct`/`Slice` entries (BF16/FP8-native) or
/// `Nvfp4`/`Nvfp4Slice` entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quant {
    None,
    Nvfp4,
}

/// Strip a trailing `.weight` from a BFL key to get the source-base used by
/// `Nvfp4` / `Nvfp4Slice`.
fn weight_base(bfl_full_key: &str) -> &str {
    bfl_full_key.strip_suffix(".weight").unwrap_or(bfl_full_key)
}

/// Header-peek the safetensors at `path` and return the set of bases (BFL
/// keys minus the `.weight` suffix) that have NVFP4 sidecars
/// (`{base}.weight_scale` AND `{base}.weight_scale_2`). Anything not in this
/// set is routed Direct even in NVFP4-flagged checkpoints — necessary because
/// Civitai/ComfyUI exporters typically leave input/output projections and
/// small MLPs as BF16 (cv:2759597 has 11 such BF16 weights mixed in with the
/// 110 NVFP4 layers).
fn collect_nvfp4_bases(path: &Path) -> Result<std::collections::BTreeSet<String>> {
    use std::collections::BTreeSet;
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: serde_json::Value =
        serde_json::from_slice(&header_buf).with_context(|| "parse safetensors header JSON")?;
    let obj = header
        .as_object()
        .ok_or_else(|| anyhow!("safetensors header is not a JSON object"))?;

    let mut has_scale: BTreeSet<String> = BTreeSet::new();
    let mut has_scale_2: BTreeSet<String> = BTreeSet::new();
    for key in obj.keys() {
        if key == "__metadata__" {
            continue;
        }
        if let Some(base) = key.strip_suffix(".weight_scale") {
            has_scale.insert(base.to_string());
        } else if let Some(base) = key.strip_suffix(".weight_scale_2") {
            has_scale_2.insert(base.to_string());
        }
    }
    // Require both — a stray `weight_scale` without `weight_scale_2` would
    // indicate a malformed checkpoint or a different quantization scheme.
    Ok(has_scale.intersection(&has_scale_2).cloned().collect())
}

/// Two spellings exist in the wild for one FP8 layer's dequantization scale.
/// BFL's own FP8
/// conversions (`FLUX.2-klein-{4b,9b}-fp8`) and Comfy-Org's `fp8mixed` write
/// `<base>.weight_scale` (plus an `<base>.input_scale` that only matters to a
/// tensor-core FP8 matmul, which mold does not run — dequantizing the weight
/// makes it irrelevant); ComfyUI's own re-exports write `<base>.scale_weight`.
/// Dropping the scale is not a rounding difference: `quantize` divides by it
/// (`comfy/quant_ops.py:138`), so ignoring a scale of ~2e-3 inflates every
/// weight by ~500x and the render is noise.
///
/// Header-peek `path` and collect every base whose `.weight` is stored as
/// `F8_E4M3` AND carries a per-tensor scale sidecar. Mixed files are the norm
/// (Comfy-Org's dev `fp8mixed` leaves all attention projections in BF16), so
/// this is a per-tensor decision, exactly like `collect_nvfp4_bases`.
fn collect_fp8_scaled_bases(path: &Path) -> Result<BTreeMap<String, String>> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: serde_json::Value =
        serde_json::from_slice(&header_buf).with_context(|| "parse safetensors header JSON")?;
    let obj = header
        .as_object()
        .ok_or_else(|| anyhow!("safetensors header is not a JSON object"))?;

    let is_fp8 = |key: &str| {
        obj.get(key)
            .and_then(|info| info.get("dtype"))
            .and_then(|dtype| dtype.as_str())
            .is_some_and(|dtype| dtype == "F8_E4M3" || dtype == "F8_E5M2")
    };

    let mut scaled: BTreeMap<String, String> = BTreeMap::new();
    for key in obj.keys() {
        if key == "__metadata__" {
            continue;
        }
        let Some(base) = key
            .strip_suffix(".weight_scale")
            .or_else(|| key.strip_suffix(".scale_weight"))
        else {
            continue;
        };
        // `weight_scale` is also NVFP4's block-scale key; there it is F8E4M3
        // with a shape, and `collect_nvfp4_bases` claims it. Only a base whose
        // own `.weight` is FP8 is an FP8-scaled linear.
        if is_fp8(&format!("{base}.weight")) {
            scaled.insert(base.to_string(), key.clone());
        }
    }
    Ok(scaled)
}

/// Header-peek the safetensors at `path` and decide which suffix the BFL
/// `*.norm.{query,key}_norm.*` RMSNorm tensors use.
///
/// Canonical BFL exports (and Civitai NVFP4 wraps such as cv:2759597) use
/// `.scale`; some community BF16 fine-tunes (e.g. cv:2765147 prototype
/// Klein-9B) ship the same tensors as `.weight`. We probe the first
/// double-block's `img_attn.norm.query_norm.*` to decide. If neither is
/// present we return `"scale"` so the canonical missing-key error surfaces
/// downstream instead of a synthesized one.
fn detect_rms_norm_suffix(path: &Path, prefix: &str) -> Result<&'static str> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header: serde_json::Value =
        serde_json::from_slice(&header_buf).with_context(|| "parse safetensors header JSON")?;
    let obj = header
        .as_object()
        .ok_or_else(|| anyhow!("safetensors header is not a JSON object"))?;

    let probe_scale = format!("{prefix}double_blocks.0.img_attn.norm.query_norm.scale");
    let probe_weight = format!("{prefix}double_blocks.0.img_attn.norm.query_norm.weight");
    if obj.contains_key(&probe_scale) {
        Ok("scale")
    } else if obj.contains_key(&probe_weight) {
        Ok("weight")
    } else {
        Ok("scale")
    }
}

/// Emit the three NVFP4 sub-key entries for a `diffusers_key.weight` →
/// `source_base` (`.nvfp4_packed`, `.nvfp4_block_scales`, `.nvfp4_tensor_scale`).
/// All three point at the same `source_base`, varying only in
/// `Nvfp4Component`.
fn nvfp4_subkeys(diffusers_key: &str, source_base: &str) -> Vec<(String, BackendEntry)> {
    debug_assert!(
        diffusers_key.ends_with(".weight"),
        "NVFP4 routing only applies to `.weight` keys"
    );
    vec![
        (
            format!("{diffusers_key}.nvfp4_packed"),
            BackendEntry::Nvfp4Component {
                source_base: source_base.to_string(),
                component: Nvfp4Component::Packed,
            },
        ),
        (
            format!("{diffusers_key}.nvfp4_block_scales"),
            BackendEntry::Nvfp4Component {
                source_base: source_base.to_string(),
                component: Nvfp4Component::BlockScales,
            },
        ),
        (
            format!("{diffusers_key}.nvfp4_tensor_scale"),
            BackendEntry::Nvfp4Component {
                source_base: source_base.to_string(),
                component: Nvfp4Component::TensorScale,
            },
        ),
    ]
}

/// The subkey `Flux2Linear::new` probes for an FP8 layer's per-tensor scale.
/// Both on-disk spellings are normalized to this one name so the transformer
/// stays unaware of which exporter wrote the checkpoint.
const FP8_SCALE_SUBKEY: &str = "scale_weight";

/// Register the FP8 scale sidecar for `diffusers`, when its source base has
/// one. A `.weight` target becomes `<layer>.scale_weight`; the scale is
/// per-tensor, so the three QKV slices of one fused slab all point at the
/// same scalar.
fn fp8_scale_entry(
    diffusers: &str,
    source_base: &str,
    fp8_scaled: &BTreeMap<String, String>,
) -> Option<(String, BackendEntry)> {
    let scale_key = fp8_scaled.get(source_base)?;
    let layer = diffusers.strip_suffix(".weight")?;
    Some((
        format!("{layer}.{FP8_SCALE_SUBKEY}"),
        BackendEntry::Direct {
            source_key: scale_key.clone(),
        },
    ))
}

fn direct(
    diffusers: &str,
    bfl_suffix: &str,
    prefix: &str,
    quant: Quant,
    nvfp4_bases: &std::collections::BTreeSet<String>,
    fp8_scaled: &BTreeMap<String, String>,
) -> Vec<(String, BackendEntry)> {
    let source_key = format!("{prefix}{bfl_suffix}");
    let source_base = weight_base(&source_key).to_string();
    // NVFP4 routing requires three things: (a) NVFP4 mode globally, (b) the
    // BFL target ends in `.weight` (so RMSNorm `query_norm.scale` keys stay
    // Direct), AND (c) the actual source base has NVFP4 sidecars in this
    // file. Civitai exports often leave input/output projections as BF16;
    // per-tensor decision is the only correct routing.
    let route_nvfp4 = quant == Quant::Nvfp4
        && bfl_suffix.ends_with(".weight")
        && nvfp4_bases.contains(&source_base);
    if !route_nvfp4 {
        let mut entries = vec![(diffusers.to_string(), BackendEntry::Direct { source_key })];
        entries.extend(fp8_scale_entry(diffusers, &source_base, fp8_scaled));
        return entries;
    }
    nvfp4_subkeys(diffusers, &source_base)
}

fn slice_qkv(
    diffusers: &str,
    bfl_suffix: &str,
    component: usize,
    prefix: &str,
    quant: Quant,
    nvfp4_bases: &std::collections::BTreeSet<String>,
    fp8_scaled: &BTreeMap<String, String>,
) -> Vec<(String, BackendEntry)> {
    let source_key = format!("{prefix}{bfl_suffix}");
    let source_base = weight_base(&source_key).to_string();
    let route_nvfp4 = quant == Quant::Nvfp4 && nvfp4_bases.contains(&source_base);
    if !route_nvfp4 {
        let mut entries = vec![(
            diffusers.to_string(),
            BackendEntry::Slice {
                source_key,
                axis: 0,
                component,
                num_components: 3,
            },
        )];
        entries.extend(fp8_scale_entry(diffusers, &source_base, fp8_scaled));
        return entries;
    }
    // NVFP4 sliced QKV — three sub-keys + a slice-meta sub-key. All four
    // share the same `source_base`; the streaming `Flux2Linear` reads the
    // meta tensor to know how to narrow the dequanted weight at forward
    // time. The packed/block_scales/tensor_scale sub-keys are identical
    // across `to_q`/`to_k`/`to_v` for the same `source_base`, so the BF16
    // cache populated by the first slice is shared with the other two.
    let mut entries = nvfp4_subkeys(diffusers, &source_base);
    entries.push((
        format!("{diffusers}.nvfp4_slice_meta"),
        BackendEntry::Nvfp4Component {
            source_base,
            component: Nvfp4Component::SliceMeta {
                axis: 0,
                component: component as u32,
                num_components: 3,
            },
        },
    ));
    entries
}

/// Build the diffusers-key → BFL-native projection table from a `Flux2Config`.
///
/// `prefix` is either `"model.diffusion_model."` (Civitai/ComfyUI exports
/// that wrap the transformer in a namespace) or `""` (community FP8
/// conversions that keep BFL keys at the root). `quant` selects whether to
/// emit plain Direct/Slice entries (BF16/FP8-native) or NVFP4 entries that
/// dequantise on lookup.
fn build_flux2_entries(
    cfg: &Flux2Config,
    prefix: &str,
    quant: Quant,
    nvfp4_bases: &std::collections::BTreeSet<String>,
    rms_suffix: &str,
    fp8_scaled: &BTreeMap<String, String>,
) -> BTreeMap<String, BackendEntry> {
    let mut e: BTreeMap<String, BackendEntry> = BTreeMap::new();

    // --- Top-level direct mappings ---
    let top_level = [
        ("x_embedder.weight", "img_in.weight"),
        ("context_embedder.weight", "txt_in.weight"),
        (
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            "time_in.in_layer.weight",
        ),
        (
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            "time_in.out_layer.weight",
        ),
        ("proj_out.weight", "final_layer.linear.weight"),
        (
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_img.lin.weight",
        ),
        (
            "double_stream_modulation_txt.linear.weight",
            "double_stream_modulation_txt.lin.weight",
        ),
        (
            "single_stream_modulation.linear.weight",
            "single_stream_modulation.lin.weight",
        ),
    ];
    for (d, b) in top_level {
        for (k, v) in direct(d, b, prefix, quant, nvfp4_bases, fp8_scaled) {
            e.insert(k, v);
        }
    }

    // BFL-native checkpoints store `final_layer.adaLN_modulation.1.weight`
    // with (shift, scale) row ordering, while diffusers' AdaLayerNormContinuous
    // convention — and `LastLayer::forward` — expects (scale, shift).
    // The diffusers model converter applies `swap_scale_shift` before saving
    // diffusers checkpoints, so the swap is already baked in for HF hub files.
    // For BFL-native single-file exports (this code path), we apply the swap
    // at load time so `LastLayer::forward` always receives (scale, shift).
    let ada_ln_bfl_key = format!("{prefix}final_layer.adaLN_modulation.1.weight");
    e.insert(
        "norm_out.linear.weight".to_string(),
        BackendEntry::SwapHalves {
            source_key: ada_ln_bfl_key,
            axis: 0,
        },
    );

    // --- Conditional: pooled-vector embedder (disabled in Klein). ---
    if cfg.vec_in_dim > 0 {
        for (d, b) in [
            ("vector_in.linear_1.weight", "vector_in.in_layer.weight"),
            ("vector_in.linear_2.weight", "vector_in.out_layer.weight"),
        ] {
            for (k, v) in direct(d, b, prefix, quant, nvfp4_bases, fp8_scaled) {
                e.insert(k, v);
            }
        }
    }

    // --- Conditional: guidance embedder (disabled in distilled Klein). ---
    if cfg.guidance_embed {
        for (d, b) in [
            (
                "time_guidance_embed.guidance_embedder.linear_1.weight",
                "guidance_in.in_layer.weight",
            ),
            (
                "time_guidance_embed.guidance_embedder.linear_2.weight",
                "guidance_in.out_layer.weight",
            ),
        ] {
            for (k, v) in direct(d, b, prefix, quant, nvfp4_bases, fp8_scaled) {
                e.insert(k, v);
            }
        }
    }

    // --- Per double-block ---
    for i in 0..cfg.depth {
        // Image-side fused QKV → split Q/K/V.
        for (component, comp_name) in [(0usize, "to_q"), (1, "to_k"), (2, "to_v")] {
            for (k, v) in slice_qkv(
                &format!("transformer_blocks.{i}.attn.{comp_name}.weight"),
                &format!("double_blocks.{i}.img_attn.qkv.weight"),
                component,
                prefix,
                quant,
                nvfp4_bases,
                fp8_scaled,
            ) {
                e.insert(k, v);
            }
        }
        // Image-side direct. RMSNorm rows use `rms_suffix` because canonical
        // BFL exports name them `.scale` while some community BF16 fine-tunes
        // (cv:2765147 etc.) ship the same tensors as `.weight`.
        let img_direct: [(&str, String); 5] = [
            ("attn.to_out.0.weight", "img_attn.proj.weight".to_string()),
            (
                "attn.norm_q.weight",
                format!("img_attn.norm.query_norm.{rms_suffix}"),
            ),
            (
                "attn.norm_k.weight",
                format!("img_attn.norm.key_norm.{rms_suffix}"),
            ),
            ("ff.linear_in.weight", "img_mlp.0.weight".to_string()),
            ("ff.linear_out.weight", "img_mlp.2.weight".to_string()),
        ];
        for (d_suffix, b_suffix) in &img_direct {
            for (k, v) in direct(
                &format!("transformer_blocks.{i}.{d_suffix}"),
                &format!("double_blocks.{i}.{b_suffix}"),
                prefix,
                quant,
                nvfp4_bases,
                fp8_scaled,
            ) {
                e.insert(k, v);
            }
        }

        // Text-side fused QKV → split add_q/k/v_proj.
        for (component, comp_name) in [(0usize, "add_q_proj"), (1, "add_k_proj"), (2, "add_v_proj")]
        {
            for (k, v) in slice_qkv(
                &format!("transformer_blocks.{i}.attn.{comp_name}.weight"),
                &format!("double_blocks.{i}.txt_attn.qkv.weight"),
                component,
                prefix,
                quant,
                nvfp4_bases,
                fp8_scaled,
            ) {
                e.insert(k, v);
            }
        }
        // Text-side direct. See note above re: `rms_suffix`.
        let txt_direct: [(&str, String); 5] = [
            ("attn.to_add_out.weight", "txt_attn.proj.weight".to_string()),
            (
                "attn.norm_added_q.weight",
                format!("txt_attn.norm.query_norm.{rms_suffix}"),
            ),
            (
                "attn.norm_added_k.weight",
                format!("txt_attn.norm.key_norm.{rms_suffix}"),
            ),
            (
                "ff_context.linear_in.weight",
                "txt_mlp.0.weight".to_string(),
            ),
            (
                "ff_context.linear_out.weight",
                "txt_mlp.2.weight".to_string(),
            ),
        ];
        for (d_suffix, b_suffix) in &txt_direct {
            for (k, v) in direct(
                &format!("transformer_blocks.{i}.{d_suffix}"),
                &format!("double_blocks.{i}.{b_suffix}"),
                prefix,
                quant,
                nvfp4_bases,
                fp8_scaled,
            ) {
                e.insert(k, v);
            }
        }
    }

    // --- Per single-block --- (RMSNorm suffix as in double-blocks above).
    for i in 0..cfg.depth_single_blocks {
        let single_direct: [(&str, String); 4] = [
            ("attn.to_qkv_mlp_proj.weight", "linear1.weight".to_string()),
            ("attn.to_out.weight", "linear2.weight".to_string()),
            (
                "attn.norm_q.weight",
                format!("norm.query_norm.{rms_suffix}"),
            ),
            ("attn.norm_k.weight", format!("norm.key_norm.{rms_suffix}")),
        ];
        for (d_suffix, b_suffix) in &single_direct {
            for (k, v) in direct(
                &format!("single_transformer_blocks.{i}.{d_suffix}"),
                &format!("single_blocks.{i}.{b_suffix}"),
                prefix,
                quant,
                nvfp4_bases,
                fp8_scaled,
            ) {
                e.insert(k, v);
            }
        }
    }

    e
}

impl SimpleBackend for SingleFileBackend {
    /// Shape-checked lookup. Mirrors the `SafeTensorWithRouting` /
    /// `MmapedSafetensors` / `HashMap<String, Tensor>` SimpleBackend impls
    /// (which all return `UnexpectedShape` on mismatch) so candle constructors
    /// that probe for alternative tensor layouts via `Ok / Err` keep working.
    ///
    /// The motivating case is `stable_diffusion::attention::get_qkv_linear`
    /// (used by VAE mid-block attention): it probes for the HF Linear shape
    /// `(channels, channels)` first, and on `Err` falls back to the A1111
    /// Conv2d 1×1 shape `(channels, channels, 1, 1)` plus a reshape to
    /// `(channels, channels)`. Without the shape check the rank-2 probe
    /// silently succeeds with a rank-4 tensor, the resulting `Linear` is
    /// constructed with a 4D weight, and the next forward call blows up
    /// with `shape mismatch in matmul, lhs: [1, 9216, 512],
    /// rhs: [1, 512, 512, 1, 1]`.
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let t = self.lookup(name, dev)?;
        if t.shape() != &s {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("single-file backend: shape mismatch for {name}"),
                expected: s,
                got: t.shape().clone(),
            }
            .bt());
        }
        self.cast(t, dtype)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let t = self.lookup(name, dev)?;
        self.cast(t, dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::single_file::load as load_bundle;
    use crate::loader::{build_sd15_remap, build_sdxl_remap};
    use mold_catalog::families::Family;
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// Build a synthetic safetensors with caller-supplied (key, shape, F32 data) tensors.
    /// Like `write_synthetic` but accepts owned `String` keys (for dynamically
    /// constructed key names).
    fn write_synthetic_with_tensors(
        name: &str,
        tensors: &[(String, Vec<usize>, Vec<f32>)],
    ) -> PathBuf {
        let refs: Vec<(&str, Vec<usize>, Vec<f32>)> = tensors
            .iter()
            .map(|(k, s, d)| (k.as_str(), s.clone(), d.clone()))
            .collect();
        write_synthetic(name, &refs)
    }

    fn write_synthetic(name: &str, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mold-sf-backend-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        let buffers: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, _, data)| {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for v in data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            })
            .collect();

        let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
        for ((key, shape, _), buf) in tensors.iter().zip(buffers.iter()) {
            views.insert(
                (*key).to_string(),
                TensorView::new(SafeDtype::F32, shape.clone(), buf).unwrap(),
            );
        }
        serialize_to_file(&views, &None, &path).unwrap();
        path
    }

    #[test]
    fn sd15_backend_resolves_diffusers_key_to_a1111_tensor() {
        // Build a tiny SD1.5 single-file with one CLIP-L weight whose
        // A1111 key is `cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight`.
        // The rename pass produces the diffusers key
        // `text_model.encoder.layers.0.self_attn.q_proj.weight`.
        // Verify backend.get(diffusers_key) returns the original bytes.
        let path = write_synthetic(
            "sd15-direct",
            &[(
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                vec![2, 2],
                vec![1.5, 2.5, 3.5, 4.5],
            )],
        );
        let bundle = load_bundle(&path, Family::Sd15).expect("partition sd15");
        let remap = build_sd15_remap(&bundle).expect("build remap");

        let backend = SingleFileBackend::from_sd15_remap(&path, &remap).expect("backend");
        let dev = Device::Cpu;
        let t = SimpleBackend::get_unchecked(
            &backend,
            "text_model.encoder.layers.0.self_attn.q_proj.weight",
            DType::F32,
            &dev,
        )
        .expect("direct lookup must hit");

        assert_eq!(t.dims(), &[2, 2]);
        let flat: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, vec![1.5, 2.5, 3.5, 4.5]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sdxl_backend_slices_clip_g_fused_qkv_weight() {
        // CLIP-G fused QKV weight: shape [3*d, d]. Fill rows 0..d with
        // 1.0 (Q), d..2d with 2.0 (K), 2d..3d with 3.0 (V) so the slice
        // boundaries are unambiguous. Verify each diffusers key returns
        // exactly one component, with the correct sentinel values.
        let d: usize = 4;
        let mut data = Vec::with_capacity(3 * d * d);
        for component in 1..=3 {
            for _row in 0..d {
                for _col in 0..d {
                    data.push(component as f32);
                }
            }
        }

        let path = write_synthetic(
            "sdxl-fused-qkv-w",
            &[
                // CLIP-L key — required so build_sdxl_remap doesn't error
                // on an empty CLIP-L bucket.
                (
                    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                    vec![2, 2],
                    vec![0.1, 0.2, 0.3, 0.4],
                ),
                (
                    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
                    vec![3 * d, d],
                    data,
                ),
            ],
        );
        let bundle = load_bundle(&path, Family::Sdxl).expect("partition sdxl");
        let remap = build_sdxl_remap(&bundle).expect("build remap");

        let backend = SingleFileBackend::from_sdxl_remap(&path, &remap).expect("backend");
        let dev = Device::Cpu;

        for (component, expected_value) in [(0usize, 1.0f32), (1, 2.0), (2, 3.0)] {
            let diffusers_key = match component {
                0 => "text_model.encoder.layers.0.self_attn.q_proj.weight",
                1 => "text_model.encoder.layers.0.self_attn.k_proj.weight",
                2 => "text_model.encoder.layers.0.self_attn.v_proj.weight",
                _ => unreachable!(),
            };

            let t = SimpleBackend::get_unchecked(&backend, diffusers_key, DType::F32, &dev)
                .unwrap_or_else(|e| panic!("slice lookup for component {component}: {e}"));
            assert_eq!(
                t.dims(),
                &[d, d],
                "{diffusers_key}: slice must be [d, d], not full [3*d, d]",
            );
            let flat: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                flat.iter().all(|&v| v == expected_value),
                "{diffusers_key}: every value must be {expected_value} (got {flat:?})",
            );
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sdxl_backend_slices_clip_g_fused_qkv_bias() {
        // 1D bias version of the QKV split: shape [3*d].
        let d: usize = 5;
        let mut data: Vec<f32> = Vec::with_capacity(3 * d);
        for component in 1..=3 {
            for _ in 0..d {
                data.push(component as f32 * 10.0);
            }
        }

        let path = write_synthetic(
            "sdxl-fused-qkv-b",
            &[
                (
                    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                    vec![2, 2],
                    vec![0.1, 0.2, 0.3, 0.4],
                ),
                (
                    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_bias",
                    vec![3 * d],
                    data,
                ),
            ],
        );
        let bundle = load_bundle(&path, Family::Sdxl).expect("partition sdxl");
        let remap = build_sdxl_remap(&bundle).expect("build remap");

        let backend = SingleFileBackend::from_sdxl_remap(&path, &remap).expect("backend");
        let dev = Device::Cpu;

        for (component, expected_value) in [(0usize, 10.0f32), (1, 20.0), (2, 30.0)] {
            let diffusers_key = match component {
                0 => "text_model.encoder.layers.0.self_attn.q_proj.bias",
                1 => "text_model.encoder.layers.0.self_attn.k_proj.bias",
                2 => "text_model.encoder.layers.0.self_attn.v_proj.bias",
                _ => unreachable!(),
            };

            let t = SimpleBackend::get_unchecked(&backend, diffusers_key, DType::F32, &dev)
                .unwrap_or_else(|e| panic!("bias slice for component {component}: {e}"));
            assert_eq!(
                t.dims(),
                &[d],
                "{diffusers_key}: 1D bias slice must be [d], not [3*d]",
            );
            let flat: Vec<f32> = t.to_vec1().unwrap();
            assert!(
                flat.iter().all(|&v| v == expected_value),
                "{diffusers_key}: every value must be {expected_value} (got {flat:?})",
            );
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn backend_unmapped_key_returns_error() {
        // Defensive: every candle-issued diffusers key must resolve, but
        // accidentally requesting a key the remap never registered must
        // surface as a legible error rather than a silent zero-fill.
        let path = write_synthetic(
            "sd15-empty-lookup",
            &[(
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                vec![1],
                vec![0.0],
            )],
        );
        let bundle = load_bundle(&path, Family::Sd15).expect("partition sd15");
        let remap = build_sd15_remap(&bundle).expect("build remap");

        let backend = SingleFileBackend::from_sd15_remap(&path, &remap).expect("backend");
        let dev = Device::Cpu;

        let err = SimpleBackend::get_unchecked(
            &backend,
            "totally.bogus.key.no.diffusers.path",
            DType::F32,
            &dev,
        )
        .expect_err("unmapped key must error");

        assert!(
            err.to_string().contains("no rename rule"),
            "expected legible error mentioning 'no rename rule', got: {err}",
        );

        let _ = std::fs::remove_file(path);
    }

    /// Bug 2 from `tasks/catalog-run-bridge-option-c-handoff.md`: when both
    /// CLIP-L and CLIP-G keys are present in the same `SdxlRemap`, the all-in-one
    /// `from_sdxl_remap` collapses them under one BTreeMap. CLIP-L and CLIP-G
    /// produce identical diffusers keys (`text_model.embeddings.token_embedding.weight`,
    /// every encoder layer's `self_attn.{q,k,v}_proj.weight`, …); CLIP-G's
    /// insertion overwrites CLIP-L's because clip_g is inserted second.
    ///
    /// On a real Juggernaut XL Ragnarok pull, that means CLIP-L's
    /// `ClipTextTransformer` materialises with CLIP-G's `[vocab, 1280]`
    /// `token_embedding`, and the next `Embedding::forward` reshape blows
    /// up with `shape mismatch in reshape, lhs: [77, 1280], rhs: [1, 77, 768]`
    /// the moment we run encode_prompt.
    ///
    /// Scoped factories (`from_sdxl_clip_l` / `from_sdxl_clip_g`) avoid the
    /// collision by including only one component's entries per backend —
    /// each `ClipTextTransformer::new` then sees only the keys it actually
    /// owns. Production code in `crates/mold-inference/src/sdxl/pipeline.rs`
    /// uses the scoped factories; this test locks the contract on the
    /// underlying backend.
    #[test]
    fn sdxl_clip_l_scoped_backend_returns_clip_l_tensor_when_keys_collide_with_clip_g() {
        // d_l = CLIP-L hidden dim (toy stand-in for 768).
        // d_g = CLIP-G hidden dim (toy stand-in for 1280) — different from d_l
        // so a wrong-component lookup surfaces as a shape mismatch.
        let d_l: usize = 4;
        let d_g: usize = 6;

        let l_data: Vec<f32> = (0..d_l * d_l).map(|i| 0.5 + i as f32 * 0.1).collect();
        let g_qkv_data: Vec<f32> = (0..3 * d_g * d_g).map(|i| 10.0 + i as f32).collect();

        let path = write_synthetic(
            "sdxl-no-collision-clip-l",
            &[
                // CLIP-L's q_proj — Direct rename to text_model.encoder.layers.0.self_attn.q_proj.weight.
                (
                    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                    vec![d_l, d_l],
                    l_data.clone(),
                ),
                // CLIP-G's fused QKV — FusedSlice expands into THREE diffusers
                // entries, one of which collides with CLIP-L's q_proj above.
                (
                    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
                    vec![3 * d_g, d_g],
                    g_qkv_data,
                ),
            ],
        );
        let bundle = load_bundle(&path, Family::Sdxl).expect("partition sdxl");
        let remap = build_sdxl_remap(&bundle).expect("build remap");

        // Sanity check: the all-in-one `from_sdxl_remap` exhibits the
        // collision — the diffusers q_proj key resolves to CLIP-G's slice
        // (shape `[d_g, d_g]`), not CLIP-L's tensor (shape `[d_l, d_l]`).
        // Documents the bug so a future refactor of `from_sdxl_remap`
        // doesn't silently change the all-in-one semantics without
        // explicit thought.
        let all_in_one = SingleFileBackend::from_sdxl_remap(&path, &remap).expect("backend");
        let collided = SimpleBackend::get_unchecked(
            &all_in_one,
            "text_model.encoder.layers.0.self_attn.q_proj.weight",
            DType::F32,
            &Device::Cpu,
        )
        .expect("lookup");
        assert_eq!(
            collided.dims(),
            &[d_g, d_g],
            "all-in-one from_sdxl_remap must still exhibit the collision (CLIP-G wins) — \
             this is the bug that motivates the scoped factories",
        );

        // Real assertion: CLIP-L-scoped backend returns CLIP-L's tensor.
        let backend_l =
            SingleFileBackend::from_sdxl_clip_l(&path, &remap).expect("clip-l scoped backend");
        let t_l = SimpleBackend::get_unchecked(
            &backend_l,
            "text_model.encoder.layers.0.self_attn.q_proj.weight",
            DType::F32,
            &Device::Cpu,
        )
        .expect("clip-l scoped lookup");
        assert_eq!(
            t_l.dims(),
            &[d_l, d_l],
            "CLIP-L scoped backend must return CLIP-L's [d_l, d_l] tensor, \
             not CLIP-G's [d_g, d_g] slice — collision elimination is the \
             whole point of the scoped factory",
        );
        let flat: Vec<f32> = t_l.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, l_data, "values must match the CLIP-L source tensor");

        // CLIP-G scoped backend gets the slice (independent verification).
        let backend_g =
            SingleFileBackend::from_sdxl_clip_g(&path, &remap).expect("clip-g scoped backend");
        let t_g = SimpleBackend::get_unchecked(
            &backend_g,
            "text_model.encoder.layers.0.self_attn.q_proj.weight",
            DType::F32,
            &Device::Cpu,
        )
        .expect("clip-g scoped lookup");
        assert_eq!(
            t_g.dims(),
            &[d_g, d_g],
            "CLIP-G scoped backend keeps the slice semantics — q_proj is the \
             0th component of the [3*d_g, d_g] in_proj_weight slab",
        );

        let _ = std::fs::remove_file(path);
    }

    /// Companion to the slice/collision test above: scoped CLIP-L backend
    /// must NOT see CLIP-G's `text_projection`, `token_embedding`, or any
    /// other CLIP-G-only key. Locks the per-scope visibility contract.
    #[test]
    fn sdxl_clip_l_scoped_backend_excludes_clip_g_only_keys() {
        let path = write_synthetic(
            "sdxl-clip-l-isolation",
            &[
                (
                    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                    vec![2, 2],
                    vec![0.1, 0.2, 0.3, 0.4],
                ),
                (
                    "conditioner.embedders.1.model.text_projection",
                    vec![1],
                    vec![99.0],
                ),
            ],
        );
        let bundle = load_bundle(&path, Family::Sdxl).unwrap();
        let remap = build_sdxl_remap(&bundle).unwrap();

        let backend_l = SingleFileBackend::from_sdxl_clip_l(&path, &remap).unwrap();
        // text_projection.weight is a CLIP-G-only diffusers key — must NOT
        // be present in the CLIP-L scope.
        assert!(
            !SimpleBackend::contains_tensor(&backend_l, "text_projection.weight"),
            "CLIP-L scoped backend must not advertise CLIP-G-only keys",
        );

        let backend_g = SingleFileBackend::from_sdxl_clip_g(&path, &remap).unwrap();
        assert!(
            SimpleBackend::contains_tensor(&backend_g, "text_projection.weight"),
            "CLIP-G scoped backend must include CLIP-G's text_projection.weight",
        );

        let _ = std::fs::remove_file(path);
    }

    /// Bug 3 from `tasks/catalog-run-bridge-option-c-handoff.md` (surfaced
    /// during the killswitch UAT after Bug 1 + Bug 2 landed): SDXL VAE decode
    /// failed with `shape mismatch in matmul, lhs: [1, 9216, 512],
    /// rhs: [1, 512, 512, 1, 1]` — the rank-5 rhs is a Conv2d weight
    /// `[512, 512, 1, 1]` after `broadcast_left + .t()`, meaning candle
    /// constructed a `Linear` with a Conv2d-shaped weight.
    ///
    /// Cause: candle's `stable_diffusion::attention::get_qkv_linear` probes
    /// for the HF Linear weight shape `(C, C)` first, and on `Err` falls
    /// back to the A1111 Conv2d 1×1 shape `(C, C, 1, 1)` + reshape. The
    /// `Err` branch relies on the SimpleBackend enforcing the requested
    /// shape — every other backend impl in candle (`SafeTensorWithRouting`,
    /// `MmapedSafetensors`, `HashMap<String, Tensor>`, `NpzTensors`,
    /// `PthTensors`) returns `UnexpectedShape` on mismatch.
    ///
    /// Our `SingleFileBackend::get` previously ignored the requested shape,
    /// so the rank-2 probe silently succeeded with the rank-4 on-disk tensor
    /// and candle took the wrong branch. Fix locked here: rank-2 probe must
    /// error so candle's fallback path fires; rank-4 probe must succeed.
    #[test]
    fn backend_get_validates_shape_so_candle_attnblock_falls_through_to_conv_path() {
        let c = 4usize;
        let on_disk: Vec<f32> = (0..c * c).map(|i| 1.0 + i as f32 * 0.1).collect();
        let path = write_synthetic(
            "sd15-vae-attn-conv-shape",
            &[
                // SD1.5 VAE attention query weight in A1111 layout:
                // Conv2d 1×1, shape [C, C, 1, 1].
                (
                    "first_stage_model.encoder.mid.attn_1.q.weight",
                    vec![c, c, 1, 1],
                    on_disk.clone(),
                ),
            ],
        );
        let bundle = load_bundle(&path, Family::Sd15).expect("partition");
        let remap = build_sd15_remap(&bundle).expect("remap");
        let backend = SingleFileBackend::from_sd15_vae(&path, &remap).expect("backend");
        let dev = Device::Cpu;

        // The diffusers-side rename for the VAE mid-block attention query —
        // see `loader::vae_keys::rename_vae_mid_attn`.
        let diffusers_key = "encoder.mid_block.attentions.0.to_q.weight";

        // Probe with the (C, C) Linear shape — must error so candle's
        // get_qkv_linear falls through to the conv path. The on-disk
        // tensor is rank-4, so a rank-2 request must report a shape
        // mismatch.
        let result_rank2 = SimpleBackend::get(
            &backend,
            candle_core::Shape::from((c, c)),
            diffusers_key,
            candle_nn::Init::Const(0.0),
            DType::F32,
            &dev,
        );
        let err = result_rank2.expect_err(
            "rank-2 probe must error so candle's get_qkv_linear falls through to the conv path",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("shape mismatch") || msg.contains("UnexpectedShape"),
            "expected shape-mismatch error so candle's Err-arm fires; got: {msg}",
        );

        // Probe with the actual (C, C, 1, 1) shape — must succeed. After
        // candle's fallback fires, it requests this shape and reshapes
        // the result to (C, C) for the Linear constructor.
        let t = SimpleBackend::get(
            &backend,
            candle_core::Shape::from((c, c, 1, 1)),
            diffusers_key,
            candle_nn::Init::Const(0.0),
            DType::F32,
            &dev,
        )
        .expect("rank-4 probe must succeed for A1111 Conv2d 1×1 weight");
        assert_eq!(t.dims(), &[c, c, 1, 1]);
        let flat: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(flat, on_disk);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn backend_dtype_promotes_when_caller_requests_other_dtype() {
        // candle constructors often request F16 / BF16. The backend
        // stores F32, so a dtype mismatch must convert via to_dtype()
        // before returning to the caller. Verify the conversion fires.
        let path = write_synthetic(
            "sd15-dtype",
            &[(
                "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                vec![1],
                vec![1.0],
            )],
        );
        let bundle = load_bundle(&path, Family::Sd15).unwrap();
        let remap = build_sd15_remap(&bundle).unwrap();
        let backend = SingleFileBackend::from_sd15_remap(&path, &remap).unwrap();

        let t = SimpleBackend::get_unchecked(
            &backend,
            "text_model.encoder.layers.0.self_attn.q_proj.weight",
            DType::F16,
            &Device::Cpu,
        )
        .expect("F16 lookup");

        assert_eq!(t.dtype(), DType::F16);
        let _ = std::fs::remove_file(path);
    }

    // -----------------------------------------------------------------
    // Flux.2 single-file backend tests
    // -----------------------------------------------------------------

    /// Flux.2 Klein config with depth=1, depth_single_blocks=1 — keeps the
    /// synthetic checkpoints small while exercising every per-block code path.
    fn flux2_test_config() -> Flux2Config {
        Flux2Config {
            in_channels: 128,
            vec_in_dim: 0,
            context_in_dim: 7680,
            hidden_size: 3072,
            mlp_ratio: 3.0,
            num_heads: 24,
            depth: 1,
            depth_single_blocks: 1,
            axes_dim: vec![32, 32, 32, 32],
            theta: 2000,
            guidance_embed: false,
        }
    }

    /// Synthetic on-disk Flux.2 BFL-native fixture. Only emits the keys the
    /// rename table consults — every Direct entry above plus the two fused
    /// QKV slabs per double block. Single-block linear1 is a single direct
    /// load (the per-head split happens inside `SingleStreamBlock::forward`,
    /// not at backend lookup time).
    fn write_flux2_bfl_fixture(cfg: &Flux2Config, override_qkv: Option<Vec<f32>>) -> PathBuf {
        write_flux2_bfl_fixture_with_rms(cfg, override_qkv, "scale")
    }

    fn write_flux2_bfl_fixture_with_rms(
        cfg: &Flux2Config,
        override_qkv: Option<Vec<f32>>,
        rms_suffix: &str,
    ) -> PathBuf {
        let prefix = "model.diffusion_model";
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

        let push = |t: &mut Vec<(String, Vec<usize>, Vec<f32>)>, key: &str, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            t.push((key.to_string(), shape, data));
        };

        for (suffix, shape) in [
            ("img_in.weight", vec![1, 1]),
            ("txt_in.weight", vec![1, 1]),
            ("time_in.in_layer.weight", vec![1, 1]),
            ("time_in.out_layer.weight", vec![1, 1]),
            ("final_layer.linear.weight", vec![1, 1]),
            // SwapHalves requires an even axis-0 dim: use 2 rows (scale half + shift half).
            ("final_layer.adaLN_modulation.1.weight", vec![2, 1]),
            ("double_stream_modulation_img.lin.weight", vec![1, 1]),
            ("double_stream_modulation_txt.lin.weight", vec![1, 1]),
            ("single_stream_modulation.lin.weight", vec![1, 1]),
        ] {
            push(&mut tensors, &format!("{prefix}.{suffix}"), shape);
        }

        for i in 0..cfg.depth {
            // Fused QKV slabs (img + txt). Allow caller to inject custom
            // payloads so the slice tests can plant sentinel values.
            let img_qkv_key = format!("{prefix}.double_blocks.{i}.img_attn.qkv.weight");
            let txt_qkv_key = format!("{prefix}.double_blocks.{i}.txt_attn.qkv.weight");
            if let Some(data) = override_qkv.as_ref() {
                let d = (data.len() / 3).isqrt();
                tensors.push((img_qkv_key.clone(), vec![3 * d, d], data.clone()));
                tensors.push((txt_qkv_key.clone(), vec![3 * d, d], data.clone()));
            } else {
                push(&mut tensors, &img_qkv_key, vec![3, 1]);
                push(&mut tensors, &txt_qkv_key, vec![3, 1]);
            }

            let rms_q_img = format!("img_attn.norm.query_norm.{rms_suffix}");
            let rms_k_img = format!("img_attn.norm.key_norm.{rms_suffix}");
            let rms_q_txt = format!("txt_attn.norm.query_norm.{rms_suffix}");
            let rms_k_txt = format!("txt_attn.norm.key_norm.{rms_suffix}");
            for suffix in [
                "img_attn.proj.weight",
                rms_q_img.as_str(),
                rms_k_img.as_str(),
                "img_mlp.0.weight",
                "img_mlp.2.weight",
                "txt_attn.proj.weight",
                rms_q_txt.as_str(),
                rms_k_txt.as_str(),
                "txt_mlp.0.weight",
                "txt_mlp.2.weight",
            ] {
                push(
                    &mut tensors,
                    &format!("{prefix}.double_blocks.{i}.{suffix}"),
                    vec![1, 1],
                );
            }
        }

        for i in 0..cfg.depth_single_blocks {
            let rms_q = format!("norm.query_norm.{rms_suffix}");
            let rms_k = format!("norm.key_norm.{rms_suffix}");
            for suffix in [
                "linear1.weight",
                "linear2.weight",
                rms_q.as_str(),
                rms_k.as_str(),
            ] {
                push(
                    &mut tensors,
                    &format!("{prefix}.single_blocks.{i}.{suffix}"),
                    vec![1, 1],
                );
            }
        }

        let refs: Vec<(&str, Vec<usize>, Vec<f32>)> = tensors
            .iter()
            .map(|(k, s, d)| (k.as_str(), s.clone(), d.clone()))
            .collect();
        write_synthetic("flux2-bfl", &refs)
    }

    /// Write a BFL-native root-layout fixture where `img_mlp.0` is FP8 with a
    /// per-tensor scale under `scale_key`, mirroring the mixed FP8 files BFL
    /// and Comfy-Org publish (attention BF16, MLP slabs FP8).
    fn write_flux2_fp8_fixture(scale_key: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mold-sf-backend-flux2-fp8-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));

        // Two FP8 bytes: 0x40 == 2.0, 0x38 == 1.0 in e4m3.
        let fp8 = vec![0x40u8, 0x40];
        let scale = 0.25f32.to_le_bytes().to_vec();
        let ones = 1.0f32.to_le_bytes().to_vec();
        let two_rows: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
        let insert_f32 = |views: &mut HashMap<String, TensorView<'_>>,
                          key: &str,
                          shape: Vec<usize>,
                          buf: &'static [u8]| {
            views.insert(
                key.to_string(),
                TensorView::new(SafeDtype::F32, shape, buf).unwrap(),
            );
        };
        let ones_leaked: &'static [u8] = Box::leak(ones.into_boxed_slice());
        let two_leaked: &'static [u8] = Box::leak(two_rows.into_boxed_slice());
        let fp8_leaked: &'static [u8] = Box::leak(fp8.into_boxed_slice());
        let scale_leaked: &'static [u8] = Box::leak(scale.into_boxed_slice());

        for key in [
            "img_in.weight",
            "txt_in.weight",
            "time_in.in_layer.weight",
            "time_in.out_layer.weight",
            "final_layer.linear.weight",
            "double_stream_modulation_img.lin.weight",
            "double_stream_modulation_txt.lin.weight",
            "single_stream_modulation.lin.weight",
            "double_blocks.0.img_attn.proj.weight",
            "double_blocks.0.img_attn.norm.query_norm.scale",
            "double_blocks.0.img_attn.norm.key_norm.scale",
            "double_blocks.0.img_mlp.2.weight",
            "double_blocks.0.txt_attn.proj.weight",
            "double_blocks.0.txt_attn.norm.query_norm.scale",
            "double_blocks.0.txt_attn.norm.key_norm.scale",
            "double_blocks.0.txt_mlp.0.weight",
            "double_blocks.0.txt_mlp.2.weight",
            "single_blocks.0.linear1.weight",
            "single_blocks.0.linear2.weight",
            "single_blocks.0.norm.query_norm.scale",
            "single_blocks.0.norm.key_norm.scale",
        ] {
            insert_f32(&mut views, key, vec![1, 1], ones_leaked);
        }
        insert_f32(
            &mut views,
            "final_layer.adaLN_modulation.1.weight",
            vec![2, 1],
            two_leaked,
        );
        for key in [
            "double_blocks.0.img_attn.qkv.weight",
            "double_blocks.0.txt_attn.qkv.weight",
        ] {
            views.insert(
                key.to_string(),
                TensorView::new(SafeDtype::F32, vec![3, 1], {
                    static QKV: [u8; 12] = [0; 12];
                    &QKV
                })
                .unwrap(),
            );
        }
        // The one FP8 layer plus its scale sidecar.
        views.insert(
            "double_blocks.0.img_mlp.0.weight".to_string(),
            TensorView::new(SafeDtype::F8_E4M3, vec![2, 1], fp8_leaked).unwrap(),
        );
        views.insert(
            format!("double_blocks.0.img_mlp.0.{scale_key}"),
            TensorView::new(SafeDtype::F32, vec![], scale_leaked).unwrap(),
        );
        serialize_to_file(&views, &None, &path).unwrap();
        path
    }

    /// Both spellings of the per-tensor FP8 scale must reach the transformer
    /// under the one subkey `Flux2Linear::new` probes for. Without this the
    /// scale is simply absent and the layer's weights are ~500x too large.
    #[test]
    fn flux2_fp8_scale_sidecars_resolve_under_one_subkey() {
        for scale_key in ["weight_scale", "scale_weight"] {
            let cfg = flux2_test_config();
            let path = write_flux2_fp8_fixture(scale_key);
            let backend =
                SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("fp8 backend");
            let diffusers_scale = "transformer_blocks.0.ff.linear_in.scale_weight";
            assert!(
                backend.contains_tensor(diffusers_scale),
                "{scale_key}: no rule for {diffusers_scale}",
            );
            let scale = backend
                .get_unchecked(diffusers_scale, DType::F32, &Device::Cpu)
                .expect("scale lookup");
            assert_eq!(scale.rank(), 0);
            assert_eq!(scale.to_vec0::<f32>().unwrap(), 0.25);
            std::fs::remove_file(&path).ok();
        }
    }

    /// The FP8 weight itself must survive the VarBuilder's dtype: casting it
    /// to BF16 at load makes `Flux2Linear::new` take its unscaled BF16 arm,
    /// which silently drops the scale the sidecar exists to supply. BF16
    /// tensors in the same mixed file still cast.
    #[test]
    fn flux2_fp8_weights_keep_their_dtype_while_the_rest_casts() {
        let cfg = flux2_test_config();
        let path = write_flux2_fp8_fixture("weight_scale");
        let backend = SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("fp8 backend");

        let fp8 = backend
            .get_unchecked(
                "transformer_blocks.0.ff.linear_in.weight",
                DType::BF16,
                &Device::Cpu,
            )
            .expect("fp8 weight");
        assert_eq!(fp8.dtype(), DType::F8E4M3);

        let bf16 = backend
            .get_unchecked(
                "transformer_blocks.0.ff.linear_out.weight",
                DType::BF16,
                &Device::Cpu,
            )
            .expect("bf16 weight");
        assert_eq!(bf16.dtype(), DType::BF16);
        std::fs::remove_file(&path).ok();
    }

    /// A checkpoint with no FP8 tensors must behave exactly as before: no
    /// scale entries, and every weight cast to the VarBuilder's dtype.
    #[test]
    fn flux2_bf16_checkpoints_gain_no_fp8_entries() {
        let cfg = flux2_test_config();
        let path = write_flux2_bfl_fixture(&cfg, None);
        let backend = SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("backend");
        assert!(!backend.preserve_fp8);
        assert!(!backend
            .entries
            .keys()
            .any(|key| key.ends_with(".scale_weight")));
        let w = backend
            .get_unchecked("x_embedder.weight", DType::BF16, &Device::Cpu)
            .expect("weight");
        assert_eq!(w.dtype(), DType::BF16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn flux2_singlefile_backend_remaps_top_level_keys() {
        let cfg = flux2_test_config();
        let path = write_flux2_bfl_fixture(&cfg, None);
        let backend =
            SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("flux2 backend must load");

        // Spot-check every top-level diffusers key — confirms the rename
        // table maps to the right BFL-native source by data round-trip.
        let dev = Device::Cpu;
        for diffusers_key in [
            "x_embedder.weight",
            "context_embedder.weight",
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            "proj_out.weight",
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_txt.linear.weight",
            "single_stream_modulation.linear.weight",
        ] {
            let t = SimpleBackend::get_unchecked(&backend, diffusers_key, DType::F32, &dev)
                .unwrap_or_else(|e| panic!("{diffusers_key}: {e}"));
            assert_eq!(t.dims(), &[1, 1], "{diffusers_key}: shape");
        }
        // norm_out.linear.weight uses SwapHalves on a [2, 1] source, so shape
        // is preserved as [2, 1] after the swap (swaps rows 0↔1, same dims).
        let t = SimpleBackend::get_unchecked(&backend, "norm_out.linear.weight", DType::F32, &dev)
            .expect("norm_out.linear.weight must be accessible");
        assert_eq!(
            t.dims(),
            &[2, 1],
            "norm_out.linear.weight: SwapHalves preserves shape"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux2_singlefile_backend_slices_double_block_qkv() {
        // 3 × d × d fused slab: rows 0..d carry sentinel 1.0 (Q),
        // d..2d carry 2.0 (K), 2d..3d carry 3.0 (V).
        let d = 4usize;
        let mut data: Vec<f32> = Vec::with_capacity(3 * d * d);
        for component in 1..=3 {
            for _ in 0..d {
                for _ in 0..d {
                    data.push(component as f32);
                }
            }
        }

        let cfg = flux2_test_config();
        let path = write_flux2_bfl_fixture(&cfg, Some(data));
        let backend = SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("backend");
        let dev = Device::Cpu;

        for (component, sentinel, name) in
            [(0usize, 1.0f32, "to_q"), (1, 2.0, "to_k"), (2, 3.0, "to_v")]
        {
            let key = format!("transformer_blocks.0.attn.{name}.weight");
            let t = SimpleBackend::get_unchecked(&backend, &key, DType::F32, &dev)
                .unwrap_or_else(|e| panic!("{key}: {e}"));
            assert_eq!(t.dims(), &[d, d], "{key}: slice shape");
            let flat: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                flat.iter().all(|&v| v == sentinel),
                "{key} (component {component}): values must all be {sentinel}, got {flat:?}",
            );
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux2_singlefile_backend_loads_rms_norm_weight_suffix() {
        // Some community BF16 Klein-9B fine-tunes (cv:2765147 prototype) name
        // the BFL RMSNorm tensors `*.norm.{query,key}_norm.weight` instead of
        // the canonical `.scale`. The header probe must detect that and
        // remap accordingly so every diffusers `attn.norm_{q,k}.weight`
        // (and `attn.norm_added_{q,k}.weight`, plus single-block `norm_{q,k}`)
        // resolves to the on-disk tensor without a missing-key error.
        let cfg = flux2_test_config();
        let path = write_flux2_bfl_fixture_with_rms(&cfg, None, "weight");
        let backend = SingleFileBackend::from_flux2_singlefile(&path, &cfg)
            .expect("RMSNorm `.weight`-suffix checkpoint must load");

        let dev = Device::Cpu;
        for diffusers_key in [
            "transformer_blocks.0.attn.norm_q.weight",
            "transformer_blocks.0.attn.norm_k.weight",
            "transformer_blocks.0.attn.norm_added_q.weight",
            "transformer_blocks.0.attn.norm_added_k.weight",
            "single_transformer_blocks.0.attn.norm_q.weight",
            "single_transformer_blocks.0.attn.norm_k.weight",
        ] {
            SimpleBackend::get_unchecked(&backend, diffusers_key, DType::F32, &dev)
                .unwrap_or_else(|e| panic!("{diffusers_key}: {e}"));
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux2_singlefile_backend_slices_double_block_added_qkv() {
        let d = 4usize;
        let mut data: Vec<f32> = Vec::with_capacity(3 * d * d);
        for component in 1..=3 {
            for _ in 0..d {
                for _ in 0..d {
                    data.push((component as f32) * 10.0);
                }
            }
        }

        let cfg = flux2_test_config();
        let path = write_flux2_bfl_fixture(&cfg, Some(data));
        let backend = SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("backend");
        let dev = Device::Cpu;

        for (component, sentinel, name) in [
            (0usize, 10.0f32, "add_q_proj"),
            (1, 20.0, "add_k_proj"),
            (2, 30.0, "add_v_proj"),
        ] {
            let key = format!("transformer_blocks.0.attn.{name}.weight");
            let t = SimpleBackend::get_unchecked(&backend, &key, DType::F32, &dev)
                .unwrap_or_else(|e| panic!("{key}: {e}"));
            assert_eq!(t.dims(), &[d, d], "{key}: slice shape");
            let flat: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                flat.iter().all(|&v| v == sentinel),
                "{key} (component {component}): values must all be {sentinel}",
            );
        }

        let _ = std::fs::remove_file(path);
    }

    /// Write a safetensors with mixed-dtype tensors. Each tuple is
    /// (key, on-disk dtype, shape, raw bytes). Bytes ownership is passed in.
    fn write_typed_synthetic(
        name: &str,
        tensors: Vec<(String, SafeDtype, Vec<usize>, Vec<u8>)>,
    ) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mold-sf-backend-typed-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let owned: Vec<(String, SafeDtype, Vec<usize>, Vec<u8>)> = tensors;
        let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
        for (key, dtype, shape, bytes) in owned.iter() {
            views.insert(
                key.clone(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            );
        }
        serialize_to_file(&views, &None, &path).unwrap();
        path
    }

    /// Smallest NVFP4 layer fixture: one block of 16 weights at one row.
    /// All nibbles are 0b0010 (E2M1 = +1.0); block scale 0x38 (E4M3 = 1.0);
    /// per-tensor scale `weight_scale_2`. The fully-dequanted weight is
    /// 16 copies of `weight_scale_2`.
    fn one_layer_nvfp4_bytes(weight_scale_2: f32) -> Vec<(String, SafeDtype, Vec<usize>, Vec<u8>)> {
        // 8 packed bytes (= 16 weights), each byte = 0x22 (low + high nibble both 2).
        let weight_bytes = vec![0x22u8; 8];
        // 1 block scale byte: 0x38 = E4M3 1.0.
        let scale_bytes = vec![0x38u8];
        let scale_2_bytes = weight_scale_2.to_le_bytes().to_vec();
        let base = "model.diffusion_model.double_blocks.0.img_attn.proj";
        vec![
            (
                format!("{base}.weight"),
                SafeDtype::U8,
                vec![1, 8],
                weight_bytes,
            ),
            (
                format!("{base}.weight_scale"),
                SafeDtype::F8_E4M3,
                vec![1, 1],
                scale_bytes,
            ),
            (
                format!("{base}.weight_scale_2"),
                SafeDtype::F32,
                vec![],
                scale_2_bytes,
            ),
        ]
    }

    #[test]
    fn flux2_singlefile_backend_accepts_nvfp4_format() {
        // Minimal NVFP4 fixture sufficient for the header-detector to trip.
        // Construction must succeed (was previously rejected with anyhow).
        let path = write_typed_synthetic("flux2-nvfp4-accept", one_layer_nvfp4_bytes(0.5));
        let cfg = flux2_test_config();
        let _backend = SingleFileBackend::from_flux2_singlefile(&path, &cfg)
            .expect("NVFP4 checkpoint must now load (no longer rejected)");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux2_nvfp4_emits_three_subkeys_per_layer() {
        // For every NVFP4-quantised layer the backend must emit the
        // `weight.nvfp4_packed`, `weight.nvfp4_block_scales`, and
        // `weight.nvfp4_tensor_scale` sub-keys. Streaming `Flux2Linear`
        // probes for these to detect NVFP4 layers.
        let path = write_typed_synthetic("flux2-nvfp4-subkeys", one_layer_nvfp4_bytes(0.5));
        let cfg = flux2_test_config();
        let backend = SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("nvfp4 backend");

        let base = "transformer_blocks.0.attn.to_out.0.weight";
        for sub in ["nvfp4_packed", "nvfp4_block_scales", "nvfp4_tensor_scale"] {
            let key = format!("{base}.{sub}");
            assert!(
                SimpleBackend::contains_tensor(&backend, &key),
                "{key}: NVFP4 sub-key must be present",
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux2_nvfp4_does_not_emit_bare_weight_for_nvfp4_layers() {
        // Streaming dequant routes everything through NVFP4 sub-keys;
        // the bare `weight` lookup must NOT resolve for NVFP4 layers (or
        // `Flux2Linear::load_with_bias` would fall through to the FP8/Standard
        // path and try to load a non-existent fused tensor).
        let path = write_typed_synthetic("flux2-nvfp4-no-bare", one_layer_nvfp4_bytes(0.5));
        let cfg = flux2_test_config();
        let backend = SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("nvfp4 backend");

        for key in [
            "transformer_blocks.0.attn.to_out.0.weight",
            "transformer_blocks.0.attn.to_out.0.scale_weight",
        ] {
            assert!(
                !SimpleBackend::contains_tensor(&backend, key),
                "{key}: bare weight / scale_weight must NOT exist for NVFP4 layers (sub-keys only)",
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux2_nvfp4_subkey_lookup_returns_cpu_tensors_with_correct_dtypes() {
        // Each sub-key returns a CPU tensor with the on-disk dtype:
        // packed → U8, block_scales → F8E4M3, tensor_scale → F32 scalar.
        // These are the inputs `Flux2Linear::Nvfp4Streaming` consumes.
        let path = write_typed_synthetic("flux2-nvfp4-subkey-dtypes", one_layer_nvfp4_bytes(0.5));
        let cfg = flux2_test_config();
        let backend = SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("backend");
        let base = "transformer_blocks.0.attn.to_out.0.weight";
        let dev = Device::Cpu;

        let packed = SimpleBackend::get_unchecked(
            &backend,
            &format!("{base}.nvfp4_packed"),
            DType::U8,
            &dev,
        )
        .expect("packed lookup");
        assert_eq!(packed.dtype(), DType::U8);
        assert_eq!(packed.dims(), &[1, 8]);

        let scales = SimpleBackend::get_unchecked(
            &backend,
            &format!("{base}.nvfp4_block_scales"),
            DType::F8E4M3,
            &dev,
        )
        .expect("scales lookup");
        assert_eq!(scales.dtype(), DType::F8E4M3);

        let tscale = SimpleBackend::get_unchecked(
            &backend,
            &format!("{base}.nvfp4_tensor_scale"),
            DType::F32,
            &dev,
        )
        .expect("tensor_scale lookup");
        assert_eq!(tscale.dtype(), DType::F32);
        let v: f32 = tscale.to_scalar().unwrap();
        assert!((v - 0.5).abs() < 1e-6, "tensor_scale must be 0.5, got {v}",);

        let _ = std::fs::remove_file(path);
    }

    /// Sliced QKV NVFP4: each `to_q`/`to_k`/`to_v` diffusers key gets the
    /// three NVFP4 sub-keys *plus* a `nvfp4_slice_meta` sub-key encoding
    /// `[axis, component, num_components]` as U32 [3]. The streaming
    /// `Flux2Linear` reads this metadata at construction time so it can
    /// narrow the dequanted full weight at forward time.
    #[test]
    fn flux2_nvfp4_slice_qkv_emits_meta() {
        // Build a synthetic fused-QKV NVFP4 fixture: img_attn.qkv with N=3,
        // K=16. 1 byte per row × 3 rows of packed FP4 = 3 packed bytes
        // *per K/2 column*; for N=3, K/2=8 → 24 bytes total. Row 0 = Q
        // (all 0x22 = nibble 2 = +1.0), row 1 = K (0x44 = nibble 4 = +2.0),
        // row 2 = V (0x66 = nibble 6 = +4.0). Block scales = 1.0 each.
        let weight_bytes: Vec<u8> = (0..3)
            .flat_map(|n| {
                let nibble = match n {
                    0 => 0x22,
                    1 => 0x44,
                    2 => 0x66,
                    _ => unreachable!(),
                };
                vec![nibble; 8]
            })
            .collect();
        let scale_bytes = vec![0x38u8; 3]; // 3 rows × 1 block each, all = 1.0
        let scale_2 = 0.5f32;
        let scale_2_bytes = scale_2.to_le_bytes().to_vec();

        let qkv_base = "model.diffusion_model.double_blocks.0.img_attn.qkv";
        let tensors = vec![
            (
                format!("{qkv_base}.weight"),
                SafeDtype::U8,
                vec![3, 8],
                weight_bytes,
            ),
            (
                format!("{qkv_base}.weight_scale"),
                SafeDtype::F8_E4M3,
                vec![3, 1],
                scale_bytes,
            ),
            (
                format!("{qkv_base}.weight_scale_2"),
                SafeDtype::F32,
                vec![],
                scale_2_bytes,
            ),
        ];
        let path = write_typed_synthetic("flux2-nvfp4-slice-meta", tensors);
        let cfg = flux2_test_config();
        let backend = SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("backend");

        for (component, comp_name) in [(0u32, "to_q"), (1, "to_k"), (2, "to_v")] {
            let meta_key = format!("transformer_blocks.0.attn.{comp_name}.weight.nvfp4_slice_meta");
            assert!(
                SimpleBackend::contains_tensor(&backend, &meta_key),
                "{meta_key}: slice meta sub-key must be present",
            );
            let meta = SimpleBackend::get_unchecked(&backend, &meta_key, DType::U32, &Device::Cpu)
                .expect("slice meta lookup");
            assert_eq!(meta.dtype(), DType::U32);
            assert_eq!(meta.dims(), &[3]);
            let v: Vec<u32> = meta.to_vec1().unwrap();
            assert_eq!(
                v,
                vec![0u32, component, 3],
                "{meta_key}: meta must encode [axis=0, component={component}, num_components=3]",
            );
        }

        // The three sliced linears all reference the same fused source, so
        // their packed sub-keys all resolve to the same on-disk tensor.
        for comp_name in ["to_q", "to_k", "to_v"] {
            let packed_key = format!("transformer_blocks.0.attn.{comp_name}.weight.nvfp4_packed");
            assert!(
                SimpleBackend::contains_tensor(&backend, &packed_key),
                "{packed_key}: packed sub-key must be present for sliced QKV component",
            );
        }

        let _ = std::fs::remove_file(path);
    }

    /// Real-file load probe — gated behind `MOLD_NVFP4_PROBE_PATH` so it
    /// only runs when explicitly pointed at a Civitai NVFP4 Flux.2 Klein-9B
    /// checkpoint. Constructs the backend, builds a `VarBuilder` over it,
    /// and instantiates `Flux2Transformer::new(klein_9b(), vb)` end-to-end —
    /// which exercises every linear-layer load (`Flux2Linear::Fp8`), every
    /// fused-QKV slice, every norm passthrough, and the NVFP4 dequant on a
    /// real on-disk file. Ignored by default; run with:
    ///
    /// ```bash
    /// MOLD_NVFP4_PROBE_PATH=/path/to/file.safetensors \
    ///   cargo test -p mold-ai-inference --lib \
    ///   loader::single_file_backend::tests::flux2_nvfp4_real_file_loads_full_klein_9b -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires MOLD_NVFP4_PROBE_PATH env var pointing at a real NVFP4 .safetensors"]
    fn flux2_nvfp4_real_file_loads_full_klein_9b() {
        use crate::flux2::transformer::Flux2Transformer;
        use std::time::Instant;

        let path = match std::env::var("MOLD_NVFP4_PROBE_PATH") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                eprintln!("skipping: MOLD_NVFP4_PROBE_PATH not set");
                return;
            }
        };
        assert!(
            path.is_file(),
            "MOLD_NVFP4_PROBE_PATH must point at a real file (got {})",
            path.display(),
        );
        let size_gb = std::fs::metadata(&path).unwrap().len() as f64 / 1e9;
        eprintln!("probing NVFP4 file: {} ({:.2} GB)", path.display(), size_gb,);

        let cfg = crate::flux2::Flux2Config::klein_9b();
        let t0 = Instant::now();
        let backend =
            SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("backend construction");
        eprintln!("  backend constructed in {:?}", t0.elapsed());

        let dev = Device::Cpu;
        let vb = candle_nn::VarBuilder::from_backend(Box::new(backend), DType::BF16, dev.clone());

        let t1 = Instant::now();
        let _transformer = Flux2Transformer::new(&cfg, vb)
            .expect("Flux2Transformer::new must succeed end-to-end on the real NVFP4 checkpoint");
        eprintln!("  transformer loaded in {:?}", t1.elapsed());
        eprintln!(
            "  total time: {:?} (every NVFP4 layer set up streaming dequant; BF16 cache populated lazily on first forward)",
            t0.elapsed(),
        );
    }

    #[test]
    fn flux2_singlefile_backend_rejects_non_bfl_native_under_nvfp4_routing_too() {
        // Non-BFL-native (root-level diffusers) checkpoints still error
        // with the same legible message regardless of quantization mode.
        let path = write_synthetic(
            "flux2-already-diffusers-no-prefix",
            &[("x_embedder.weight", vec![1, 1], vec![1.0])],
        );
        let cfg = flux2_test_config();
        let err = match SingleFileBackend::from_flux2_singlefile(&path, &cfg) {
            Ok(_) => panic!("non-BFL-native must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("model.diffusion_model"),
            "error must mention model.diffusion_model, got: {err}",
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_safetensors_not_truncated_passes_for_intact_file() {
        let path = write_synthetic(
            "intact",
            &[(
                "model.diffusion_model.img_in.weight",
                vec![2, 2],
                vec![1.0, 2.0, 3.0, 4.0],
            )],
        );
        check_safetensors_not_truncated(&path).expect("intact file must validate");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn check_safetensors_not_truncated_flags_short_file() {
        // Mirror cv:2739091: write a valid safetensors then chop trailing
        // tensor bytes so the on-disk size is shorter than the header
        // declares. The validator must reject with a message that names
        // the byte gap and points at re-downloading.
        let path = write_synthetic(
            "truncated",
            &[(
                "model.diffusion_model.img_in.weight",
                vec![4, 4],
                (0..16).map(|i| i as f32).collect(),
            )],
        );
        let full_size = std::fs::metadata(&path).unwrap().len();
        let truncated_size = full_size - 16; // drop the last 4 f32 elements
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(truncated_size).unwrap();
        drop(f);

        let err =
            check_safetensors_not_truncated(&path).expect_err("truncated file must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("truncated"),
            "error must say 'truncated', got: {msg}",
        );
        assert!(
            msg.contains("missing") || msg.contains("Re-download") || msg.contains("re-fetch"),
            "error must hint at re-downloading, got: {msg}",
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn from_flux2_singlefile_surfaces_truncation_clearly() {
        // End-to-end on the public entry point used by cv:2739091. The
        // outer `with_context` wrapper carries through the chain so the
        // user-visible message contains both "validate" and "truncated".
        let cfg = flux2_test_config();
        let path = write_flux2_bfl_fixture(&cfg, None);
        let full_size = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(full_size - 4).unwrap();
        drop(f);

        let err = match SingleFileBackend::from_flux2_singlefile(&path, &cfg) {
            Ok(_) => panic!("truncated Flux.2 single-file must be rejected before mmap"),
            Err(e) => e,
        };
        let chained = format!("{err:#}");
        assert!(
            chained.contains("validate single-file checkpoint") && chained.contains("truncated"),
            "expected outer wrapper + truncated root, got: {chained}",
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux2_singlefile_backend_rejects_non_bfl_native() {
        // Diffusers-naming root keys with no `model.diffusion_model.` prefix.
        let path = write_synthetic(
            "flux2-already-diffusers",
            &[("x_embedder.weight", vec![1, 1], vec![1.0])],
        );
        let cfg = flux2_test_config();
        let err = match SingleFileBackend::from_flux2_singlefile(&path, &cfg) {
            Ok(_) => panic!("non-BFL-native must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("model.diffusion_model"),
            "error must mention model.diffusion_model, got: {err}",
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn flux2_singlefile_backend_swaps_ada_ln_halves_for_diffusers_ordering() {
        // BFL-native checkpoints store `final_layer.adaLN_modulation.1.weight`
        // in (shift, scale) row order (BFL format), but mold's LastLayer and
        // diffusers both expect (scale, shift). SingleFileBackend must swap the
        // two halves when loading `norm_out.linear.weight`.
        //
        // Plant sentinel values: rows 0..N/2 = 10.0 (shift half),
        //                        rows N/2..N = 20.0 (scale half).
        // After SwapHalves, rows 0..N/2 must be 20.0, rows N/2..N must be 10.0.
        let n = 4usize; // even; each half is 2 rows
                        // BFL ordering: first half = shift (10.0), second half = scale (20.0).
                        // After SwapHalves the halves are exchanged → first = scale (20.0), second = shift (10.0).
        let mut ada_data: Vec<f32> = vec![10.0f32; n / 2]; // shift half
        ada_data.extend(vec![20.0f32; n / 2]); // scale half

        let cfg = flux2_test_config();
        let prefix = "model.diffusion_model";

        // Write a fixture with the sentinel ada_ln weight and stubs for
        // all other keys the backend validation needs.
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for suffix in [
            "img_in.weight",
            "txt_in.weight",
            "time_in.in_layer.weight",
            "time_in.out_layer.weight",
            "final_layer.linear.weight",
            "double_stream_modulation_img.lin.weight",
            "double_stream_modulation_txt.lin.weight",
            "single_stream_modulation.lin.weight",
        ] {
            tensors.push((format!("{prefix}.{suffix}"), vec![1, 1], vec![0.0f32]));
        }
        // ada_ln modulation weight: n rows × 1 col, with sentinel pattern
        tensors.push((
            format!("{prefix}.final_layer.adaLN_modulation.1.weight"),
            vec![n, 1],
            ada_data,
        ));
        // stubs for double / single blocks
        for i in 0..cfg.depth {
            for suffix in [
                "img_attn.qkv.weight",
                "txt_attn.qkv.weight",
                "img_attn.proj.weight",
                "img_attn.norm.query_norm.scale",
                "img_attn.norm.key_norm.scale",
                "img_mlp.0.weight",
                "img_mlp.2.weight",
                "txt_attn.proj.weight",
                "txt_attn.norm.query_norm.scale",
                "txt_attn.norm.key_norm.scale",
                "txt_mlp.0.weight",
                "txt_mlp.2.weight",
            ] {
                tensors.push((
                    format!("{prefix}.double_blocks.{i}.{suffix}"),
                    vec![3, 1],
                    vec![0.0; 3],
                ));
            }
        }
        for i in 0..cfg.depth_single_blocks {
            for suffix in [
                "single_blocks.attn.to_qkv_mlp_proj.weight",
                "single_blocks.attn.to_out.weight",
                "single_blocks.attn.norm.query_norm.scale",
                "single_blocks.attn.norm.key_norm.scale",
            ] {
                // Use individual block index in suffix
                let _ = i; // suppress unused warning
                tensors.push((
                    format!("{prefix}.single_blocks.{i}.{suffix}"),
                    vec![1, 1],
                    vec![0.0],
                ));
            }
        }

        let path = write_synthetic_with_tensors("flux2-ada-swap-test", &tensors);
        let backend =
            SingleFileBackend::from_flux2_singlefile(&path, &cfg).expect("backend must load");

        let dev = Device::Cpu;
        let t = SimpleBackend::get_unchecked(&backend, "norm_out.linear.weight", DType::F32, &dev)
            .expect("norm_out.linear.weight must be accessible");

        assert_eq!(t.dims(), &[n, 1], "SwapHalves must preserve shape ({n}, 1)");
        let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        // After swap: first half should be the original second half (20.0 = scale)
        for (i, &v) in vals[..n / 2].iter().enumerate() {
            assert!(
                (v - 20.0).abs() < 1e-6,
                "row {i}: expected 20.0 (scale, now first) after swap, got {v}",
            );
        }
        // After swap: second half should be the original first half (10.0 = shift)
        for (i, &v) in vals[n / 2..].iter().enumerate() {
            assert!(
                (v - 10.0).abs() < 1e-6,
                "row {i}: expected 10.0 (shift, now second) after swap, got {v}",
            );
        }
        let _ = std::fs::remove_file(path);
    }
}
