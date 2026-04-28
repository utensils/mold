//! Custom `SimpleBackend` for Civitai single-file checkpoints (phase 2.6).
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

use crate::loader::{RenameOutput, Sd15Remap, SdxlRemap};
use anyhow::{Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Tensor};
use candle_nn::var_builder::SimpleBackend;
use std::collections::BTreeMap;
use std::path::Path;

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
}

impl SingleFileBackend {
    /// Construct from an SD1.5 remap. Every entry is `Direct` since
    /// SD1.5 has no fused QKV slabs (CLIP-L is HF layout, not OpenCLIP).
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

        let st = unsafe { MmapedSafetensors::new(checkpoint) }
            .with_context(|| format!("mmap single-file checkpoint at {}", checkpoint.display()))?;

        Ok(Self { st, entries })
    }

    /// Construct from an SDXL remap. UNet / VAE / CLIP-L are `Direct`;
    /// CLIP-G threads `RenameOutput` through — `Direct(_)` becomes a
    /// `Direct` entry, `FusedSlice {axis, component, num_components, …}`
    /// becomes a `Slice` entry.
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
        for (diffusers, (a1111_key, output)) in &remap.clip_g {
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

        let st = unsafe { MmapedSafetensors::new(checkpoint) }
            .with_context(|| format!("mmap single-file checkpoint at {}", checkpoint.display()))?;

        Ok(Self { st, entries })
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
        }
    }
}

impl SimpleBackend for SingleFileBackend {
    fn get(
        &self,
        _shape: candle_core::Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let t = self.lookup(name, dev)?;
        if t.dtype() != dtype {
            t.to_dtype(dtype)
        } else {
            Ok(t)
        }
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let t = self.lookup(name, dev)?;
        if t.dtype() != dtype {
            t.to_dtype(dtype)
        } else {
            Ok(t)
        }
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
}
