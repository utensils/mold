//! Deterministic local quantization for Hunyuan3D shape checkpoints.
//!
//! Tencent publishes the 2.0 and 2.1 shape families as combined safetensors
//! files: `model.*` is the shape transformer, while `vae.*` and
//! `conditioner.*` are the geometry VAE and vision encoder. The first policy
//! deliberately quantizes transformer matrix weights only. Norms, biases,
//! MoE routing gates, the complete vision tower, and the complete VAE retain
//! their source float storage until separate numerical qualification exists.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device};
use sha2::{Digest, Sha256};

pub const POLICY_VERSION: &str = "hunyuan3d-shape-linear-v3";
pub const SUPPORTED_POLICY_VERSIONS: &[&str] = &[
    "hunyuan3d-shape-linear-v1",
    "hunyuan3d-shape-linear-v2",
    POLICY_VERSION,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShapeQuantization {
    Q8,
    Q6,
    Q5,
    Q4,
    Q3,
}

impl ShapeQuantization {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Q8 => "q8",
            Self::Q6 => "q6",
            Self::Q5 => "q5",
            Self::Q4 => "q4",
            Self::Q3 => "q3",
        }
    }

    fn dtype(self) -> GgmlDType {
        match self {
            Self::Q8 => GgmlDType::Q8_0,
            Self::Q6 => GgmlDType::Q6K,
            Self::Q5 => GgmlDType::Q5K,
            Self::Q4 => GgmlDType::Q4K,
            Self::Q3 => GgmlDType::Q3K,
        }
    }
}

impl fmt::Display for ShapeQuantization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.tag())
    }
}

impl FromStr for ShapeQuantization {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "q8" | "q8_0" => Ok(Self::Q8),
            "q6" | "q6_k" => Ok(Self::Q6),
            "q5" | "q5_k_m" => Ok(Self::Q5),
            "q4" | "q4_k_m" => Ok(Self::Q4),
            "q3" | "q3_k_m" => Ok(Self::Q3),
            other => bail!(
                "unsupported Hunyuan3D quantization {other}; expected q8, q6, q5, q4, or q3"
            ),
        }
    }
}

fn float_storage_dtype(dtype: DType) -> Result<GgmlDType> {
    match dtype {
        DType::F16 => Ok(GgmlDType::F16),
        DType::BF16 => Ok(GgmlDType::BF16),
        DType::F32 => Ok(GgmlDType::F32),
        other => bail!("Hunyuan3D quantization cannot preserve {other:?} source storage"),
    }
}

fn is_router(name: &str) -> bool {
    name.ends_with(".moe.gate.weight") || name.contains(".moe.gate.")
}

fn is_precision_sensitive(name: &str) -> bool {
    [
        "model.latent_in.",
        "model.cond_in.",
        "model.time_in.",
        "model.guidance_in.",
        "model.x_embedder.",
        "model.t_embedder.",
        "model.final_layer.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
        || name.contains(".img_mod.")
        || name.contains(".txt_mod.")
        || name.contains(".modulation.")
        || name.contains(".adaLN_modulation.")
}

fn storage_dtype(
    name: &str,
    dims: &[usize],
    source_dtype: DType,
    tier: ShapeQuantization,
    moe_model: bool,
) -> Result<GgmlDType> {
    let quantized = tier.dtype();
    // Hunyuan3D 2.1's dense path is substantially more sensitive below Q8.
    // Its sparse experts hold most transformer parameters, so quantizing only
    // those retains useful compression while preserving the always-active
    // attention, dense MLP, shared expert, router, and modulation paths.
    let qualified_moe_layer = !moe_model
        || tier == ShapeQuantization::Q8
        || name.contains(".moe.experts.");
    let eligible = name.starts_with("model.")
        && name.ends_with(".weight")
        && dims.len() == 2
        && !is_router(name)
        && !is_precision_sensitive(name)
        && qualified_moe_layer
        && dims[1].is_multiple_of(quantized.block_size());
    if eligible {
        Ok(quantized)
    } else {
        float_storage_dtype(source_dtype)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizationReport {
    pub source: PathBuf,
    pub output: PathBuf,
    pub source_sha256: String,
    pub source_bytes: u64,
    pub output_bytes: u64,
    pub tensor_count: usize,
    pub quantized_tensor_count: usize,
    pub tier: ShapeQuantization,
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("open source checkpoint {}", path.display()))?,
    );
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("hash source checkpoint {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Convert one combined Hunyuan3D safetensors checkpoint into mold's
/// deterministic GGUF layout. The source is read-only and the destination is
/// created atomically; an existing destination is never replaced.
pub fn quantize_checkpoint(
    source: &Path,
    output: &Path,
    source_model: &str,
    tier: ShapeQuantization,
) -> Result<QuantizationReport> {
    if output.exists() {
        bail!(
            "refusing to replace existing quantized checkpoint {}",
            output.display()
        );
    }
    let source_bytes = source
        .metadata()
        .with_context(|| format!("inspect source checkpoint {}", source.display()))?
        .len();
    let source_sha256 = sha256_file(source)?;
    let mapped = unsafe { candle_core::safetensors::MmapedSafetensors::new(source) }
        .with_context(|| format!("map source checkpoint {}", source.display()))?;
    let mut names = mapped
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    names.sort();
    let moe_model = names.iter().any(|name| name.contains(".moe.experts."));

    let mut tensors = Vec::with_capacity(names.len());
    let mut quantized_tensor_count = 0;
    for name in &names {
        let tensor = mapped
            .load(name, &Device::Cpu)
            .with_context(|| format!("load Hunyuan3D tensor {name}"))?;
        if tensor.rank() == 0 {
            bail!("Hunyuan3D tensor {name} is scalar-shaped and cannot be represented in GGUF");
        }
        let dtype = storage_dtype(name, tensor.dims(), tensor.dtype(), tier, moe_model)?;
        if dtype == tier.dtype() {
            quantized_tensor_count += 1;
        }
        let source = tensor
            .to_dtype(DType::F32)
            .with_context(|| format!("prepare Hunyuan3D tensor {name}"))?;
        let quantized = QTensor::quantize(&source, dtype)
            .with_context(|| format!("store Hunyuan3D tensor {name} as {dtype:?}"))?;
        tensors.push((name.clone(), quantized));
    }

    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create quantized checkpoint directory {}", parent.display()))?;
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .context("quantized checkpoint output has no UTF-8 filename")?;
    let temporary = parent.join(format!(".{file_name}.partial-{}", std::process::id()));
    if temporary.exists() {
        std::fs::remove_file(&temporary)
            .with_context(|| format!("remove stale conversion file {}", temporary.display()))?;
    }

    let metadata = [
        (
            "general.architecture",
            gguf_file::Value::String("hunyuan3d".to_string()),
        ),
        (
            "mold.quantization.policy",
            gguf_file::Value::String(POLICY_VERSION.to_string()),
        ),
        (
            "mold.quantization.tier",
            gguf_file::Value::String(tier.tag().to_string()),
        ),
        (
            "mold.source.model",
            gguf_file::Value::String(source_model.to_string()),
        ),
        (
            "mold.source.sha256",
            gguf_file::Value::String(source_sha256.clone()),
        ),
        ("mold.source.size", gguf_file::Value::U64(source_bytes)),
    ];
    let metadata_refs = metadata
        .iter()
        .map(|(name, value)| (*name, value))
        .collect::<Vec<_>>();
    let tensor_refs = tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect::<Vec<_>>();

    let result = (|| -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .with_context(|| format!("create conversion file {}", temporary.display()))?;
        let mut writer = BufWriter::new(file);
        gguf_file::write(&mut writer, &metadata_refs, &tensor_refs)
            .context("write Hunyuan3D GGUF")?;
        writer.flush().context("flush Hunyuan3D GGUF")?;
        writer.get_ref().sync_all().context("sync Hunyuan3D GGUF")?;
        drop(writer);
        std::fs::rename(&temporary, output).with_context(|| {
            format!(
                "publish quantized checkpoint {} as {}",
                temporary.display(),
                output.display()
            )
        })?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }

    let output_bytes = output
        .metadata()
        .with_context(|| format!("inspect quantized checkpoint {}", output.display()))?
        .len();
    Ok(QuantizationReport {
        source: source.to_path_buf(),
        output: output.to_path_buf(),
        source_sha256,
        source_bytes,
        output_bytes,
        tensor_count: tensors.len(),
        quantized_tensor_count,
        tier,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Shape, Tensor};

    use super::*;

    #[test]
    fn q8_policy_quantizes_only_transformer_matrix_weights() {
        assert_eq!(
            storage_dtype(
                "model.blocks.0.attn1.to_q.weight",
                &[2048, 2048],
                DType::F16,
                ShapeQuantization::Q8,
                false,
            )
            .unwrap(),
            GgmlDType::Q8_0,
        );
        for (name, dims) in [
            ("model.blocks.15.moe.gate.weight", &[8, 2048][..]),
            ("vae.decoder.weight", &[2048, 64][..]),
            (
                "conditioner.main_image_encoder.model.layer.weight",
                &[1024, 1024][..],
            ),
            ("model.blocks.0.norm1.weight", &[2048][..]),
            ("model.latent_in.weight", &[1024, 64][..]),
            ("model.blocks.0.img_mod.lin.weight", &[6144, 1024][..]),
            ("model.final_layer.linear.weight", &[64, 1024][..]),
        ] {
            assert_eq!(
                storage_dtype(name, dims, DType::F16, ShapeQuantization::Q8, false).unwrap(),
                GgmlDType::F16,
                "{name}",
            );
        }
    }

    #[test]
    fn every_lower_bit_tier_has_a_distinct_ggml_storage_type() {
        let tiers = [
            (ShapeQuantization::Q8, GgmlDType::Q8_0),
            (ShapeQuantization::Q6, GgmlDType::Q6K),
            (ShapeQuantization::Q5, GgmlDType::Q5K),
            (ShapeQuantization::Q4, GgmlDType::Q4K),
            (ShapeQuantization::Q3, GgmlDType::Q3K),
        ];
        for (tier, expected) in tiers {
            assert_eq!(
                storage_dtype(
                    "model.block.weight",
                    &[256, 256],
                    DType::F16,
                    tier,
                    false,
                )
                .unwrap(),
                expected,
            );
            assert_eq!(tier.to_string().parse::<ShapeQuantization>().unwrap(), tier);
        }
    }

    #[test]
    fn low_bit_moe_policy_quantizes_sparse_experts_only() {
        assert_eq!(
            storage_dtype(
                "model.blocks.15.moe.experts.3.net.0.proj.weight",
                &[8192, 2048],
                DType::F16,
                ShapeQuantization::Q4,
                true,
            )
            .unwrap(),
            GgmlDType::Q4K,
        );
        for name in [
            "model.blocks.15.attn.to_q.weight",
            "model.blocks.15.moe.shared_experts.net.0.proj.weight",
            "model.blocks.9.mlp.fc1.weight",
        ] {
            assert_eq!(
                storage_dtype(
                    name,
                    &[8192, 2048],
                    DType::F16,
                    ShapeQuantization::Q4,
                    true,
                )
                .unwrap(),
                GgmlDType::F16,
                "{name}",
            );
        }
    }

    #[test]
    fn q2_is_below_the_qualified_shape_quality_floor() {
        let error = "q2".parse::<ShapeQuantization>().unwrap_err();
        assert!(error.to_string().contains("expected q8, q6, q5, q4, or q3"));
    }

    #[test]
    fn conversion_is_source_preserving_atomic_and_self_describing() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("source.safetensors");
        let output = directory.path().join("derived").join("shape-q8.gguf");
        let matrix = Tensor::arange(0f32, 64f32, &Device::Cpu)?.reshape((2, 32))?;
        let router = Tensor::ones((2, 32), DType::F32, &Device::Cpu)?;
        let vae = Tensor::zeros((2, 32), DType::F32, &Device::Cpu)?;
        candle_core::safetensors::save(
            &HashMap::from([
                ("model.block.weight".to_string(), matrix),
                ("model.block.moe.gate.weight".to_string(), router),
                ("vae.block.weight".to_string(), vae),
            ]),
            &source,
        )?;
        let source_before = std::fs::read(&source)?;

        let report = quantize_checkpoint(
            &source,
            &output,
            "hunyuan3d-2.1:fp16",
            ShapeQuantization::Q8,
        )?;
        assert_eq!(std::fs::read(&source)?, source_before);
        assert_eq!(report.tensor_count, 3);
        assert_eq!(report.quantized_tensor_count, 1);
        assert!(report.output_bytes > 0);

        let mut file = File::open(&output)?;
        let content = gguf_file::Content::read(&mut file)?;
        assert_eq!(
            content.tensor_infos["model.block.weight"].ggml_dtype,
            GgmlDType::Q8_0
        );
        assert_eq!(
            content.tensor_infos["model.block.moe.gate.weight"].ggml_dtype,
            GgmlDType::F32
        );
        assert_eq!(
            content.tensor_infos["vae.block.weight"].ggml_dtype,
            GgmlDType::F32
        );
        assert!(matches!(
            content.metadata.get("mold.quantization.policy"),
            Some(gguf_file::Value::String(value)) if value == POLICY_VERSION
        ));
        assert!(matches!(
            content.metadata.get("mold.source.sha256"),
            Some(gguf_file::Value::String(value)) if value == &report.source_sha256
        ));

        let error = quantize_checkpoint(
            &source,
            &output,
            "hunyuan3d-2.1:fp16",
            ShapeQuantization::Q8,
        )
        .unwrap_err();
        assert!(error.to_string().contains("refusing to replace"));
        assert_eq!(
            std::fs::read_dir(output.parent().unwrap())?
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().contains("partial"))
                .count(),
            0,
        );
        Ok(())
    }

    #[test]
    fn scalar_shape_is_explicitly_refused_before_publish() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("source.safetensors");
        let output = directory.path().join("shape-q8.gguf");
        candle_core::safetensors::save(
            &HashMap::from([(
                "model.scalar".to_string(),
                Tensor::zeros(Shape::from(()), DType::F32, &Device::Cpu)?,
            )]),
            &source,
        )?;
        let error = quantize_checkpoint(
            &source,
            &output,
            "hunyuan3d-2.1:fp16",
            ShapeQuantization::Q8,
        )
        .unwrap_err();
        assert!(error.to_string().contains("scalar-shaped"));
        assert!(!output.exists());
        Ok(())
    }
}
