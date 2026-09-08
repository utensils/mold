//! Deterministic local quantization for Hunyuan3D shape checkpoints.
//!
//! Tencent publishes the 2.0 and 2.1 shape families as combined safetensors
//! files: `model.*` is the shape transformer, while `vae.*` and
//! `conditioner.*` are the geometry VAE and vision encoder. The first policy
//! deliberately quantizes transformer matrix weights only. Norms, biases,
//! MoE routing gates, the complete vision tower, and the complete VAE retain
//! their source float storage until separate numerical qualification exists.

use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use fs2::FileExt;
use sha2::{Digest, Sha256};

pub const POLICY_VERSION: &str = "hunyuan3d-shape-linear-v3";
pub const FP8_POLICY_VERSION: &str = "hunyuan3d-shape-fp8-e4m3-group32-v1";
pub const SUPPORTED_POLICY_VERSIONS: &[&str] = &[
    "hunyuan3d-shape-linear-v1",
    "hunyuan3d-shape-linear-v2",
    POLICY_VERSION,
];

static TEMPORARY_ID: AtomicU64 = AtomicU64::new(0);

/// Serialize converters targeting one path, then re-check the destination
/// while holding the cross-process lock. The lock file deliberately remains
/// on disk: `flock` state belongs to the open handle, so a crashed converter
/// cannot strand a permanent lock and a later process can safely reuse it.
struct DestinationClaim {
    _lock: File,
}

impl DestinationClaim {
    fn acquire(output: &Path) -> Result<Self> {
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent).with_context(|| {
            format!("create quantized checkpoint directory {}", parent.display())
        })?;
        let filename = output
            .file_name()
            .context("quantized checkpoint output has no filename")?
            .to_string_lossy();
        let lock_path = parent.join(format!(".{filename}.quantize.lock"));
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("open quantization lock {}", lock_path.display()))?;
        FileExt::lock_exclusive(&lock)
            .with_context(|| format!("lock quantization destination {}", output.display()))?;
        if output.exists() {
            bail!(
                "refusing to replace existing quantized checkpoint {}",
                output.display()
            );
        }
        Ok(Self { _lock: lock })
    }
}

fn temporary_path(output: &Path) -> Result<PathBuf> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let filename = output
        .file_name()
        .and_then(|name| name.to_str())
        .context("quantized checkpoint output has no UTF-8 filename")?;
    let id = TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
    Ok(parent.join(format!(".{filename}.partial-{}-{id}", std::process::id())))
}

/// Atomically add a fully synced temporary file at an absent destination.
/// A hard link is the portable same-filesystem no-replace primitive: unlike
/// `rename`, it fails with AlreadyExists on Unix instead of replacing bytes.
fn publish_no_replace(temporary: &Path, output: &Path) -> Result<()> {
    std::fs::hard_link(temporary, output).with_context(|| {
        format!(
            "publish quantized checkpoint {} as {} without replacing an existing file",
            temporary.display(),
            output.display()
        )
    })?;
    std::fs::remove_file(temporary)
        .with_context(|| format!("remove conversion link {}", temporary.display()))?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShapeQuantization {
    Fp8,
    Q8,
    Q6,
    Q5,
    Q4,
    Q3,
}

impl ShapeQuantization {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Q8 => "q8",
            Self::Q6 => "q6",
            Self::Q5 => "q5",
            Self::Q4 => "q4",
            Self::Q3 => "q3",
        }
    }

    fn ggml_dtype(self) -> Result<GgmlDType> {
        Ok(match self {
            Self::Fp8 => bail!("FP8 uses scaled safetensors rather than GGUF storage"),
            Self::Q8 => GgmlDType::Q8_0,
            Self::Q6 => GgmlDType::Q6K,
            Self::Q5 => GgmlDType::Q5K,
            Self::Q4 => GgmlDType::Q4K,
            Self::Q3 => GgmlDType::Q3K,
        })
    }

    pub fn file_extension(self) -> &'static str {
        match self {
            Self::Fp8 => "safetensors",
            Self::Q8 | Self::Q6 | Self::Q5 | Self::Q4 | Self::Q3 => "gguf",
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
            "fp8" | "f8" | "fp8_e4m3" => Ok(Self::Fp8),
            "q8" | "q8_0" => Ok(Self::Q8),
            "q6" | "q6_k" => Ok(Self::Q6),
            "q5" | "q5_k_m" => Ok(Self::Q5),
            "q4" | "q4_k_m" => Ok(Self::Q4),
            "q3" | "q3_k_m" => Ok(Self::Q3),
            other => bail!(
                "unsupported Hunyuan3D quantization {other}; expected fp8, q8, q6, q5, q4, or q3"
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

fn is_quantizable_matrix(name: &str, dims: &[usize]) -> bool {
    name.starts_with("model.")
        && name.ends_with(".weight")
        && dims.len() == 2
        && !is_router(name)
        && !is_precision_sensitive(name)
}

fn is_fp8_quantizable_matrix(name: &str, dims: &[usize], moe_model: bool) -> bool {
    is_quantizable_matrix(name, dims)
        && if moe_model {
            name.contains(".moe.experts.")
        } else {
            name.contains(".img_mlp.") || name.contains(".txt_mlp.")
        }
}

fn storage_dtype(
    name: &str,
    dims: &[usize],
    source_dtype: DType,
    tier: ShapeQuantization,
    moe_model: bool,
) -> Result<GgmlDType> {
    let quantized = tier.ggml_dtype()?;
    // Hunyuan3D 2.1's dense path is substantially more sensitive below Q8.
    // Its sparse experts hold most transformer parameters, so quantizing only
    // those retains useful compression while preserving the always-active
    // attention, dense MLP, shared expert, router, and modulation paths.
    let qualified_moe_layer =
        !moe_model || tier == ShapeQuantization::Q8 || name.contains(".moe.experts.");
    let eligible = is_quantizable_matrix(name, dims)
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

/// Validate and describe a completed conversion that was published before
/// its config registration committed. This makes `mold quantize` restartable
/// across a process or power failure without trusting an unrelated file that
/// happens to occupy the requested destination.
pub fn recover_existing_checkpoint(
    source: &Path,
    output: &Path,
    source_model: &str,
    tier: ShapeQuantization,
) -> Result<QuantizationReport> {
    let source_bytes = source
        .metadata()
        .with_context(|| format!("inspect source checkpoint {}", source.display()))?
        .len();
    let source_sha256 = sha256_file(source)?;
    let output_bytes = output
        .metadata()
        .with_context(|| format!("inspect retained checkpoint {}", output.display()))?
        .len();

    let mismatch = |field: &str| {
        anyhow::anyhow!(
            "retained checkpoint {} does not match the requested {field}",
            output.display()
        )
    };
    let (tensor_count, quantized_tensor_count) = if tier == ShapeQuantization::Fp8 {
        validate_fp8_checkpoint(output)?;
        let mapped = unsafe { candle_core::safetensors::MmapedSafetensors::new(output) }
            .with_context(|| format!("map retained FP8 checkpoint {}", output.display()))?;
        let read_bytes = |name: &str| -> Result<Vec<u8>> {
            mapped
                .load(name, &Device::Cpu)
                .with_context(|| format!("read retained checkpoint metadata {name}"))?
                .to_vec1::<u8>()
                .with_context(|| format!("decode retained checkpoint metadata {name}"))
        };
        let read_string = |name: &str| -> Result<String> {
            String::from_utf8(read_bytes(name)?)
                .with_context(|| format!("retained checkpoint metadata {name} is not UTF-8"))
        };
        anyhow::ensure!(
            read_string("mold_metadata.architecture")? == "hunyuan3d",
            mismatch("architecture")
        );
        anyhow::ensure!(
            read_string("mold_metadata.quantization_policy")? == FP8_POLICY_VERSION,
            mismatch("quantization policy")
        );
        anyhow::ensure!(
            read_string("mold_metadata.quantization_tier")? == tier.tag(),
            mismatch("quantization tier")
        );
        anyhow::ensure!(
            read_string("mold_metadata.source_model")? == source_model,
            mismatch("source model")
        );
        anyhow::ensure!(
            read_string("mold_metadata.source_sha256")? == source_sha256,
            mismatch("source digest")
        );
        let encoded_size = read_bytes("mold_metadata.source_size")?;
        anyhow::ensure!(encoded_size.len() == 8, mismatch("source size"));
        anyhow::ensure!(
            u64::from_le_bytes(encoded_size.try_into().expect("length checked")) == source_bytes,
            mismatch("source size")
        );
        let tensors = mapped.tensors();
        let quantized = tensors
            .iter()
            .filter(|(name, tensor)| {
                name.ends_with(".weight") && tensor.dtype() == DType::F8E4M3.into()
            })
            .count();
        let originals = tensors
            .iter()
            .filter(|(name, _)| {
                !name.starts_with("mold_metadata.")
                    && *name != "scaled_fp8"
                    && !name.ends_with(".scale_weight")
            })
            .count();
        (originals, quantized)
    } else {
        let mut file = File::open(output)
            .with_context(|| format!("open retained GGUF checkpoint {}", output.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("read retained GGUF checkpoint {}", output.display()))?;
        let metadata_string = |name: &str| match content.metadata.get(name) {
            Some(gguf_file::Value::String(value)) => Ok(value.as_str()),
            _ => Err(mismatch(name)),
        };
        anyhow::ensure!(
            metadata_string("general.architecture")? == "hunyuan3d",
            mismatch("architecture")
        );
        anyhow::ensure!(
            SUPPORTED_POLICY_VERSIONS.contains(&metadata_string("mold.quantization.policy")?),
            mismatch("quantization policy")
        );
        anyhow::ensure!(
            metadata_string("mold.quantization.tier")? == tier.tag(),
            mismatch("quantization tier")
        );
        anyhow::ensure!(
            metadata_string("mold.source.model")? == source_model,
            mismatch("source model")
        );
        anyhow::ensure!(
            metadata_string("mold.source.sha256")? == source_sha256,
            mismatch("source digest")
        );
        anyhow::ensure!(
            matches!(
                content.metadata.get("mold.source.size"),
                Some(gguf_file::Value::U64(size)) if *size == source_bytes
            ),
            mismatch("source size")
        );
        let target = tier.ggml_dtype()?;
        (
            content.tensor_infos.len(),
            content
                .tensor_infos
                .values()
                .filter(|info| info.ggml_dtype == target)
                .count(),
        )
    };

    Ok(QuantizationReport {
        source: source.to_path_buf(),
        output: output.to_path_buf(),
        source_sha256,
        source_bytes,
        output_bytes,
        tensor_count,
        quantized_tensor_count,
        tier,
    })
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
    if tier == ShapeQuantization::Fp8 {
        return quantize_fp8_checkpoint(source, output, source_model);
    }
    let _destination = DestinationClaim::acquire(output)?;
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
        if dtype == tier.ggml_dtype()? {
            quantized_tensor_count += 1;
        }
        let source = tensor
            .to_dtype(DType::F32)
            .with_context(|| format!("prepare Hunyuan3D tensor {name}"))?;
        let quantized = QTensor::quantize(&source, dtype)
            .with_context(|| format!("store Hunyuan3D tensor {name} as {dtype:?}"))?;
        tensors.push((name.clone(), quantized));
    }

    let temporary = temporary_path(output)?;

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
        publish_no_replace(&temporary, output)?;
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

/// Convert a combined checkpoint to mold's group-32 scaled E4M3 safetensors.
/// The largest magnitude in each consecutive 32-value weight group maps to
/// 448, and `<module>.scale_weight` restores each group's original magnitude
/// during the linear forward. Non-transformer components and sensitive
/// transformer paths retain their exact source dtype.
fn quantize_fp8_checkpoint(
    source: &Path,
    output: &Path,
    source_model: &str,
) -> Result<QuantizationReport> {
    let _destination = DestinationClaim::acquire(output)?;
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
    if names
        .iter()
        .any(|name| name == "scaled_fp8" || name.ends_with(".scale_weight"))
    {
        bail!("source checkpoint is already scaled FP8")
    }

    const FP8_E4M3_MAX: f32 = 448.0;
    let mut tensors = HashMap::with_capacity(names.len() * 2);
    let mut quantized_tensor_count = 0;
    for name in &names {
        let tensor = mapped
            .load(name, &Device::Cpu)
            .with_context(|| format!("load Hunyuan3D tensor {name}"))?;
        if !is_fp8_quantizable_matrix(name, tensor.dims(), moe_model) {
            tensors.insert(name.clone(), tensor);
            continue;
        }
        let source = tensor
            .to_dtype(DType::F32)
            .with_context(|| format!("prepare Hunyuan3D FP8 tensor {name}"))?;
        const GROUP_SIZE: usize = 32;
        let [output_dim, input_dim] = tensor.dims() else {
            unreachable!("FP8 eligibility requires a matrix")
        };
        if !input_dim.is_multiple_of(GROUP_SIZE) {
            tensors.insert(name.clone(), tensor);
            continue;
        }
        let grouped = source.reshape((*output_dim, *input_dim / GROUP_SIZE, GROUP_SIZE))?;
        let peak = grouped.abs()?.max_keepdim(2)?;
        if !peak.max_all()?.to_scalar::<f32>()?.is_finite() {
            bail!("Hunyuan3D tensor {name} has a non-finite magnitude")
        }
        let scale = peak
            .affine(1.0 / f64::from(FP8_E4M3_MAX), 0.0)?
            .clamp(1e-12f32, f32::MAX)?;
        let fp8 = grouped
            .broadcast_div(&scale)?
            .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)?
            .to_dtype(DType::F8E4M3)?
            .reshape((*output_dim, *input_dim))?;
        tensors.insert(name.clone(), fp8);
        tensors.insert(
            format!("{}.scale_weight", name.trim_end_matches(".weight")),
            scale.squeeze(2)?,
        );
        quantized_tensor_count += 1;
    }
    tensors.insert(
        "scaled_fp8".to_string(),
        Tensor::zeros(2, DType::F8E4M3, &Device::Cpu)?,
    );

    let temporary = temporary_path(output)?;
    for (name, value) in [
        ("mold_metadata.architecture", "hunyuan3d"),
        ("mold_metadata.quantization_policy", FP8_POLICY_VERSION),
        ("mold_metadata.quantization_tier", "fp8"),
        ("mold_metadata.source_model", source_model),
        ("mold_metadata.source_sha256", source_sha256.as_str()),
    ] {
        tensors.insert(
            name.to_string(),
            Tensor::from_vec(value.as_bytes().to_vec(), value.len(), &Device::Cpu)?,
        );
    }
    tensors.insert(
        "mold_metadata.source_size".to_string(),
        Tensor::from_vec(source_bytes.to_le_bytes().to_vec(), 8, &Device::Cpu)?,
    );
    let result = (|| -> Result<()> {
        candle_core::safetensors::save(&tensors, &temporary)
            .context("write Hunyuan3D scaled FP8 safetensors")?;
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(&temporary)?
            .sync_all()
            .context("sync Hunyuan3D FP8 safetensors")?;
        publish_no_replace(&temporary, output)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }

    let output_bytes = output.metadata()?.len();
    Ok(QuantizationReport {
        source: source.to_path_buf(),
        output: output.to_path_buf(),
        source_sha256,
        source_bytes,
        output_bytes,
        tensor_count: names.len(),
        quantized_tensor_count,
        tier: ShapeQuantization::Fp8,
    })
}

pub(super) fn validate_fp8_checkpoint(path: &Path) -> Result<()> {
    let header = crate::weight_loader::read_safetensors_header(path)
        .with_context(|| format!("read Hunyuan3D FP8 header {}", path.display()))?;
    let dtype = |name: &str| {
        header
            .get(name)
            .and_then(|value| value.get("dtype"))
            .and_then(serde_json::Value::as_str)
    };
    let shape = |name: &str| {
        header
            .get(name)
            .and_then(|value| value.get("shape"))
            .and_then(serde_json::Value::as_array)
    };

    if dtype("scaled_fp8") != Some("F8_E4M3")
        || shape("scaled_fp8").is_none_or(|dims| dims.len() != 1 || dims[0].as_u64() != Some(2))
    {
        bail!("Hunyuan3D FP8 checkpoint is missing its scaled_fp8 marker")
    }
    if dtype("mold_metadata.quantization_policy") != Some("U8") {
        bail!("Hunyuan3D FP8 checkpoint is missing its mold quantization policy")
    }

    let mut fp8_weights = 0usize;
    for (name, value) in &header {
        let tensor_dtype = value.get("dtype").and_then(serde_json::Value::as_str);
        if tensor_dtype == Some("F8_E4M3") && name != "scaled_fp8" {
            let Some(module) = name.strip_suffix(".weight") else {
                bail!("Hunyuan3D FP8 tensor {name} is not a matrix weight")
            };
            let scale = format!("{module}.scale_weight");
            let weight_shape = shape(name).context("Hunyuan3D FP8 weight has no shape")?;
            let scale_shape = shape(&scale);
            let valid_scale = weight_shape.len() == 2
                && weight_shape[1]
                    .as_u64()
                    .is_some_and(|input| input.is_multiple_of(32))
                && scale_shape.is_some_and(|dims| {
                    dims.len() == 2
                        && dims[0] == weight_shape[0]
                        && dims[1].as_u64() == weight_shape[1].as_u64().map(|input| input / 32)
                });
            if dtype(&scale) != Some("F32") || !valid_scale {
                bail!("Hunyuan3D FP8 weight {name} has no group-32 F32 scale_weight")
            }
            fp8_weights += 1;
        }
        if let Some(module) = name.strip_suffix(".scale_weight") {
            let weight = format!("{module}.weight");
            if dtype(&weight) != Some("F8_E4M3") {
                bail!("Hunyuan3D FP8 scale {name} has no FP8 weight")
            }
        }
    }
    if fp8_weights == 0 {
        bail!("Hunyuan3D FP8 checkpoint contains no scaled FP8 matrix weights")
    }

    let mapped = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
        .with_context(|| format!("map Hunyuan3D FP8 checkpoint {}", path.display()))?;
    let policy = mapped
        .load("mold_metadata.quantization_policy", &Device::Cpu)?
        .to_vec1::<u8>()?;
    if policy != FP8_POLICY_VERSION.as_bytes() {
        bail!(
            "unsupported Hunyuan3D FP8 quantization policy {}; expected {FP8_POLICY_VERSION}",
            String::from_utf8_lossy(&policy)
        )
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Shape, Tensor};

    use super::*;

    #[test]
    fn publication_never_replaces_an_existing_destination() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let temporary = directory.path().join("complete.partial");
        let output = directory.path().join("shape-q8.gguf");
        std::fs::write(&temporary, b"new conversion")?;
        std::fs::write(&output, b"registered conversion")?;

        let error = publish_no_replace(&temporary, &output).unwrap_err();
        assert!(error.to_string().contains("without replacing"));
        assert_eq!(std::fs::read(&output)?, b"registered conversion");
        assert_eq!(std::fs::read(&temporary)?, b"new conversion");
        Ok(())
    }

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
                storage_dtype("model.block.weight", &[256, 256], DType::F16, tier, false,).unwrap(),
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
                storage_dtype(name, &[8192, 2048], DType::F16, ShapeQuantization::Q4, true,)
                    .unwrap(),
                GgmlDType::F16,
                "{name}",
            );
        }
    }

    #[test]
    fn q2_is_below_the_qualified_shape_quality_floor() {
        let error = "q2".parse::<ShapeQuantization>().unwrap_err();
        assert!(error
            .to_string()
            .contains("expected fp8, q8, q6, q5, q4, or q3"));
    }

    #[test]
    fn fp8_conversion_is_scaled_source_preserving_and_self_describing() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = directory.path().join("source.safetensors");
        let output = directory
            .path()
            .join("derived")
            .join("shape-fp8.safetensors");
        let matrix = (Tensor::arange(0f32, 64f32, &Device::Cpu)? - 31.5)?.reshape((2, 32))?;
        let sensitive = Tensor::ones((2, 32), DType::F16, &Device::Cpu)?;
        candle_core::safetensors::save(
            &HashMap::from([
                (
                    "model.double_blocks.0.img_mlp.0.weight".to_string(),
                    matrix.clone(),
                ),
                ("model.latent_in.weight".to_string(), sensitive),
            ]),
            &source,
        )?;

        let report =
            quantize_checkpoint(&source, &output, "hunyuan3d:test", ShapeQuantization::Fp8)?;
        assert!(source.is_file());
        assert!(output.is_file());
        assert_eq!(report.quantized_tensor_count, 1);
        let stored = candle_core::safetensors::load(&output, &Device::Cpu)?;
        assert_eq!(
            stored["model.double_blocks.0.img_mlp.0.weight"].dtype(),
            DType::F8E4M3
        );
        assert_eq!(stored["model.latent_in.weight"].dtype(), DType::F16);
        assert_eq!(stored["scaled_fp8"].dims(), &[2]);
        let scale = &stored["model.double_blocks.0.img_mlp.0.scale_weight"];
        assert_eq!(scale.dims(), &[2, 1]);
        let reconstructed = stored["model.double_blocks.0.img_mlp.0.weight"]
            .to_dtype(DType::F32)?
            .reshape((2, 1, 32))?
            .broadcast_mul(&scale.unsqueeze(2)?)?
            .reshape((2, 32))?;
        let max_error = (reconstructed - matrix.clone())?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(max_error <= 1.0, "FP8 reconstruction error {max_error}");
        let policy = stored["mold_metadata.quantization_policy"].to_vec1::<u8>()?;
        assert_eq!(std::str::from_utf8(&policy)?, FP8_POLICY_VERSION);
        validate_fp8_checkpoint(&output)?;
        let recovered = recover_existing_checkpoint(
            &source,
            &output,
            "hunyuan3d:test",
            ShapeQuantization::Fp8,
        )?;
        assert_eq!(recovered.source_sha256, report.source_sha256);
        assert_eq!(recovered.quantized_tensor_count, 1);

        let invalid = directory.path().join("missing-scale.safetensors");
        candle_core::safetensors::save(
            &HashMap::from([
                (
                    "model.double_blocks.0.img_mlp.0.weight".to_string(),
                    matrix.to_dtype(DType::F8E4M3)?,
                ),
                (
                    "scaled_fp8".to_string(),
                    Tensor::zeros(2, DType::F8E4M3, &Device::Cpu)?,
                ),
                (
                    "mold_metadata.quantization_policy".to_string(),
                    Tensor::from_vec(
                        FP8_POLICY_VERSION.as_bytes().to_vec(),
                        FP8_POLICY_VERSION.len(),
                        &Device::Cpu,
                    )?,
                ),
            ]),
            &invalid,
        )?;
        assert!(validate_fp8_checkpoint(&invalid)
            .unwrap_err()
            .to_string()
            .contains("has no group-32 F32 scale_weight"));
        Ok(())
    }

    #[test]
    fn fp8_policy_keeps_attention_and_fused_blocks_in_source_precision() {
        let dims = [4096, 1024];
        assert!(is_fp8_quantizable_matrix(
            "model.double_blocks.0.img_mlp.0.weight",
            &dims,
            false
        ));
        for name in [
            "model.double_blocks.0.img_attn.qkv.weight",
            "model.single_blocks.0.linear1.weight",
            "model.double_blocks.0.img_mod.lin.weight",
        ] {
            assert!(!is_fp8_quantizable_matrix(name, &dims, false), "{name}");
        }
        assert!(is_fp8_quantizable_matrix(
            "model.blocks.0.moe.experts.0.fc1.weight",
            &dims,
            true
        ));
        assert!(!is_fp8_quantizable_matrix(
            "model.blocks.0.attn.to_q.weight",
            &dims,
            true
        ));
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

        // Publication and config registration are separate durable commits.
        // A retry after the first succeeded and the second failed validates
        // and reuses the completed bytes instead of starting hours of work
        // over or leaving the checkpoint impossible to register.
        let recovered = recover_existing_checkpoint(
            &source,
            &output,
            "hunyuan3d-2.1:fp16",
            ShapeQuantization::Q8,
        )?;
        assert_eq!(recovered, report);
        let mismatch =
            recover_existing_checkpoint(&source, &output, "hunyuan3d:fp16", ShapeQuantization::Q8)
                .unwrap_err()
                .to_string();
        assert!(mismatch.contains("source model"), "{mismatch}");

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
