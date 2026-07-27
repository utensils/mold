//! Header-only, path-independent artifact format facts used by scheduler
//! execution-equivalence planning.
//!
//! These probes deliberately inspect the artifact format rather than model IDs
//! or filenames. Catalog IDs are opaque and aliases are not execution facts.

use candle_core::quantized::{gguf_file, GgmlDType};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MAX_SAFETENSORS_HEADER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub enum TensorDType {
    Bool,
    U8,
    I8,
    F8E5M2,
    F8E4M3,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub enum GgufTensorFormat {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum SafetensorsEncoding {
    Standard,
    Nvfp4,
    ConvRotW4A4,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum ArtifactStorageFormat {
    Safetensors {
        encoding: SafetensorsEncoding,
        tensor_dtypes: Vec<TensorDType>,
    },
    Gguf {
        tensor_formats: Vec<GgufTensorFormat>,
    },
    Json,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactProbeFailure {
    Io,
    UnsupportedContainer,
    InvalidHeader,
    UnsupportedTensorDType,
    UnsupportedGgufTensorFormat,
}

pub fn probe(path: &Path) -> Result<ArtifactStorageFormat, ArtifactProbeFailure> {
    let mut file = File::open(path).map_err(|_| ArtifactProbeFailure::Io)?;
    let mut magic = [0_u8; 8];
    let read = file
        .read(&mut magic)
        .map_err(|_| ArtifactProbeFailure::Io)?;
    if read >= 4 && (&magic[..4] == b"GGUF" || &magic[..4] == b"FUGG") {
        return probe_gguf(path);
    }
    if magic[..read]
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
        .is_some_and(|byte| matches!(byte, b'{' | b'['))
    {
        serde_json::from_reader::<_, Value>(
            File::open(path).map_err(|_| ArtifactProbeFailure::Io)?,
        )
        .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
        return Ok(ArtifactStorageFormat::Json);
    }
    probe_safetensors(path)
}

fn probe_gguf(path: &Path) -> Result<ArtifactStorageFormat, ArtifactProbeFailure> {
    let mut file = File::open(path).map_err(|_| ArtifactProbeFailure::Io)?;
    let content = gguf_file::Content::read(&mut file).map_err(|error| {
        if error.to_string().contains("unknown dtype for tensor") {
            ArtifactProbeFailure::UnsupportedGgufTensorFormat
        } else {
            ArtifactProbeFailure::InvalidHeader
        }
    })?;
    let mut formats = BTreeSet::new();
    for tensor in content.tensor_infos.values() {
        formats.insert(match tensor.ggml_dtype {
            GgmlDType::F32 => GgufTensorFormat::F32,
            GgmlDType::F16 => GgufTensorFormat::F16,
            GgmlDType::BF16 => GgufTensorFormat::Bf16,
            GgmlDType::Q4_0 => GgufTensorFormat::Q4_0,
            GgmlDType::Q4_1 => GgufTensorFormat::Q4_1,
            GgmlDType::Q5_0 => GgufTensorFormat::Q5_0,
            GgmlDType::Q5_1 => GgufTensorFormat::Q5_1,
            GgmlDType::Q8_0 => GgufTensorFormat::Q8_0,
            GgmlDType::Q8_1 => GgufTensorFormat::Q8_1,
            GgmlDType::Q2K => GgufTensorFormat::Q2K,
            GgmlDType::Q3K => GgufTensorFormat::Q3K,
            GgmlDType::Q4K => GgufTensorFormat::Q4K,
            GgmlDType::Q5K => GgufTensorFormat::Q5K,
            GgmlDType::Q6K => GgufTensorFormat::Q6K,
            GgmlDType::Q8K => GgufTensorFormat::Q8K,
        });
    }
    Ok(ArtifactStorageFormat::Gguf {
        tensor_formats: formats.into_iter().collect(),
    })
}

fn probe_safetensors(path: &Path) -> Result<ArtifactStorageFormat, ArtifactProbeFailure> {
    let mut file = File::open(path).map_err(|_| ArtifactProbeFailure::Io)?;
    let file_len = file.metadata().map_err(|_| ArtifactProbeFailure::Io)?.len();
    let mut length = [0_u8; 8];
    file.read_exact(&mut length)
        .map_err(|_| ArtifactProbeFailure::UnsupportedContainer)?;
    let header_len = u64::from_le_bytes(length);
    if header_len == 0
        || header_len > MAX_SAFETENSORS_HEADER_BYTES
        || header_len > file_len.saturating_sub(8)
    {
        return Err(ArtifactProbeFailure::InvalidHeader);
    }
    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
    let header: serde_json::Map<String, Value> =
        serde_json::from_slice(&header).map_err(|_| ArtifactProbeFailure::InvalidHeader)?;

    let keys = header
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .collect::<BTreeSet<_>>();
    let has_nvfp4 = keys.iter().any(|key| {
        key.ends_with(".weight_scale_2")
            || key.ends_with(".weight.nvfp4_packed")
            || key.ends_with(".weight.nvfp4_block_scales")
            || key.ends_with(".weight.nvfp4_tensor_scale")
            || key.ends_with(".nvfp4_packed")
            || key.ends_with(".nvfp4_block_scales")
            || key.ends_with(".nvfp4_tensor_scale")
            || key.ends_with(".input_scale")
            || key.ends_with(".comfy_quant")
    });
    let has_convrot = crate::ltx2::convrot::checkpoint_is_convrot_w4a4(path);
    let encoding = if has_nvfp4 {
        SafetensorsEncoding::Nvfp4
    } else if has_convrot {
        SafetensorsEncoding::ConvRotW4A4
    } else {
        SafetensorsEncoding::Standard
    };

    let mut dtypes = BTreeSet::new();
    for (key, value) in &header {
        if key == "__metadata__" {
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or(ArtifactProbeFailure::InvalidHeader)?;
        dtypes.insert(parse_safetensors_dtype(dtype)?);
    }
    Ok(ArtifactStorageFormat::Safetensors {
        encoding,
        tensor_dtypes: dtypes.into_iter().collect(),
    })
}

fn parse_safetensors_dtype(value: &str) -> Result<TensorDType, ArtifactProbeFailure> {
    Ok(match value {
        "BOOL" => TensorDType::Bool,
        "U8" => TensorDType::U8,
        "I8" => TensorDType::I8,
        "F8_E5M2" => TensorDType::F8E5M2,
        "F8_E4M3" => TensorDType::F8E4M3,
        "I16" => TensorDType::I16,
        "U16" => TensorDType::U16,
        "F16" => TensorDType::F16,
        "BF16" => TensorDType::Bf16,
        "I32" => TensorDType::I32,
        "U32" => TensorDType::U32,
        "F32" => TensorDType::F32,
        "F64" => TensorDType::F64,
        "I64" => TensorDType::I64,
        "U64" => TensorDType::U64,
        _ => return Err(ArtifactProbeFailure::UnsupportedTensorDType),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QTensor;
    use candle_core::{Device, Tensor};
    use std::io::Write;

    fn write_safetensors(path: &Path, header: Value, data: &[u8]) {
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.write_all(data).unwrap();
    }

    #[test]
    fn opaque_fp8_safetensors_is_resolved_from_header_not_name() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("cv-3143864.asset");
        write_safetensors(
            &path,
            serde_json::json!({
                "model.diffusion_model.img_in.weight": {
                    "dtype": "F8_E4M3",
                    "shape": [1],
                    "data_offsets": [0, 1]
                }
            }),
            &[0],
        );
        assert_eq!(
            probe(&path),
            Ok(ArtifactStorageFormat::Safetensors {
                encoding: SafetensorsEncoding::Standard,
                tensor_dtypes: vec![TensorDType::F8E4M3],
            })
        );
    }

    #[test]
    fn runtime_nvfp4_markers_win_over_container_dtype() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("hf-opaque.asset");
        write_safetensors(
            &path,
            serde_json::json!({
                "layer.weight.nvfp4_packed": {
                    "dtype": "U8",
                    "shape": [1],
                    "data_offsets": [0, 1]
                },
                "layer.weight.nvfp4_block_scales": {
                    "dtype": "F8_E4M3",
                    "shape": [1],
                    "data_offsets": [1, 2]
                },
                "layer.weight.nvfp4_tensor_scale": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [2, 6]
                }
            }),
            &[0; 6],
        );
        assert!(matches!(
            probe(&path),
            Ok(ArtifactStorageFormat::Safetensors {
                encoding: SafetensorsEncoding::Nvfp4,
                ..
            })
        ));
    }

    #[test]
    fn existing_convrot_probe_is_the_w4a4_authority() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("cv-opaque.asset");
        write_safetensors(
            &path,
            serde_json::json!({
                "transformer_blocks.0.attn.to_q.weight": {
                    "dtype": "I8",
                    "shape": [1, 1],
                    "data_offsets": [0, 1]
                },
                "transformer_blocks.0.attn.to_q.weight_scale": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [1, 5]
                }
            }),
            &[0; 5],
        );
        assert!(crate::ltx2::convrot::checkpoint_is_convrot_w4a4(&path));
        assert!(matches!(
            probe(&path),
            Ok(ArtifactStorageFormat::Safetensors {
                encoding: SafetensorsEncoding::ConvRotW4A4,
                ..
            })
        ));
    }

    #[test]
    fn gguf_probe_reports_exact_supported_quantization() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("opaque");
        let source = Tensor::zeros((32,), candle_core::DType::F32, &Device::Cpu).unwrap();
        let quantized = QTensor::quantize(&source, GgmlDType::Q4_0).unwrap();
        let mut file = File::create(&path).unwrap();
        gguf_file::write(&mut file, &[], &[("weight", &quantized)]).unwrap();
        drop(file);
        assert_eq!(
            probe(&path),
            Ok(ArtifactStorageFormat::Gguf {
                tensor_formats: vec![GgufTensorFormat::Q4_0],
            })
        );
    }

    #[test]
    fn json_identity_is_content_probed_without_extension_authority() {
        let root = tempfile::tempdir().unwrap();
        let json_without_extension = root.path().join("tokenizer");
        std::fs::write(&json_without_extension, b" {\"kind\":\"tokenizer\"}").unwrap();
        assert_eq!(
            probe(&json_without_extension),
            Ok(ArtifactStorageFormat::Json)
        );

        let non_json_with_extension = root.path().join("weights.json");
        std::fs::write(&non_json_with_extension, b"not json").unwrap();
        assert_ne!(
            probe(&non_json_with_extension),
            Ok(ArtifactStorageFormat::Json)
        );
    }
}
