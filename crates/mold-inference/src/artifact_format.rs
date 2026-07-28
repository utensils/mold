//! Header-only, path-independent artifact format facts used by scheduler
//! execution-equivalence planning.
//!
//! These probes deliberately inspect the artifact format rather than model IDs
//! or filenames. Catalog IDs are opaque and aliases are not execution facts.

#[cfg(test)]
use candle_core::quantized::{gguf_file, GgmlDType};
use serde::de::{Deserializer as _, Error as _, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MAX_SAFETENSORS_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_SAFETENSORS_TENSORS: usize = 262_144;
const MAX_SAFETENSORS_KEY_BYTES: usize = 16 * 1024;
const MAX_CONVROT_MARKERS: usize = 131_072;
const MAX_JSON_PROBE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_GGUF_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_GGUF_TENSORS: u64 = 262_144;
const MAX_GGUF_METADATA_ITEMS: u64 = 262_144;
const MAX_GGUF_ARRAY_ITEMS: u64 = 1_048_576;
const MAX_GGUF_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_GGUF_VALUE_DEPTH: usize = 8;
const UNSUPPORTED_DTYPE_MARKER: &str = "mold-unsupported-safetensors-dtype";

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
        let file = File::open(path).map_err(|_| ArtifactProbeFailure::Io)?;
        if file.metadata().map_err(|_| ArtifactProbeFailure::Io)?.len() > MAX_JSON_PROBE_BYTES {
            return Err(ArtifactProbeFailure::InvalidHeader);
        }
        let mut deserializer = serde_json::Deserializer::from_reader(file);
        IgnoredAny::deserialize(&mut deserializer)
            .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
        deserializer
            .end()
            .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
        return Ok(ArtifactStorageFormat::Json);
    }
    probe_safetensors(path)
}

fn probe_gguf(path: &Path) -> Result<ArtifactStorageFormat, ArtifactProbeFailure> {
    let file = File::open(path).map_err(|_| ArtifactProbeFailure::Io)?;
    let mut reader = file.take(MAX_GGUF_HEADER_BYTES);
    let magic = read_exact_array::<4>(&mut reader)?;
    if &magic != b"GGUF" && &magic != b"FUGG" {
        return Err(ArtifactProbeFailure::InvalidHeader);
    }
    let version = read_u32(&mut reader)?;
    if !(1..=3).contains(&version) {
        return Err(ArtifactProbeFailure::InvalidHeader);
    }
    let tensor_count = read_versioned_count(&mut reader, version)?;
    let metadata_count = read_versioned_count(&mut reader, version)?;
    if tensor_count > MAX_GGUF_TENSORS || metadata_count > MAX_GGUF_METADATA_ITEMS {
        return Err(ArtifactProbeFailure::InvalidHeader);
    }
    for _ in 0..metadata_count {
        skip_gguf_string(&mut reader, version)?;
        let value_type = read_u32(&mut reader)?;
        skip_gguf_value(&mut reader, version, value_type, 0)?;
    }
    let mut formats = BTreeSet::new();
    for _ in 0..tensor_count {
        skip_gguf_string(&mut reader, version)?;
        let dimensions = read_u32(&mut reader)?;
        if dimensions > 8 {
            return Err(ArtifactProbeFailure::InvalidHeader);
        }
        skip_exact(
            &mut reader,
            u64::from(dimensions) * if version == 1 { 4 } else { 8 },
        )?;
        formats.insert(match read_u32(&mut reader)? {
            0 => GgufTensorFormat::F32,
            1 => GgufTensorFormat::F16,
            2 => GgufTensorFormat::Q4_0,
            3 => GgufTensorFormat::Q4_1,
            6 => GgufTensorFormat::Q5_0,
            7 => GgufTensorFormat::Q5_1,
            8 => GgufTensorFormat::Q8_0,
            9 => GgufTensorFormat::Q8_1,
            10 => GgufTensorFormat::Q2K,
            11 => GgufTensorFormat::Q3K,
            12 => GgufTensorFormat::Q4K,
            13 => GgufTensorFormat::Q5K,
            14 => GgufTensorFormat::Q6K,
            15 => GgufTensorFormat::Q8K,
            30 => GgufTensorFormat::Bf16,
            _ => return Err(ArtifactProbeFailure::UnsupportedGgufTensorFormat),
        });
        skip_exact(&mut reader, 8)?;
    }
    Ok(ArtifactStorageFormat::Gguf {
        tensor_formats: formats.into_iter().collect(),
    })
}

fn read_exact_array<const N: usize>(
    reader: &mut impl Read,
) -> Result<[u8; N], ArtifactProbeFailure> {
    let mut bytes = [0_u8; N];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
    Ok(bytes)
}

fn read_u32(reader: &mut impl Read) -> Result<u32, ArtifactProbeFailure> {
    Ok(u32::from_le_bytes(read_exact_array(reader)?))
}

fn read_u64(reader: &mut impl Read) -> Result<u64, ArtifactProbeFailure> {
    Ok(u64::from_le_bytes(read_exact_array(reader)?))
}

fn read_versioned_count(reader: &mut impl Read, version: u32) -> Result<u64, ArtifactProbeFailure> {
    if version == 1 {
        read_u32(reader).map(u64::from)
    } else {
        read_u64(reader)
    }
}

fn skip_exact(reader: &mut impl Read, mut bytes: u64) -> Result<(), ArtifactProbeFailure> {
    let mut buffer = [0_u8; 8192];
    while bytes > 0 {
        let chunk = usize::try_from(bytes.min(buffer.len() as u64))
            .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
        reader
            .read_exact(&mut buffer[..chunk])
            .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
        bytes -= chunk as u64;
    }
    Ok(())
}

fn skip_gguf_string(reader: &mut impl Read, version: u32) -> Result<(), ArtifactProbeFailure> {
    let bytes = read_versioned_count(reader, version)?;
    if bytes > MAX_GGUF_STRING_BYTES {
        return Err(ArtifactProbeFailure::InvalidHeader);
    }
    skip_exact(reader, bytes)
}

fn skip_gguf_value(
    reader: &mut impl Read,
    version: u32,
    value_type: u32,
    depth: usize,
) -> Result<(), ArtifactProbeFailure> {
    if depth > MAX_GGUF_VALUE_DEPTH {
        return Err(ArtifactProbeFailure::InvalidHeader);
    }
    match value_type {
        0 | 1 | 7 => skip_exact(reader, 1),
        2 | 3 => skip_exact(reader, 2),
        4..=6 => skip_exact(reader, 4),
        8 => skip_gguf_string(reader, version),
        9 => {
            let nested_type = read_u32(reader)?;
            let items = read_versioned_count(reader, version)?;
            if items > MAX_GGUF_ARRAY_ITEMS {
                return Err(ArtifactProbeFailure::InvalidHeader);
            }
            for _ in 0..items {
                skip_gguf_value(reader, version, nested_type, depth + 1)?;
            }
            Ok(())
        }
        10..=12 => skip_exact(reader, 8),
        _ => Err(ArtifactProbeFailure::InvalidHeader),
    }
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
    // Stream exactly the declared header rather than allocating or mapping it.
    // A concurrently truncated mmap can fault the process when a mapped page
    // is touched; ordinary bounded reads instead fail closed as an invalid
    // header.
    let header = file.take(header_len);
    let mut deserializer = serde_json::Deserializer::from_reader(header);
    let facts = deserializer
        .deserialize_map(SafetensorsHeaderVisitor)
        .map_err(|error| {
            if error.to_string().contains(UNSUPPORTED_DTYPE_MARKER) {
                ArtifactProbeFailure::UnsupportedTensorDType
            } else {
                ArtifactProbeFailure::InvalidHeader
            }
        })?;
    deserializer
        .end()
        .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
    let has_convrot = facts
        .convrot_weights
        .iter()
        .any(|base| facts.convrot_scales.contains(base));
    let encoding = if has_convrot {
        SafetensorsEncoding::ConvRotW4A4
    } else if facts.has_nvfp4 {
        SafetensorsEncoding::Nvfp4
    } else {
        SafetensorsEncoding::Standard
    };
    Ok(ArtifactStorageFormat::Safetensors {
        encoding,
        tensor_dtypes: facts.dtypes.into_iter().collect(),
    })
}

#[derive(Deserialize)]
struct TensorHeader {
    dtype: String,
}

#[derive(Default)]
struct SafetensorsHeaderFacts {
    dtypes: BTreeSet<TensorDType>,
    has_nvfp4: bool,
    convrot_weights: BTreeSet<[u8; 32]>,
    convrot_scales: BTreeSet<[u8; 32]>,
}

struct SafetensorsHeaderVisitor;

impl<'de> Visitor<'de> for SafetensorsHeaderVisitor {
    type Value = SafetensorsHeaderFacts;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a bounded safetensors JSON header map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut facts = SafetensorsHeaderFacts::default();
        let mut tensors = 0usize;
        while let Some(key) = map.next_key::<String>()? {
            if key.len() > MAX_SAFETENSORS_KEY_BYTES {
                return Err(A::Error::custom("safetensors tensor key is too large"));
            }
            if key == "__metadata__" {
                map.next_value::<IgnoredAny>()?;
                continue;
            }
            tensors = tensors.saturating_add(1);
            if tensors > MAX_SAFETENSORS_TENSORS {
                return Err(A::Error::custom(
                    "safetensors tensor count exceeds probe bound",
                ));
            }
            let header = map.next_value::<TensorHeader>()?;
            let dtype = parse_safetensors_dtype(&header.dtype)
                .map_err(|_| A::Error::custom(UNSUPPORTED_DTYPE_MARKER))?;
            facts.dtypes.insert(dtype);
            facts.has_nvfp4 |= is_explicit_nvfp4_marker(&key);
            if dtype == TensorDType::I8 {
                if let Some(base) = key.strip_suffix(".weight") {
                    insert_bounded_marker::<A::Error>(&mut facts.convrot_weights, base)?;
                }
            }
            if let Some(base) = key.strip_suffix(".weight_scale") {
                insert_bounded_marker::<A::Error>(&mut facts.convrot_scales, base)?;
            }
        }
        Ok(facts)
    }
}

fn insert_bounded_marker<E>(markers: &mut BTreeSet<[u8; 32]>, base: &str) -> Result<(), E>
where
    E: serde::de::Error,
{
    if markers.len() >= MAX_CONVROT_MARKERS {
        return Err(E::custom(
            "safetensors ConvRot marker count exceeds probe bound",
        ));
    }
    use sha2::{Digest, Sha256};
    markers.insert(Sha256::digest(base.as_bytes()).into());
    Ok(())
}

fn is_explicit_nvfp4_marker(key: &str) -> bool {
    key.ends_with(".weight_scale_2") || key.ends_with(".comfy_quant")
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
                "layer.weight_scale": {
                    "dtype": "F8_E4M3",
                    "shape": [1],
                    "data_offsets": [0, 1]
                },
                "layer.weight_scale_2": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [1, 5]
                }
            }),
            &[0; 5],
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
                },
                "transformer_blocks.0.attn.to_q.input_scale": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [5, 9]
                }
            }),
            &[0; 9],
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
    fn fp8_input_scale_is_not_misclassified_as_nvfp4() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("opaque-fp8.asset");
        write_safetensors(
            &path,
            serde_json::json!({
                "transformer_blocks.0.attn.to_q.weight": {
                    "dtype": "F8_E4M3",
                    "shape": [1],
                    "data_offsets": [0, 1]
                },
                "transformer_blocks.0.attn.to_q.input_scale": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [1, 5]
                }
            }),
            &[0; 5],
        );
        assert_eq!(
            probe(&path),
            Ok(ArtifactStorageFormat::Safetensors {
                encoding: SafetensorsEncoding::Standard,
                tensor_dtypes: vec![TensorDType::F8E4M3, TensorDType::F32],
            })
        );
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
    fn gguf_declared_counts_are_bounded_before_container_allocation() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("hostile.gguf");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3_u32.to_le_bytes()).unwrap();
        file.write_all(&(MAX_GGUF_TENSORS + 1).to_le_bytes())
            .unwrap();
        file.write_all(&0_u64.to_le_bytes()).unwrap();
        drop(file);

        assert_eq!(probe(&path), Err(ArtifactProbeFailure::InvalidHeader));
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

    #[test]
    fn oversized_safetensors_header_is_rejected_before_reading_or_allocating_it() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("oversized");
        let mut file = File::create(&path).unwrap();
        file.write_all(&(MAX_SAFETENSORS_HEADER_BYTES + 1).to_le_bytes())
            .unwrap();
        file.set_len(8 + MAX_SAFETENSORS_HEADER_BYTES + 1).unwrap();
        drop(file);

        assert_eq!(probe(&path), Err(ArtifactProbeFailure::InvalidHeader));
    }

    #[test]
    fn json_probe_streams_large_values_without_materializing_a_dom() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("large-tokenizer");
        let mut file = File::create(&path).unwrap();
        file.write_all(b"{\"payload\":\"").unwrap();
        for _ in 0..8 {
            file.write_all(&vec![b'x'; 1024 * 1024]).unwrap();
        }
        file.write_all(b"\"}").unwrap();
        drop(file);

        assert_eq!(probe(&path), Ok(ArtifactStorageFormat::Json));
    }
}
