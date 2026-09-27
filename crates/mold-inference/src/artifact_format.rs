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

/// The weight encoding of a Qwen Image 2.1 transformer artifact.
///
/// Decided from the artifact's own header, never its tag or filename, so a
/// renamed or config-registered file loads through the right arm and the
/// transformer code itself never branches on a format — it asks for linears by
/// name and the loader answers with the matching `Q21Linear` arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum QwenImage21TransformerFormat {
    /// Plain floating-point safetensors — the diffusers BF16 shards
    /// (`Qwen/Qwen-Image-2.1` `transformer/`), with split `gate_layer`/`proj`.
    Bf16,
    /// Comfy `int8_tensorwise` with a regular 256-wide ConvRot: every quantized
    /// linear carries an `I8` weight, an `F32 [out, 1]` `weight_scale`, and a
    /// `.comfy_quant` marker naming the format (ComfyUI `comfy/ops.py:1224-1235`,
    /// `:1315-1317`). The MLP's gate and up projections are FUSED as
    /// `img_mlp.gate_up` (`comfy/ldm/qwen_image21/model.py:49-63`).
    ComfyInt8ConvRot,
    /// torchao `Float8Tensor` (unsloth `Qwen-Image-2.1-FP8`): `X._weight_qdata`
    /// `F8_E4M3` plus a per-output-row `X._weight_scale` `F32 [out, 1]`, with
    /// the MLP split as in the diffusers shards.
    TorchaoFp8,
    /// A GGUF (leejet / stable-diffusion.cpp or unsloth). Unsloth prefixes every
    /// tensor with [`QWEN_IMAGE21_GGUF_PREFIX`] and mixes block types per
    /// tensor; both ship the fused `img_mlp.gate_up`.
    Gguf { diffusion_model_prefix: bool },
}

impl QwenImage21TransformerFormat {
    /// Short label for logs and the per-step finiteness error.
    pub fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::ComfyInt8ConvRot => "int8-conv",
            Self::TorchaoFp8 => "fp8",
            Self::Gguf { .. } => "gguf",
        }
    }

    /// Whether the checkpoint fuses the MLP gate and up projections into one
    /// `img_mlp.gate_up` linear (gate rows first).
    pub fn fused_gate_up(self) -> bool {
        matches!(self, Self::ComfyInt8ConvRot | Self::Gguf { .. })
    }
}

/// The ComfyUI model prefix unsloth's GGUF conversions keep on every tensor.
pub const QWEN_IMAGE21_GGUF_PREFIX: &str = "model.diffusion_model.";

/// Largest `.comfy_quant` marker the probe will read. Real markers are a
/// 72-byte JSON object; anything near this bound is not a marker.
const MAX_COMFY_QUANT_MARKER_BYTES: usize = 4096;

/// Entries in [`probe_qwen_image21_transformer`]'s process-wide cache. A host
/// holds a handful of Qwen Image 2.1 tiers; the bound only keeps a
/// pathological caller from growing it.
const QWEN21_PROBE_CACHE_ENTRIES: usize = 16;

/// What identifies one on-disk transformer artifact for the probe cache: its
/// canonical path plus the metadata a rewrite or replacement changes (length,
/// mtime and, on Unix, the inode an atomic rename swaps).
#[derive(Clone, Debug, Eq, PartialEq)]
struct Qwen21ProbeKey {
    path: std::path::PathBuf,
    len: u64,
    modified: Option<std::time::SystemTime>,
    inode: Option<(u64, u64)>,
}

impl Qwen21ProbeKey {
    fn of(path: &Path) -> Option<Self> {
        let path = std::fs::canonicalize(path).ok()?;
        let metadata = std::fs::metadata(&path).ok()?;
        #[cfg(unix)]
        let inode = {
            use std::os::unix::fs::MetadataExt;
            Some((metadata.dev(), metadata.ino()))
        };
        #[cfg(not(unix))]
        let inode = None;
        Some(Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            inode,
            path,
        })
    }
}

type Qwen21ProbeCache =
    std::sync::Mutex<std::collections::VecDeque<(Qwen21ProbeKey, QwenImage21TransformerFormat)>>;

fn qwen21_probe_cache() -> &'static Qwen21ProbeCache {
    static CACHE: std::sync::OnceLock<Qwen21ProbeCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(Default::default)
}

#[cfg(test)]
thread_local! {
    static QWEN21_PROBE_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Header reads this thread's probes performed (cache misses).
#[cfg(test)]
fn qwen21_probe_reads() -> usize {
    QWEN21_PROBE_READS.with(std::cell::Cell::get)
}

/// Probe the transformer artifact at `path` (the first shard of a sharded
/// checkpoint is enough: every shard of one checkpoint shares its encoding).
///
/// One request asks this several times (residency sizing, the loader, the
/// tier planners), and an int8-conv checkpoint's answer reads hundreds of
/// per-layer `.comfy_quant` records, so a successful answer is cached per
/// file identity ([`Qwen21ProbeKey`]) in a small process-wide LRU. A failure
/// is never cached: the next call re-reads the file.
pub fn probe_qwen_image21_transformer(
    path: &Path,
) -> Result<QwenImage21TransformerFormat, ArtifactProbeFailure> {
    let Some(key) = Qwen21ProbeKey::of(path) else {
        return probe_qwen_image21_transformer_uncached(path);
    };
    {
        let mut cache = qwen21_probe_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(index) = cache.iter().position(|(cached, _)| *cached == key) {
            let entry = cache.remove(index).expect("index is in range");
            let format = entry.1;
            cache.push_back(entry);
            return Ok(format);
        }
    }
    let format = probe_qwen_image21_transformer_uncached(path)?;
    let mut cache = qwen21_probe_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // A stale entry for the same path (the file changed) is replaced, not
    // kept beside the new one.
    cache.retain(|(cached, _)| cached.path != key.path);
    if cache.len() >= QWEN21_PROBE_CACHE_ENTRIES {
        cache.pop_front();
    }
    cache.push_back((key, format));
    Ok(format)
}

fn probe_qwen_image21_transformer_uncached(
    path: &Path,
) -> Result<QwenImage21TransformerFormat, ArtifactProbeFailure> {
    #[cfg(test)]
    QWEN21_PROBE_READS.with(|reads| reads.set(reads.get() + 1));
    let mut file = File::open(path).map_err(|_| ArtifactProbeFailure::Io)?;
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic)
        .map_err(|_| ArtifactProbeFailure::UnsupportedContainer)?;
    if &magic == b"GGUF" {
        return probe_qwen_image21_gguf(path);
    }
    probe_qwen_image21_safetensors(path)
}

fn probe_qwen_image21_gguf(
    path: &Path,
) -> Result<QwenImage21TransformerFormat, ArtifactProbeFailure> {
    let file = File::open(path).map_err(|_| ArtifactProbeFailure::Io)?;
    let mut reader = file.take(MAX_GGUF_HEADER_BYTES);
    let magic = read_exact_array::<4>(&mut reader)?;
    if &magic != b"GGUF" {
        return Err(ArtifactProbeFailure::InvalidHeader);
    }
    let version = read_u32(&mut reader)?;
    if !(2..=3).contains(&version) {
        return Err(ArtifactProbeFailure::InvalidHeader);
    }
    let tensor_count = read_versioned_count(&mut reader, version)?;
    let metadata_count = read_versioned_count(&mut reader, version)?;
    if tensor_count == 0
        || tensor_count > MAX_GGUF_TENSORS
        || metadata_count > MAX_GGUF_METADATA_ITEMS
    {
        return Err(ArtifactProbeFailure::InvalidHeader);
    }
    for _ in 0..metadata_count {
        skip_gguf_string(&mut reader, version)?;
        let value_type = read_u32(&mut reader)?;
        skip_gguf_value(&mut reader, version, value_type, 0)?;
    }
    let (mut prefixed, mut bare) = (0_u64, 0_u64);
    for _ in 0..tensor_count {
        let length = read_versioned_count(&mut reader, version)?;
        if length > MAX_SAFETENSORS_KEY_BYTES as u64 {
            return Err(ArtifactProbeFailure::InvalidHeader);
        }
        let mut name = vec![0_u8; length as usize];
        reader
            .read_exact(&mut name)
            .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
        if name.starts_with(QWEN_IMAGE21_GGUF_PREFIX.as_bytes()) {
            prefixed += 1;
        } else {
            bare += 1;
        }
        let dimensions = read_u32(&mut reader)?;
        if dimensions > 8 {
            return Err(ArtifactProbeFailure::InvalidHeader);
        }
        skip_exact(&mut reader, u64::from(dimensions) * 8)?;
        // Tensor type, then the data offset.
        skip_exact(&mut reader, 4 + 8)?;
    }
    // A checkpoint either carries ComfyUI's model prefix on every tensor or on
    // none; a mixture has no single key space a loader could strip.
    match (prefixed, bare) {
        (0, _) => Ok(QwenImage21TransformerFormat::Gguf {
            diffusion_model_prefix: false,
        }),
        (_, 0) => Ok(QwenImage21TransformerFormat::Gguf {
            diffusion_model_prefix: true,
        }),
        _ => Err(ArtifactProbeFailure::InvalidHeader),
    }
}

#[derive(Deserialize)]
struct Qwen21TensorHeader {
    dtype: String,
    data_offsets: (u64, u64),
}

fn probe_qwen_image21_safetensors(
    path: &Path,
) -> Result<QwenImage21TransformerFormat, ArtifactProbeFailure> {
    use std::io::{Seek, SeekFrom};

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
    let mut header = Vec::new();
    (&mut file)
        .take(header_len)
        .read_to_end(&mut header)
        .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
    let mut entries: std::collections::BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header).map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
    entries.remove("__metadata__");
    if entries.is_empty() || entries.len() > MAX_SAFETENSORS_TENSORS {
        return Err(ArtifactProbeFailure::InvalidHeader);
    }
    let mut tensors = std::collections::BTreeMap::new();
    for (name, value) in entries {
        let tensor: Qwen21TensorHeader =
            serde_json::from_value(value).map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
        let dtype = parse_safetensors_dtype(&tensor.dtype)?;
        tensors.insert(name, (dtype, tensor.data_offsets));
    }
    let data_start = 8 + header_len;

    let markers = tensors
        .iter()
        .filter(|(name, _)| name.ends_with(".comfy_quant"))
        .map(|(name, (_, offsets))| (name.clone(), *offsets))
        .collect::<Vec<_>>();
    if !markers.is_empty() {
        for (marker, (begin, end)) in markers {
            let bytes = end
                .checked_sub(begin)
                .filter(|bytes| *bytes as usize <= MAX_COMFY_QUANT_MARKER_BYTES)
                .ok_or(ArtifactProbeFailure::InvalidHeader)?;
            let mut raw = vec![0_u8; bytes as usize];
            file.seek(SeekFrom::Start(data_start + begin))
                .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
            file.read_exact(&mut raw)
                .map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
            let config: serde_json::Value =
                serde_json::from_slice(&raw).map_err(|_| ArtifactProbeFailure::InvalidHeader)?;
            if !is_int8_convrot_256_marker(&config) {
                return Err(ArtifactProbeFailure::UnsupportedTensorDType);
            }
            let base = marker.trim_end_matches(".comfy_quant");
            let weight = tensors.get(&format!("{base}.weight"));
            let scale = tensors.get(&format!("{base}.weight_scale"));
            if !matches!(weight, Some((TensorDType::I8, _)))
                || !matches!(scale, Some((TensorDType::F32, _)))
            {
                return Err(ArtifactProbeFailure::InvalidHeader);
            }
        }
        return Ok(QwenImage21TransformerFormat::ComfyInt8ConvRot);
    }

    let fp8 = tensors
        .iter()
        .filter(|(name, _)| name.ends_with("._weight_qdata"))
        .collect::<Vec<_>>();
    if !fp8.is_empty() {
        for (name, (dtype, _)) in fp8 {
            let base = name.trim_end_matches("._weight_qdata");
            if *dtype != TensorDType::F8E4M3
                || !matches!(
                    tensors.get(&format!("{base}._weight_scale")),
                    Some((TensorDType::F32, _))
                )
            {
                return Err(ArtifactProbeFailure::InvalidHeader);
            }
        }
        return Ok(QwenImage21TransformerFormat::TorchaoFp8);
    }

    // No recognised quantization side-channel: every tensor must be a plain
    // float, or this is a quantized layout mold does not know how to read.
    if tensors.values().all(|(dtype, _)| {
        matches!(
            dtype,
            TensorDType::Bf16 | TensorDType::F16 | TensorDType::F32
        )
    }) {
        Ok(QwenImage21TransformerFormat::Bf16)
    } else {
        Err(ArtifactProbeFailure::UnsupportedTensorDType)
    }
}

/// The one Comfy marker Qwen Image 2.1's INT8 tier uses:
/// `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`,
/// with the options either at the top level or under `params` exactly as
/// ComfyUI reads them (`comfy/ops.py:1228-1235`; the group size defaults to 256).
fn is_int8_convrot_256_marker(config: &serde_json::Value) -> bool {
    let params = config.get("params");
    let field = |name: &str| {
        config
            .get(name)
            .or_else(|| params.and_then(|params| params.get(name)))
    };
    config.get("format").and_then(serde_json::Value::as_str) == Some("int8_tensorwise")
        && field("convrot").and_then(serde_json::Value::as_bool) == Some(true)
        && field("convrot_groupsize").map_or(Some(256), serde_json::Value::as_u64) == Some(256)
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

    /// Write a safetensors file from `(name, dtype, shape, bytes)` records.
    fn write_safetensors_records(path: &Path, records: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in records {
            let begin = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [begin, data.len()],
                }),
            );
        }
        write_safetensors(path, Value::Object(header), &data);
    }

    fn marker(json: &str) -> Vec<u8> {
        json.as_bytes().to_vec()
    }

    const INT8_MARKER: &str =
        r#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#;

    #[test]
    fn qwen21_probe_reads_the_diffusers_bf16_shards_as_bf16() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("shard.safetensors");
        write_safetensors_records(
            &path,
            &[
                ("img_in.weight", "BF16", vec![2, 2], vec![0; 8]),
                (
                    "transformer_blocks.0.img_mlp.gate_layer.weight",
                    "BF16",
                    vec![1, 2],
                    vec![0; 4],
                ),
                ("txt_in.text_norm.weight", "F32", vec![1], vec![0; 4]),
            ],
        );
        assert_eq!(
            probe_qwen_image21_transformer(&path),
            Ok(QwenImage21TransformerFormat::Bf16)
        );
    }

    /// A request resolves the transformer's format several times (residency
    /// sizing, load, per-tier planning); the header is read once per file
    /// identity, and a changed file — new length or mtime — is re-read.
    #[test]
    fn qwen21_probe_is_cached_per_file_identity_and_invalidated_on_change() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("shard.safetensors");
        let bf16 = [
            ("img_in.weight", "BF16", vec![2, 2], vec![0; 8]),
            ("txt_in.text_norm.weight", "F32", vec![1], vec![0; 4]),
        ];
        write_safetensors_records(&path, &bf16);
        let before = qwen21_probe_reads();
        for _ in 0..2 {
            assert_eq!(
                probe_qwen_image21_transformer(&path),
                Ok(QwenImage21TransformerFormat::Bf16)
            );
        }
        assert_eq!(qwen21_probe_reads() - before, 1, "second probe must hit");

        // Rewrite in place as the int8 tier and move the mtime forward, so a
        // coarse filesystem clock cannot hide the change.
        write_safetensors_records(
            &path,
            &[
                ("img_in.weight", "BF16", vec![2, 2], vec![0; 8]),
                (
                    "transformer_blocks.0.img_mlp.gate_up.weight",
                    "I8",
                    vec![2, 256],
                    vec![0; 512],
                ),
                (
                    "transformer_blocks.0.img_mlp.gate_up.weight_scale",
                    "F32",
                    vec![2, 1],
                    vec![0; 8],
                ),
                (
                    "transformer_blocks.0.img_mlp.gate_up.comfy_quant",
                    "U8",
                    vec![INT8_MARKER.len()],
                    marker(INT8_MARKER),
                ),
            ],
        );
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();
        assert_eq!(
            probe_qwen_image21_transformer(&path),
            Ok(QwenImage21TransformerFormat::ComfyInt8ConvRot)
        );
        assert_eq!(qwen21_probe_reads() - before, 2);

        // A failure is never cached: the file is re-read every time.
        std::fs::write(&path, b"not a checkpoint").unwrap();
        for _ in 0..2 {
            assert!(probe_qwen_image21_transformer(&path).is_err());
        }
        assert_eq!(qwen21_probe_reads() - before, 4);
    }

    #[test]
    fn qwen21_probe_recognises_the_comfy_int8_convrot_tier_by_its_markers() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("opaque.asset");
        let records = [
            ("img_in.weight", "BF16", vec![2, 2], vec![0; 8]),
            (
                "transformer_blocks.0.img_mlp.gate_up.weight",
                "I8",
                vec![2, 256],
                vec![0; 512],
            ),
            (
                "transformer_blocks.0.img_mlp.gate_up.weight_scale",
                "F32",
                vec![2, 1],
                vec![0; 8],
            ),
            (
                "transformer_blocks.0.img_mlp.gate_up.comfy_quant",
                "U8",
                vec![INT8_MARKER.len()],
                marker(INT8_MARKER),
            ),
        ];
        write_safetensors_records(&path, &records);
        let format = probe_qwen_image21_transformer(&path).unwrap();
        assert_eq!(format, QwenImage21TransformerFormat::ComfyInt8ConvRot);
        assert!(format.fused_gate_up());
        assert_eq!(format.label(), "int8-conv");

        // The options may also sit under `params`, as ComfyUI reads them.
        let nested = r#"{"format":"int8_tensorwise","params":{"convrot":true}}"#;
        let mut nested_records = records.clone();
        nested_records[3] = (
            "transformer_blocks.0.img_mlp.gate_up.comfy_quant",
            "U8",
            vec![nested.len()],
            marker(nested),
        );
        write_safetensors_records(&path, &nested_records);
        assert_eq!(
            probe_qwen_image21_transformer(&path),
            Ok(QwenImage21TransformerFormat::ComfyInt8ConvRot)
        );

        // Any other Comfy format — W4A4, unrotated INT8, another group size —
        // is a layout this loader does not execute.
        for other in [
            r#"{"format":"convrot_w4a4","convrot_groupsize":256}"#,
            r#"{"format":"int8_tensorwise"}"#,
            r#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":128}"#,
        ] {
            let mut other_records = records.clone();
            other_records[3] = (
                "transformer_blocks.0.img_mlp.gate_up.comfy_quant",
                "U8",
                vec![other.len()],
                marker(other),
            );
            write_safetensors_records(&path, &other_records);
            assert_eq!(
                probe_qwen_image21_transformer(&path),
                Err(ArtifactProbeFailure::UnsupportedTensorDType),
                "{other}"
            );
        }
    }

    #[test]
    fn qwen21_probe_recognises_torchao_fp8_by_its_qdata_and_scale_pairs() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("fp8.safetensors");
        let good = [
            ("img_in.weight", "BF16", vec![2, 2], vec![0; 8]),
            (
                "transformer_blocks.0.attn.to_q._weight_qdata",
                "F8_E4M3",
                vec![2, 2],
                vec![0; 4],
            ),
            (
                "transformer_blocks.0.attn.to_q._weight_scale",
                "F32",
                vec![2, 1],
                vec![0; 8],
            ),
        ];
        write_safetensors_records(&path, &good);
        let format = probe_qwen_image21_transformer(&path).unwrap();
        assert_eq!(format, QwenImage21TransformerFormat::TorchaoFp8);
        assert!(!format.fused_gate_up());

        // A qdata with no scale is not a checkpoint this loader can widen.
        write_safetensors_records(&path, &good[..2]);
        assert_eq!(
            probe_qwen_image21_transformer(&path),
            Err(ArtifactProbeFailure::InvalidHeader)
        );
        // A bare F8 tensor with no torchao side channel is an unknown layout.
        write_safetensors_records(
            &path,
            &[("img_in.weight", "F8_E4M3", vec![2, 2], vec![0; 4])],
        );
        assert_eq!(
            probe_qwen_image21_transformer(&path),
            Err(ArtifactProbeFailure::UnsupportedTensorDType)
        );
    }

    fn gguf_tensor(dtype: GgmlDType, rows: usize) -> candle_core::quantized::QTensor {
        let source = Tensor::zeros((rows, 256), candle_core::DType::F32, &Device::Cpu).unwrap();
        QTensor::quantize(&source, dtype).unwrap()
    }

    /// Unsloth's GGUFs keep ComfyUI's `model.diffusion_model.` prefix on every
    /// tensor and mix block types per tensor (Q4_K/Q5_K/Q6_K/Q8_0 blocks, Q5_K
    /// modulation, F32 norms) — the header of `qwen-image-2.1-Q4_K_M.gguf`,
    /// mirrored here in miniature. leejet's carry no prefix at all.
    #[test]
    fn qwen21_probe_reports_the_gguf_key_space() {
        let root = tempfile::tempdir().unwrap();
        let unsloth = root.path().join("unsloth.gguf");
        let tensors = [
            ("model.diffusion_model.img_in.weight", GgmlDType::BF16),
            ("model.diffusion_model.modulation.1.weight", GgmlDType::Q5K),
            (
                "model.diffusion_model.norm_out.linear.weight",
                GgmlDType::F32,
            ),
            (
                "model.diffusion_model.transformer_blocks.0.attn.to_q.weight",
                GgmlDType::Q4K,
            ),
            (
                "model.diffusion_model.transformer_blocks.0.attn.to_v.weight",
                GgmlDType::Q6K,
            ),
            (
                "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight",
                GgmlDType::Q8_0,
            ),
        ]
        .map(|(name, dtype)| (name, gguf_tensor(dtype, 2)));
        let refs = tensors
            .iter()
            .map(|(name, tensor)| (*name, tensor))
            .collect::<Vec<_>>();
        let mut file = File::create(&unsloth).unwrap();
        gguf_file::write(&mut file, &[], &refs).unwrap();
        drop(file);
        assert_eq!(
            probe_qwen_image21_transformer(&unsloth),
            Ok(QwenImage21TransformerFormat::Gguf {
                diffusion_model_prefix: true
            })
        );

        let leejet = root.path().join("leejet.gguf");
        let bare = [("img_in.weight", gguf_tensor(GgmlDType::BF16, 2))];
        let refs = bare.iter().map(|(n, t)| (*n, t)).collect::<Vec<_>>();
        let mut file = File::create(&leejet).unwrap();
        gguf_file::write(&mut file, &[], &refs).unwrap();
        drop(file);
        assert_eq!(
            probe_qwen_image21_transformer(&leejet),
            Ok(QwenImage21TransformerFormat::Gguf {
                diffusion_model_prefix: false
            })
        );

        let mixed = root.path().join("mixed.gguf");
        let both = [
            ("img_in.weight", gguf_tensor(GgmlDType::BF16, 2)),
            (
                "model.diffusion_model.proj_out.weight",
                gguf_tensor(GgmlDType::BF16, 2),
            ),
        ];
        let refs = both.iter().map(|(n, t)| (*n, t)).collect::<Vec<_>>();
        let mut file = File::create(&mixed).unwrap();
        gguf_file::write(&mut file, &[], &refs).unwrap();
        drop(file);
        assert_eq!(
            probe_qwen_image21_transformer(&mixed),
            Err(ArtifactProbeFailure::InvalidHeader)
        );
    }

    /// Every staged real tier probes to the format its loader expects. The
    /// unsloth entry is the first 16 MiB of `qwen-image-2.1-Q4_K_M.gguf` — the
    /// probe reads the header only, so a truncated file is a complete fixture.
    ///
    /// `MOLD_QWEN_IMAGE21_TIERS_DIR` names the staging directory (on plato,
    /// `/storage/mold/fixtures/qwen_image21/tiers-staging`).
    #[test]
    #[ignore = "needs the staged Qwen Image 2.1 tier checkpoints"]
    fn qwen21_probe_classifies_every_staged_tier() {
        let dir = std::path::PathBuf::from(
            std::env::var("MOLD_QWEN_IMAGE21_TIERS_DIR")
                .expect("MOLD_QWEN_IMAGE21_TIERS_DIR must name the staging dir"),
        );
        let gguf = |prefix| QwenImage21TransformerFormat::Gguf {
            diffusion_model_prefix: prefix,
        };
        let expectations = [
            (
                "qwen_image_2.1_int8_convrot.safetensors",
                QwenImage21TransformerFormat::ComfyInt8ConvRot,
            ),
            (
                "Qwen-Image-2.1-FP8.safetensors",
                QwenImage21TransformerFormat::TorchaoFp8,
            ),
            ("qwen_image_2.1-Q8_0.gguf", gguf(false)),
            ("qwen_image_2.1-Q4_K.gguf", gguf(false)),
            ("unsloth-qwen-image-2.1-Q4_K_M.head.gguf", gguf(true)),
        ];
        for (file, expected) in expectations {
            let path = dir.join(file);
            assert_eq!(
                probe_qwen_image21_transformer(&path),
                Ok(expected),
                "{}",
                path.display()
            );
        }
        if let Some(shard) = std::env::var_os("MOLD_QWEN_IMAGE21_BF16_DIR") {
            let path = std::path::PathBuf::from(shard)
                .join("transformer/diffusion_pytorch_model-00001-of-00002.safetensors");
            assert_eq!(
                probe_qwen_image21_transformer(&path),
                Ok(QwenImage21TransformerFormat::Bf16)
            );
        }
    }
}
