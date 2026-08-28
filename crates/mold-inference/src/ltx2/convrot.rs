use anyhow::{Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use super::nvfp4::remap_ltx2_transformer_key;

const CONVROT_GROUP_SIZE: usize = 256;

#[derive(Debug, Clone, Copy)]
enum KeySpace {
    Ltx2Transformer,
    Identity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantizedFormat {
    ConvRotW4A4 { group_size: usize },
    Int8Tensorwise { convrot: bool, group_size: usize },
}

/// The device classes the ConvRot backend distinguishes when it decides where
/// a packed weight is decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceKind {
    Cpu,
    Cuda,
    Metal,
}

pub(crate) fn device_kind(dev: &Device) -> DeviceKind {
    if dev.is_cuda() {
        DeviceKind::Cuda
    } else if dev.is_metal() {
        DeviceKind::Metal
    } else {
        DeviceKind::Cpu
    }
}

/// How an LTX-2 INT8 ConvRot transformer linear executes, per
/// `MOLD_LTX2_INT8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ltx2Int8Arm {
    /// Comfy's W8A8 order through `mold_candle::comfy_int8` — upstream's own
    /// execution for these checkpoints, and the default.
    Native,
    /// Widen the packed weight per forward (W8A16) and multiply like a
    /// `Standard` linear. The A/B escape hatch.
    Dequant,
}

/// Parse a `MOLD_LTX2_INT8` value. `dequant` (any case, trimmed) selects the
/// widening arm; everything else — including unset and the documented
/// `native` — keeps the default. Mirrored by mold-server's
/// `runtime_semantic_variable` canonicalization, which reduces the value to
/// this same boolean decision.
pub(crate) fn parse_ltx2_int8_arm(value: Option<&str>) -> Ltx2Int8Arm {
    match value {
        Some(value) if value.trim().eq_ignore_ascii_case("dequant") => Ltx2Int8Arm::Dequant,
        _ => Ltx2Int8Arm::Native,
    }
}

/// The process-frozen `MOLD_LTX2_INT8` decision.
pub(crate) fn ltx2_int8_arm() -> Ltx2Int8Arm {
    parse_ltx2_int8_arm(crate::runtime_env::value("MOLD_LTX2_INT8").as_deref())
}

/// Provenance literals for the INT8 execution arm, logged once each as
/// `ltx2 int8 arm=<literal>`. The UAT harness greps these exact strings;
/// change them only together with it.
pub(crate) const INT8_ARM_NATIVE_W8A8: &str = "native-w8a8";
pub(crate) const INT8_ARM_DEQUANT_CUDA: &str = "dequant-cuda";
pub(crate) const INT8_ARM_DEQUANT_METAL: &str = "dequant-metal";
pub(crate) const INT8_ARM_DEQUANT_HOST: &str = "dequant-host";

/// The literal describing how INT8 ConvRot transformer linears execute on one
/// device class under one `MOLD_LTX2_INT8` decision. `Native` means W8A8
/// semantics wherever it runs (the cuBLASLt kernel on CUDA, the bit-matching
/// portable reference elsewhere); `Dequant` names the device that widens.
pub(crate) fn int8_arm_provenance(kind: DeviceKind, arm: Ltx2Int8Arm) -> &'static str {
    match (arm, kind) {
        (Ltx2Int8Arm::Native, _) => INT8_ARM_NATIVE_W8A8,
        (Ltx2Int8Arm::Dequant, DeviceKind::Cuda) => INT8_ARM_DEQUANT_CUDA,
        (Ltx2Int8Arm::Dequant, DeviceKind::Metal) => INT8_ARM_DEQUANT_METAL,
        (Ltx2Int8Arm::Dequant, DeviceKind::Cpu) => INT8_ARM_DEQUANT_HOST,
    }
}

/// Log one INT8-arm literal at INFO, once per process per literal.
///
/// The env decision is process-frozen and the device class is fixed per
/// engine, so in practice one literal describes a whole render session. The
/// guard is per literal rather than global anyway: a multi-device process
/// (say a CUDA transformer beside a CPU-placed engine) legitimately earns two
/// distinct lines, and swallowing the second would hide which arm the other
/// device took.
pub(crate) fn log_int8_arm_once(arm: &'static str) {
    use std::sync::Mutex;
    static LOGGED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|error| error.into_inner());
    if !logged.contains(&arm) {
        logged.push(arm);
        tracing::info!("ltx2 int8 arm={arm}");
    }
}

/// Whether one quantized weight is decoded by
/// `candle_core::convrot::dequantize_int8_convrot_256` on the device rather
/// than by the host loop.
///
/// The op exists for exactly one layout — tensorwise INT8 with a regular
/// 256-wide ConvRot — and has a CUDA and a Metal arm; every other layout, and
/// every CPU decode, takes the host rayon path. A pure function of the format
/// and the device class so a machine with neither accelerator can still pin
/// both answers.
fn uses_device_kernel(format: QuantizedFormat, kind: DeviceKind) -> bool {
    matches!(
        format,
        QuantizedFormat::Int8Tensorwise {
            convrot: true,
            group_size: CONVROT_GROUP_SIZE
        }
    ) && matches!(kind, DeviceKind::Cuda | DeviceKind::Metal)
}

fn parse_quantized_format(config: &serde_json::Value) -> Result<QuantizedFormat> {
    let params = config.get("params");
    let integer = |name: &str, default: usize| {
        config
            .get(name)
            .or_else(|| params.and_then(|params| params.get(name)))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(default)
    };
    match config.get("format").and_then(serde_json::Value::as_str) {
        Some("convrot_w4a4") => Ok(QuantizedFormat::ConvRotW4A4 {
            group_size: integer("convrot_groupsize", CONVROT_GROUP_SIZE),
        }),
        Some("int8_tensorwise") => Ok(QuantizedFormat::Int8Tensorwise {
            convrot: config
                .get("convrot")
                .or_else(|| params.and_then(|params| params.get("convrot")))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            group_size: integer("convrot_groupsize", CONVROT_GROUP_SIZE),
        }),
        Some(other) => anyhow::bail!("unsupported Comfy quantization format '{other}'"),
        None => anyhow::bail!("Comfy quantization marker has no format"),
    }
}

pub(crate) fn checkpoint_is_convrot_w4a4(path: &Path) -> bool {
    let Ok(st) = (unsafe { MmapedSafetensors::new(path) }) else {
        return false;
    };
    let tensors = st.tensors();
    let keys: BTreeSet<_> = tensors.iter().map(|(key, _)| key.as_str()).collect();
    tensors.iter().any(|(key, view)| {
        let scale_key = format!("{}.weight_scale", key.trim_end_matches(".weight"));
        format!("{:?}", view.dtype()) == "I8"
            && key.ends_with(".weight")
            && keys.contains(scale_key.as_str())
    })
}

pub(super) struct Ltx2ConvRotBackend {
    st: MmapedSafetensors,
    keys: BTreeSet<String>,
    quantized: HashMap<String, QuantizedFormat>,
    key_space: KeySpace,
    /// Whether tensorwise INT8 ConvRot weights are offered PACKED through the
    /// synthetic `weight.convrot_packed` / `weight.convrot_scales` sub-keys,
    /// which `LtxLinear::load_with_nvfp4_cache` probes and turns into a
    /// device-resident `ConvRotPacked` linear. CUDA-only: the W8A8 kernel and
    /// the device ConvRot op both exist there, while Metal keeps the widening
    /// arm this backend serves through the plain `.weight` key.
    expose_packed_linears: bool,
}

impl Ltx2ConvRotBackend {
    pub(super) fn from_path(path: &Path) -> Result<Self> {
        Self::from_path_with_key_space(path, KeySpace::Ltx2Transformer)
    }

    /// The transformer key space with the packed side channel resolved for
    /// the device the `VarBuilder` will hand out tensors on.
    pub(super) fn from_path_for_device(path: &Path, device: &Device) -> Result<Self> {
        Ok(Self::from_path(path)?
            .with_packed_linears(matches!(device_kind(device), DeviceKind::Cuda)))
    }

    fn with_packed_linears(mut self, expose: bool) -> Self {
        self.expose_packed_linears = expose;
        self
    }

    pub(super) fn from_flattened_path(path: &Path) -> Result<Self> {
        Self::from_path_with_key_space(path, KeySpace::Identity)
    }

    fn from_path_with_key_space(path: &Path, key_space: KeySpace) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(path) }
            .with_context(|| format!("mmap LTX-2 ConvRot checkpoint at {}", path.display()))?;
        let keys: BTreeSet<String> = st.tensors().into_iter().map(|(key, _)| key).collect();
        let mut quantized = keys
            .iter()
            .filter_map(|key| key.strip_suffix(".weight_scale"))
            .filter(|base| keys.contains(&format!("{base}.weight")))
            .map(|base| {
                (
                    base.to_string(),
                    QuantizedFormat::ConvRotW4A4 {
                        group_size: CONVROT_GROUP_SIZE,
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        for marker in keys.iter().filter(|key| key.ends_with(".comfy_quant")) {
            let base = marker.trim_end_matches(".comfy_quant");
            let bytes = st.get(marker)?.data();
            let config: serde_json::Value = serde_json::from_slice(bytes).with_context(|| {
                format!(
                    "parse Comfy quantization marker {marker} in {}",
                    path.display()
                )
            })?;
            let format = parse_quantized_format(&config).with_context(|| {
                format!(
                    "invalid Comfy quantization marker {marker} in {}",
                    path.display()
                )
            })?;
            quantized.insert(base.to_string(), format);
        }
        Ok(Self {
            st,
            keys,
            quantized,
            key_space,
            expose_packed_linears: false,
        })
    }

    /// Whether `source_key` names an INT8 ConvRot weight the packed side
    /// channel serves: the exact `Int8Tensorwise { convrot: true, 256 }`
    /// layout `ComfyInt8ConvRotLinear` executes. W4A4 and unrotated INT8
    /// weights keep the widening arm whatever the device.
    fn packed_side_channel_serves(&self, source_key: &str) -> bool {
        self.expose_packed_linears
            && source_key.strip_suffix(".weight").is_some_and(|base| {
                matches!(
                    self.quantized.get(base),
                    Some(QuantizedFormat::Int8Tensorwise {
                        convrot: true,
                        group_size: CONVROT_GROUP_SIZE,
                    })
                )
            })
    }

    /// Load one component of the packed side channel for the weight at
    /// `source_key` onto `dev`: the raw two's-complement bytes, or the
    /// per-output-row F32 scales reshaped to their source `[rows, 1]`.
    fn load_packed_component(
        &self,
        source_key: &str,
        component: ConvRotComponent,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        match component {
            ConvRotComponent::Packed => {
                let view = self.st.get(source_key)?;
                if format!("{:?}", view.dtype()) != "I8" {
                    return Err(candle_core::Error::Msg(format!(
                        "LTX-2 ConvRot expected I8 packed weight at {source_key}, got {:?}",
                        view.dtype()
                    )));
                }
                let [rows, cols] = view.shape() else {
                    return Err(candle_core::Error::Msg(format!(
                        "LTX-2 ConvRot expected rank-2 packed weight at {source_key}, got {:?}",
                        view.shape()
                    )));
                };
                Tensor::from_slice(view.data(), (*rows, *cols), dev)
            }
            ConvRotComponent::Scales => {
                let base = source_key.trim_end_matches(".weight");
                let rows = self.st.get(source_key)?.shape()[0];
                self.st
                    .load(&format!("{base}.weight_scale"), dev)?
                    .to_dtype(DType::F32)?
                    .reshape((rows, 1))
            }
        }
    }

    fn source_key(&self, logical_name: &str) -> Option<String> {
        if matches!(self.key_space, KeySpace::Identity) {
            return self
                .keys
                .contains(logical_name)
                .then(|| logical_name.to_string());
        }
        let prefixed = remap_ltx2_transformer_key(logical_name);
        if self.keys.contains(&prefixed) {
            return Some(prefixed);
        }
        let stripped = prefixed.strip_prefix("model.diffusion_model.")?;
        self.keys.contains(stripped).then(|| stripped.to_string())
    }

    fn is_quantized_weight(&self, source_key: &str) -> bool {
        source_key
            .strip_suffix(".weight")
            .is_some_and(|base| self.quantized.contains_key(base))
    }

    /// The transformer key space widening an INT8 ConvRot weight is the
    /// non-CUDA execution arm for those linears, so name it once. On CUDA
    /// the packed side channel serves them and `LtxLinear` logs instead.
    fn log_widening_arm(&self, format: QuantizedFormat, kind: DeviceKind) {
        if !matches!(self.key_space, KeySpace::Ltx2Transformer) {
            return;
        }
        if !matches!(
            format,
            QuantizedFormat::Int8Tensorwise {
                convrot: true,
                group_size: CONVROT_GROUP_SIZE
            }
        ) {
            return;
        }
        match kind {
            DeviceKind::Metal => log_int8_arm_once(INT8_ARM_DEQUANT_METAL),
            DeviceKind::Cpu => log_int8_arm_once(INT8_ARM_DEQUANT_HOST),
            DeviceKind::Cuda => {}
        }
    }

    fn dequantize_weight(&self, source_key: &str, dev: &Device) -> candle_core::Result<Tensor> {
        let base = source_key.strip_suffix(".weight").ok_or_else(|| {
            candle_core::Error::Msg(format!("ConvRot source is not a weight: {source_key}"))
        })?;
        // Candle does not expose an I8 tensor dtype, so preserve the raw
        // two's-complement bytes from the safetensors view. The tensorwise
        // INT8 ConvRot layout decodes on the device (CUDA and Metal share one
        // candle op); every other layout, and every CPU decode, runs the host
        // row loop below. Either way the work is bounded to one dense weight;
        // never queue whole-model reconstructions from this backend.
        let packed_view = self.st.get(source_key)?;
        if format!("{:?}", packed_view.dtype()) != "I8" {
            return Err(candle_core::Error::Msg(format!(
                "LTX-2 ConvRot expected I8 packed weight at {source_key}, got {:?}",
                packed_view.dtype()
            )));
        }
        let [rows, packed_cols] = packed_view.shape() else {
            return Err(candle_core::Error::Msg(format!(
                "LTX-2 ConvRot expected rank-2 packed weight at {source_key}, got {:?}",
                packed_view.shape()
            )));
        };
        let scales = self
            .st
            .load(&format!("{base}.weight_scale"), &Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rows = *rows;
        if scales.len() != 1 && scales.len() != rows {
            return Err(candle_core::Error::Msg(format!(
                "LTX-2 Comfy quant scale count mismatch for {source_key}: {} rows, {} scales",
                rows,
                scales.len()
            )));
        }
        let format = self.quantized.get(base).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "missing Comfy quantization format for {source_key}"
            ))
        })?;
        self.log_widening_arm(*format, device_kind(dev));
        if uses_device_kernel(*format, device_kind(dev)) {
            // Upload straight from the mmap'd view: the packed bytes are the
            // exact device input, so there is no host copy to make first.
            let packed = Tensor::from_slice(packed_view.data(), (rows, *packed_cols), dev)?;
            let scale_count = scales.len();
            let scales = Tensor::from_vec(scales, scale_count, dev)?;
            let output = candle_core::convrot::dequantize_int8_convrot_256(&packed, &scales)?;
            // On Metal, bound residency to this packed input plus its BF16
            // output: do not let streaming layer loads queue packed input
            // buffers behind later command buffers after the local tensor is
            // dropped. CUDA needs no barrier here — the streaming loop already
            // synchronizes every `streaming_prefetch_count` blocks, and the
            // caching allocator retires the packed buffer in stream order.
            if dev.is_metal() {
                dev.synchronize()?;
            }
            return Ok(output);
        }
        let packed = packed_view
            .data()
            .chunks_exact(*packed_cols)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>();
        let (values, cols) = match *format {
            QuantizedFormat::ConvRotW4A4 { group_size } => (
                dequantize_convrot_rows(&packed, &scales, group_size)?,
                packed.first().map_or(0, |row| row.len() * 2),
            ),
            QuantizedFormat::Int8Tensorwise {
                convrot,
                group_size,
            } => (
                dequantize_int8_rows(&packed, &scales, convrot, group_size)?,
                packed.first().map_or(0, Vec::len),
            ),
        };
        // Keep only the returned model-precision tensor. Streaming callers
        // drop each layer/block after use; retaining a backend cache would
        // silently reconstruct the compact checkpoint into a 40+ GB model.
        let tensor = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?.to_dtype(DType::BF16)?;
        Ok(tensor)
    }

    fn lookup(&self, name: &str, dev: &Device) -> candle_core::Result<Tensor> {
        if let Some((weight_name, component)) = convrot_component(name) {
            let source_key = self.source_key(weight_name).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "LTX-2 ConvRot side channel: no source weight for '{weight_name}'"
                ))
            })?;
            if !self.packed_side_channel_serves(&source_key) {
                return Err(candle_core::Error::Msg(format!(
                    "LTX-2 ConvRot side channel is not exposed for '{name}'"
                )));
            }
            return self.load_packed_component(&source_key, component, dev);
        }
        let source_key = self
            .source_key(name)
            .unwrap_or_else(|| remap_ltx2_transformer_key(name));
        let tensor = if self.is_quantized_weight(&source_key) {
            self.dequantize_weight(&source_key, dev).map_err(|error| {
                candle_core::Error::Msg(format!(
                    "failed to reconstruct LTX-2 ConvRot tensor {name} from {source_key}: {error}"
                ))
            })?
        } else {
            self.st.load(&source_key, &Device::Cpu).map_err(|error| {
                candle_core::Error::Msg(format!(
                    "failed to load LTX-2 ConvRot tensor {name} from {source_key}: {error}"
                ))
            })?
        };
        tensor.to_device(dev)
    }
}

fn dequantize_convrot_rows(
    packed: &[Vec<u8>],
    scales: &[f32],
    group_size: usize,
) -> candle_core::Result<Vec<f32>> {
    if group_size < 4 || !group_size.is_power_of_two() || group_size.trailing_zeros() & 1 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "ConvRot group size must be a power of four, got {group_size}"
        )));
    }
    let cols = packed.first().map_or(0, |row| row.len() * 2);
    if cols.checked_rem(group_size) != Some(0) {
        return Err(candle_core::Error::Msg(format!(
            "ConvRot input width {cols} is not divisible by group size {group_size}"
        )));
    }
    let normalization = (group_size as f32).sqrt().recip();
    let mut output = vec![0.0; packed.len() * cols];
    output
        .par_chunks_mut(cols)
        .zip(packed.par_iter())
        .enumerate()
        .for_each(|(index, (values, row))| {
            let scale = scales[if scales.len() == 1 { 0 } else { index }];
            let (pairs, remainder) = values.as_chunks_mut::<2>();
            debug_assert!(remainder.is_empty());
            for (pair, byte) in pairs.iter_mut().zip(row) {
                pair[0] = sign_extend_nibble(byte & 0x0f) as f32 * scale;
                pair[1] = sign_extend_nibble(byte >> 4) as f32 * scale;
            }
            for group in values.chunks_exact_mut(group_size) {
                hadamard4_in_place(group);
                for value in group {
                    *value *= normalization;
                }
            }
        });
    Ok(output)
}

fn sign_extend_nibble(value: u8) -> i8 {
    if value >= 8 {
        (value as i8) - 16
    } else {
        value as i8
    }
}

fn hadamard4_in_place(values: &mut [f32]) {
    let mut stride = 1;
    while stride < values.len() {
        let block = stride * 4;
        for base in (0..values.len()).step_by(block) {
            for offset in 0..stride {
                let i0 = base + offset;
                let i1 = i0 + stride;
                let i2 = i1 + stride;
                let i3 = i2 + stride;
                let (a, b, c, d) = (values[i0], values[i1], values[i2], values[i3]);
                values[i0] = a + b + c - d;
                values[i1] = a + b - c + d;
                values[i2] = a - b + c + d;
                values[i3] = -a + b + c + d;
            }
        }
        stride = block;
    }
}

impl SimpleBackend for Ltx2ConvRotBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _hints: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.lookup(name, dev)?;
        if tensor.shape() != &shape {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("LTX-2 ConvRot shape mismatch for {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            }
            .bt());
        }
        tensor.to_dtype(dtype)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.lookup(name, dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        if let Some((weight_name, _)) = convrot_component(name) {
            return self
                .source_key(weight_name)
                .is_some_and(|source_key| self.packed_side_channel_serves(&source_key));
        }
        self.source_key(name)
            .is_some_and(|source_key| !is_consumed_convrot_sidecar(&source_key, &self.quantized))
    }
}

/// One component of the packed INT8 ConvRot side channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConvRotComponent {
    /// `weight.convrot_packed`: the raw two's-complement bytes, U8 `[out, in]`.
    Packed,
    /// `weight.convrot_scales`: the per-output-row scales, F32 `[out, 1]`.
    Scales,
}

/// Split a synthetic side-channel key into the logical weight key it targets
/// and the component it asks for.
fn convrot_component(name: &str) -> Option<(&str, ConvRotComponent)> {
    if let Some(weight_name) = name.strip_suffix(".convrot_packed") {
        Some((weight_name, ConvRotComponent::Packed))
    } else {
        name.strip_suffix(".convrot_scales")
            .map(|weight_name| (weight_name, ConvRotComponent::Scales))
    }
}

fn dequantize_int8_rows(
    rows: &[Vec<u8>],
    scales: &[f32],
    convrot: bool,
    group_size: usize,
) -> candle_core::Result<Vec<f32>> {
    let cols = rows.first().map_or(0, Vec::len);
    if convrot && cols.checked_rem(group_size) != Some(0) {
        return Err(candle_core::Error::Msg(format!(
            "ConvRot INT8 input width {cols} is not divisible by group size {group_size}"
        )));
    }
    let normalization = (group_size as f32).sqrt().recip();
    let mut output = vec![0.0; rows.len() * cols];
    output
        .par_chunks_mut(cols)
        .zip(rows.par_iter())
        .enumerate()
        .for_each(|(index, (values, row))| {
            let scale = scales[if scales.len() == 1 { 0 } else { index }];
            for (value, byte) in values.iter_mut().zip(row) {
                *value = (*byte as i8) as f32 * scale;
            }
            if convrot {
                for group in values.chunks_exact_mut(group_size) {
                    hadamard4_in_place(group);
                    for value in group {
                        *value *= normalization;
                    }
                }
            }
        });
    Ok(output)
}

fn is_consumed_convrot_sidecar(
    source_key: &str,
    quantized: &HashMap<String, QuantizedFormat>,
) -> bool {
    source_key
        .strip_suffix(".weight_scale")
        .or_else(|| source_key.strip_suffix(".comfy_quant"))
        .is_some_and(|base| quantized.contains_key(base))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantizes_and_unrotates_packed_rows() {
        // Packed low-nibble first: rotated quantized row [1, 0, 0, 0]
        // with scale 2. H4 / sqrt(4) maps it back to [1, 1, 1, -1].
        let values = dequantize_convrot_rows(&[vec![0x01, 0x00]], &[2.0], 4).unwrap();
        assert_eq!(values, vec![1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn signed_nibbles_match_comfy_storage_contract() {
        assert_eq!(sign_extend_nibble(0x7), 7);
        assert_eq!(sign_extend_nibble(0x8), -8);
        assert_eq!(sign_extend_nibble(0xf), -1);
    }

    #[test]
    fn row_scales_are_hidden_after_weight_reconstruction() {
        let bases = HashMap::from([(
            "transformer_blocks.1.attn1.to_q".to_string(),
            QuantizedFormat::ConvRotW4A4 { group_size: 256 },
        )]);
        assert!(is_consumed_convrot_sidecar(
            "transformer_blocks.1.attn1.to_q.weight_scale",
            &bases,
        ));
        assert!(is_consumed_convrot_sidecar(
            "transformer_blocks.1.attn1.to_q.comfy_quant",
            &bases,
        ));
        assert!(!is_consumed_convrot_sidecar(
            "transformer_blocks.1.attn1.to_q.input_scale",
            &bases,
        ));
    }

    #[test]
    fn int8_tensorwise_rows_dequantize_and_unrotate() {
        let values = dequantize_int8_rows(&[vec![1, 0, 0, 0]], &[2.0], true, 4).unwrap();
        assert_eq!(values, vec![1.0, 1.0, 1.0, -1.0]);
        let plain = dequantize_int8_rows(&[vec![0xff, 0x02]], &[0.5], false, 256).unwrap();
        assert_eq!(plain, vec![-0.5, 1.0]);
    }

    #[test]
    fn device_kernel_is_selected_for_int8_convrot_256_on_cuda_and_metal_only() {
        let int8_convrot = QuantizedFormat::Int8Tensorwise {
            convrot: true,
            group_size: CONVROT_GROUP_SIZE,
        };
        let int8_plain = QuantizedFormat::Int8Tensorwise {
            convrot: false,
            group_size: CONVROT_GROUP_SIZE,
        };
        let int8_other_group = QuantizedFormat::Int8Tensorwise {
            convrot: true,
            group_size: 64,
        };
        let w4a4 = QuantizedFormat::ConvRotW4A4 {
            group_size: CONVROT_GROUP_SIZE,
        };
        let table = [
            (int8_convrot, DeviceKind::Cuda, true),
            (int8_convrot, DeviceKind::Metal, true),
            (int8_convrot, DeviceKind::Cpu, false),
            (int8_plain, DeviceKind::Cuda, false),
            (int8_plain, DeviceKind::Metal, false),
            (int8_other_group, DeviceKind::Cuda, false),
            (int8_other_group, DeviceKind::Metal, false),
            (w4a4, DeviceKind::Cuda, false),
            (w4a4, DeviceKind::Metal, false),
            (w4a4, DeviceKind::Cpu, false),
        ];
        for (format, kind, expected) in table {
            assert_eq!(
                uses_device_kernel(format, kind),
                expected,
                "{format:?} on {kind:?}"
            );
        }
        assert_eq!(device_kind(&Device::Cpu), DeviceKind::Cpu);
    }

    /// The candle op applies `scale / 16` once after an exact integer
    /// butterfly; the host loop scales every element first and runs the
    /// butterfly over rounded `f32` products, so a value that is exactly zero
    /// on the device can be a ~1e-8 rounding residue on the host. Both narrow
    /// to BF16 with round-to-nearest-even, so they agree to one BF16 ulp
    /// (`|x| / 128`) plus that residue rather than bit-for-bit.
    fn assert_within_one_bf16_ulp(device: &Device) {
        use half::bf16;
        let rows = 3;
        let cols = 2 * CONVROT_GROUP_SIZE;
        let packed = (0..rows * cols)
            .map(|index| ((index as i32 * 37 + 11) % 251 - 125) as i8 as u8)
            .collect::<Vec<_>>();
        let scales = vec![0.003f32, 0.0125, 0.7];
        let host_rows = packed
            .chunks_exact(cols)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>();
        let expected = dequantize_int8_rows(&host_rows, &scales, true, CONVROT_GROUP_SIZE)
            .unwrap()
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let packed = Tensor::from_slice(&packed, (rows, cols), device).unwrap();
        let scales = Tensor::from_vec(scales, rows, device).unwrap();
        let actual = candle_core::convrot::dequantize_int8_convrot_256(&packed, &scales)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<bf16>()
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let (actual, expected) = (actual.to_f32(), expected.to_f32());
            let tolerance = expected.abs() / 128.0 + 1e-5;
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: device {actual} vs host {expected} (tolerance {tolerance})"
            );
        }
    }

    #[test]
    fn cpu_op_matches_host_rows_within_bf16() {
        assert_within_one_bf16_ulp(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dequant_matches_host_rows_within_bf16() {
        let Ok(cuda) = Device::new_cuda(0) else {
            return;
        };
        assert_within_one_bf16_ulp(&cuda);
    }

    #[test]
    fn comfy_markers_select_exact_convrot_layouts() {
        assert_eq!(
            parse_quantized_format(&serde_json::json!({
                "format": "convrot_w4a4",
                "params": {"convrot_groupsize": 64}
            }))
            .unwrap(),
            QuantizedFormat::ConvRotW4A4 { group_size: 64 }
        );
        assert_eq!(
            parse_quantized_format(&serde_json::json!({
                "format": "int8_tensorwise",
                "convrot": true,
                "convrot_groupsize": 256
            }))
            .unwrap(),
            QuantizedFormat::Int8Tensorwise {
                convrot: true,
                group_size: 256,
            }
        );
        assert!(parse_quantized_format(&serde_json::json!({"format": "nvfp4"})).is_err());
    }

    #[test]
    fn ltx2_int8_arm_parses_dequant_and_defaults_to_native() {
        assert_eq!(parse_ltx2_int8_arm(None), Ltx2Int8Arm::Native);
        assert_eq!(parse_ltx2_int8_arm(Some("native")), Ltx2Int8Arm::Native);
        assert_eq!(parse_ltx2_int8_arm(Some("")), Ltx2Int8Arm::Native);
        assert_eq!(parse_ltx2_int8_arm(Some("w8a8")), Ltx2Int8Arm::Native);
        assert_eq!(parse_ltx2_int8_arm(Some("dequant")), Ltx2Int8Arm::Dequant);
        assert_eq!(parse_ltx2_int8_arm(Some(" DEQUANT ")), Ltx2Int8Arm::Dequant);
    }

    #[test]
    fn int8_arm_provenance_names_the_numerics_and_the_widening_device() {
        let table = [
            (DeviceKind::Cuda, Ltx2Int8Arm::Native, INT8_ARM_NATIVE_W8A8),
            (DeviceKind::Cpu, Ltx2Int8Arm::Native, INT8_ARM_NATIVE_W8A8),
            (DeviceKind::Metal, Ltx2Int8Arm::Native, INT8_ARM_NATIVE_W8A8),
            (
                DeviceKind::Cuda,
                Ltx2Int8Arm::Dequant,
                INT8_ARM_DEQUANT_CUDA,
            ),
            (
                DeviceKind::Metal,
                Ltx2Int8Arm::Dequant,
                INT8_ARM_DEQUANT_METAL,
            ),
            (DeviceKind::Cpu, Ltx2Int8Arm::Dequant, INT8_ARM_DEQUANT_HOST),
        ];
        for (kind, arm, expected) in table {
            assert_eq!(int8_arm_provenance(kind, arm), expected, "{kind:?} {arm:?}");
        }
        assert_eq!(INT8_ARM_NATIVE_W8A8, "native-w8a8");
        assert_eq!(INT8_ARM_DEQUANT_CUDA, "dequant-cuda");
        assert_eq!(INT8_ARM_DEQUANT_METAL, "dequant-metal");
        assert_eq!(INT8_ARM_DEQUANT_HOST, "dequant-host");
    }

    fn write_int8_convrot_fixture(path: &Path) {
        use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};

        let rotated_weight = (0..CONVROT_GROUP_SIZE)
            .map(|index| ((index as i32 % 17) - 8) as i8 as u8)
            .collect::<Vec<_>>();
        let scale = 0.25f32.to_le_bytes().to_vec();
        let convrot_marker =
            br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}"#.to_vec();
        let plain_marker = br#"{"format":"int8_tensorwise","convrot":false}"#.to_vec();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "transformer_blocks.0.attn1.to_q.weight".to_string(),
            TensorView::new(SafeDtype::I8, vec![1, CONVROT_GROUP_SIZE], &rotated_weight).unwrap(),
        );
        tensors.insert(
            "transformer_blocks.0.attn1.to_q.weight_scale".to_string(),
            TensorView::new(SafeDtype::F32, vec![1, 1], &scale).unwrap(),
        );
        tensors.insert(
            "transformer_blocks.0.attn1.to_q.comfy_quant".to_string(),
            TensorView::new(SafeDtype::U8, vec![convrot_marker.len()], &convrot_marker).unwrap(),
        );
        // An unrotated INT8 weight: never packed-exposed, whatever the device.
        tensors.insert(
            "transformer_blocks.0.attn1.to_k.weight".to_string(),
            TensorView::new(SafeDtype::I8, vec![1, CONVROT_GROUP_SIZE], &rotated_weight).unwrap(),
        );
        tensors.insert(
            "transformer_blocks.0.attn1.to_k.weight_scale".to_string(),
            TensorView::new(SafeDtype::F32, vec![1, 1], &scale).unwrap(),
        );
        tensors.insert(
            "transformer_blocks.0.attn1.to_k.comfy_quant".to_string(),
            TensorView::new(SafeDtype::U8, vec![plain_marker.len()], &plain_marker).unwrap(),
        );
        serialize_to_file(&tensors, &None, path).unwrap();
    }

    #[test]
    fn packed_side_channel_serves_int8_convrot_weights_only_when_exposed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("int8-convrot.safetensors");
        write_int8_convrot_fixture(&path);

        let hidden = Ltx2ConvRotBackend::from_path(&path).unwrap();
        let packed_key = "transformer_blocks.0.attn1.to_q.weight.convrot_packed";
        let scales_key = "transformer_blocks.0.attn1.to_q.weight.convrot_scales";
        assert!(!hidden.contains_tensor(packed_key));
        assert!(hidden.lookup(packed_key, &Device::Cpu).is_err());

        let exposed = Ltx2ConvRotBackend::from_path(&path)
            .unwrap()
            .with_packed_linears(true);
        assert!(exposed.contains_tensor(packed_key));
        assert!(exposed.contains_tensor(scales_key));
        // The dense weight stays visible for non-linear consumers, and the
        // consumed sidecars stay hidden.
        assert!(exposed.contains_tensor("transformer_blocks.0.attn1.to_q.weight"));
        assert!(!exposed.contains_tensor("transformer_blocks.0.attn1.to_q.weight_scale"));
        // The unrotated INT8 weight never gets a packed channel.
        assert!(!exposed.contains_tensor("transformer_blocks.0.attn1.to_k.weight.convrot_packed"));

        let packed = exposed.lookup(packed_key, &Device::Cpu).unwrap();
        assert_eq!(packed.dtype(), DType::U8);
        assert_eq!(packed.dims(), &[1, CONVROT_GROUP_SIZE]);
        // The raw two's-complement bytes, byte for byte: -8, -7, -6, -5.
        assert_eq!(
            packed.flatten_all().unwrap().to_vec1::<u8>().unwrap()[..4],
            [0xF8u8, 0xF9, 0xFA, 0xFB][..]
        );
        let scales = exposed.lookup(scales_key, &Device::Cpu).unwrap();
        assert_eq!(scales.dtype(), DType::F32);
        assert_eq!(scales.dims(), &[1, 1]);
        assert_eq!(
            scales.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.25]
        );

        // `from_path_for_device` resolves the exposure from the device class.
        let cpu_backend = Ltx2ConvRotBackend::from_path_for_device(&path, &Device::Cpu).unwrap();
        assert!(!cpu_backend.contains_tensor(packed_key));
    }
}
