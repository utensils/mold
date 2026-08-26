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
}

impl Ltx2ConvRotBackend {
    pub(super) fn from_path(path: &Path) -> Result<Self> {
        Self::from_path_with_key_space(path, KeySpace::Ltx2Transformer)
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
        })
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

    fn dequantize_weight(&self, source_key: &str, dev: &Device) -> candle_core::Result<Tensor> {
        let base = source_key.strip_suffix(".weight").ok_or_else(|| {
            candle_core::Error::Msg(format!("ConvRot source is not a weight: {source_key}"))
        })?;
        // Candle does not expose an I8 tensor dtype, so preserve the raw
        // two's-complement bytes from the safetensors view and decode them on
        // the host. The row loop is parallel and bounded to one dense weight;
        // never queue whole-model Metal reconstructions from this backend.
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
        if matches!(
            format,
            QuantizedFormat::Int8Tensorwise {
                convrot: true,
                group_size: CONVROT_GROUP_SIZE
            }
        ) && dev.is_metal()
        {
            let packed = Tensor::from_vec(packed_view.data().to_vec(), (rows, *packed_cols), dev)?;
            let scale_count = scales.len();
            let scales = Tensor::from_vec(scales, scale_count, dev)?;
            let output = candle_core::convrot::dequantize_int8_convrot_256(&packed, &scales)?;
            // Bound residency to this packed input plus its BF16 output. In
            // particular, do not let streaming layer loads queue packed input
            // buffers behind later command buffers after the local tensor is
            // dropped.
            dev.synchronize()?;
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
        self.source_key(name)
            .is_some_and(|source_key| !is_consumed_convrot_sidecar(&source_key, &self.quantized))
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
}
