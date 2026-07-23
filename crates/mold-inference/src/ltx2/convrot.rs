use anyhow::{Context, Result};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Mutex;

use super::nvfp4::remap_ltx2_transformer_key;

const CONVROT_GROUP_SIZE: usize = 256;

pub(super) fn checkpoint_is_convrot_w4a4(path: &Path) -> bool {
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
    quantized_bases: BTreeSet<String>,
    cpu_cache: Mutex<HashMap<String, Tensor>>,
}

impl Ltx2ConvRotBackend {
    pub(super) fn from_path(path: &Path) -> Result<Self> {
        let st = unsafe { MmapedSafetensors::new(path) }
            .with_context(|| format!("mmap LTX-2 ConvRot checkpoint at {}", path.display()))?;
        let keys: BTreeSet<String> = st.tensors().into_iter().map(|(key, _)| key).collect();
        let quantized_bases = keys
            .iter()
            .filter_map(|key| key.strip_suffix(".weight_scale"))
            .filter(|base| keys.contains(&format!("{base}.weight")))
            .map(str::to_string)
            .collect();
        Ok(Self {
            st,
            keys,
            quantized_bases,
            cpu_cache: Mutex::new(HashMap::new()),
        })
    }

    fn source_key(&self, logical_name: &str) -> Option<String> {
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
            .is_some_and(|base| self.quantized_bases.contains(base))
    }

    fn dequantize_weight(&self, source_key: &str) -> candle_core::Result<Tensor> {
        if let Some(cached) = self
            .cpu_cache
            .lock()
            .map_err(|_| candle_core::Error::Msg("LTX-2 ConvRot cache poisoned".into()))?
            .get(source_key)
            .cloned()
        {
            return Ok(cached);
        }

        let base = source_key.strip_suffix(".weight").ok_or_else(|| {
            candle_core::Error::Msg(format!("ConvRot source is not a weight: {source_key}"))
        })?;
        // Candle does not expose an I8 tensor dtype, so read the packed bytes
        // straight from the safetensors view and interpret each byte as two
        // signed four-bit values below.
        let packed = self.st.get(source_key)?;
        if format!("{:?}", packed.dtype()) != "I8" {
            return Err(candle_core::Error::Msg(format!(
                "LTX-2 ConvRot expected I8 packed weight at {source_key}, got {:?}",
                packed.dtype()
            )));
        }
        let [rows, packed_cols] = packed.shape() else {
            return Err(candle_core::Error::Msg(format!(
                "LTX-2 ConvRot expected rank-2 packed weight at {source_key}, got {:?}",
                packed.shape()
            )));
        };
        let packed = packed
            .data()
            .chunks_exact(*packed_cols)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>();
        let scales = self
            .st
            .load(&format!("{base}.weight_scale"), &Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rows = *rows;
        let cols = packed.first().map_or(0, |row| row.len() * 2);
        if scales.len() != rows {
            return Err(candle_core::Error::Msg(format!(
                "LTX-2 ConvRot scale count mismatch for {source_key}: {} rows, {} scales",
                rows,
                scales.len()
            )));
        }
        let values = dequantize_convrot_rows(&packed, &scales, CONVROT_GROUP_SIZE)?;
        // Cache the reconstructed model precision, not F32: keeping all
        // streamed 22B weights as F32 would roughly double host residency.
        let tensor = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?.to_dtype(DType::BF16)?;
        self.cpu_cache
            .lock()
            .map_err(|_| candle_core::Error::Msg("LTX-2 ConvRot cache poisoned".into()))?
            .insert(source_key.to_string(), tensor.clone());
        Ok(tensor)
    }

    fn lookup(&self, name: &str, dev: &Device) -> candle_core::Result<Tensor> {
        let source_key = self
            .source_key(name)
            .unwrap_or_else(|| remap_ltx2_transformer_key(name));
        let tensor = if self.is_quantized_weight(&source_key) {
            self.dequantize_weight(&source_key)?
        } else {
            self.st.load(&source_key, &Device::Cpu)?
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
    let mut output = Vec::with_capacity(packed.len() * cols);
    for (row, scale) in packed.iter().zip(scales) {
        let mut values = Vec::with_capacity(cols);
        for byte in row {
            let byte = *byte;
            values.push(sign_extend_nibble(byte & 0x0f) as f32 * *scale);
            values.push(sign_extend_nibble(byte >> 4) as f32 * *scale);
        }
        for group in values.chunks_exact_mut(group_size) {
            hadamard4_in_place(group);
            for value in group {
                *value *= normalization;
            }
        }
        output.extend(values);
    }
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
        self.source_key(name).is_some_and(|source_key| {
            !is_consumed_convrot_sidecar(&source_key, &self.quantized_bases)
        })
    }
}

fn is_consumed_convrot_sidecar(source_key: &str, quantized_bases: &BTreeSet<String>) -> bool {
    source_key
        .strip_suffix(".weight_scale")
        .is_some_and(|base| quantized_bases.contains(base))
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
        let bases = BTreeSet::from(["transformer_blocks.1.attn1.to_q".to_string()]);
        assert!(is_consumed_convrot_sidecar(
            "transformer_blocks.1.attn1.to_q.weight_scale",
            &bases,
        ));
        assert!(!is_consumed_convrot_sidecar(
            "transformer_blocks.1.attn1.to_q.input_scale",
            &bases,
        ));
    }
}
