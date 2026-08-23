//! Private, release-unregistered NVFP4 layout / divergence / cost probe for the
//! published MiniMax H3 FL2VA checkpoints (issue #1317, part 1).
//!
//! This example is deliberately unregistered: `mold-candle`'s `Cargo.toml` has
//! no `[[example]]` table, so autodiscovery is the only thing that builds it and
//! it reaches no release surface. It also never calls
//! `open_h3_comfy_published_int8_checkpoint`: the published NVFP4 artifact
//! carries ~65 bytes of non-tensor trailing data past its payload, which the
//! production opener refuses, so this file does its own bounded header read.
//!
//! It settles three questions in one JSON report:
//!
//!   (a) Are comfy-kitchen NVFP4 block scales swizzled on H3-shaped tensors?
//!       Every H3 linear has `out % 128 == 0` and `in/16 % 4 == 0`, so the
//!       swizzled and natural scale shapes coincide and the header cannot
//!       answer. Both hypotheses are dequantized and scored against the pruned
//!       BF16 base both quantizations were derived from.
//!   (b) How far NVFP4 and INT8 ConvRot each sit from that BF16 base, in weight
//!       space and in activation space.
//!   (c) What the host-scalar NVFP4 dequantize costs against the native INT8
//!       cuBLASLt arm, and what a prototype device-side dequantize arm costs.

#[allow(dead_code)]
mod support {
    use std::collections::BTreeMap;
    use std::fmt;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::Path;

    use serde::de::{self, MapAccess, Visitor};
    use serde::{Deserialize, Deserializer};
    use serde_json::Value;

    /// Bounded header allowance. The largest H3 artifact header observed is
    /// 116,000 bytes; 8 MiB leaves headroom without letting a malformed length
    /// prefix drive an unbounded allocation.
    pub const MAX_HEADER_BYTES: u64 = 8 * 1024 * 1024;
    pub const MAX_TENSORS: usize = 16_384;
    pub const MAX_TENSOR_KEY_BYTES: usize = 1024;
    pub const MAX_TENSOR_RANK: usize = 8;

    /// NVFP4 block width: sixteen weights share one FP8-E4M3 scale.
    pub const NVFP4_BLOCK_SIZE: usize = 16;

    /// Signed E2M1 lookup indexed by 4-bit nibble; exponent bias 1.
    pub const E2M1_LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct HeaderTensor {
        pub dtype: String,
        pub shape: Vec<usize>,
        pub data_offsets: [u64; 2],
    }

    #[derive(Clone, Debug)]
    pub struct ProbeHeader {
        pub header_len: u64,
        pub file_len: u64,
        pub payload_end: u64,
        /// Bytes between the end of the tensor payload and the end of the file.
        /// The published NVFP4 artifact carries a trailing marker here; this
        /// reader tolerates and reports it rather than refusing the file.
        pub trailing_bytes: u64,
        pub tensor_count: usize,
        pub tensors: BTreeMap<String, HeaderTensor>,
    }

    impl ProbeHeader {
        pub fn tensor(&self, name: &str) -> Result<&HeaderTensor, String> {
            self.tensors
                .get(name)
                .ok_or_else(|| format!("missing tensor {name:?}"))
        }
    }

    pub fn dtype_size(dtype: &str) -> Result<u64, String> {
        Ok(match dtype {
            "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
            "I16" | "U16" | "F16" | "BF16" => 2,
            "I32" | "U32" | "F32" => 4,
            "I64" | "U64" | "F64" => 8,
            other => return Err(format!("unsupported safetensors dtype {other:?}")),
        })
    }

    /// A safetensors header object that rejects duplicate top-level keys.
    ///
    /// `serde_json`'s own `Map` silently keeps the last of two identical keys,
    /// which would let one tensor name resolve to a byte range the file's other
    /// readers never see.
    struct StrictHeaderObject(serde_json::Map<String, Value>);

    impl<'de> Deserialize<'de> for StrictHeaderObject {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            struct StrictVisitor;

            impl<'de> Visitor<'de> for StrictVisitor {
                type Value = StrictHeaderObject;

                fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str("a safetensors header object without duplicate keys")
                }

                fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                where
                    A: MapAccess<'de>,
                {
                    let mut object = serde_json::Map::new();
                    while let Some(key) = map.next_key::<String>()? {
                        let value = map.next_value::<Value>()?;
                        if object.insert(key.clone(), value).is_some() {
                            return Err(de::Error::custom(format!(
                                "duplicate safetensors header key {key:?}"
                            )));
                        }
                    }
                    Ok(StrictHeaderObject(object))
                }
            }

            deserializer.deserialize_map(StrictVisitor)
        }
    }

    /// Parse one safetensors header, tolerating trailing non-tensor bytes.
    ///
    /// Every other structural rule stays strict: duplicate keys are refused,
    /// each byte range must match its dtype and shape, and the ranges must tile
    /// the payload contiguously from zero.
    pub fn parse_header(
        header_bytes: &[u8],
        header_len: u64,
        file_len: u64,
    ) -> Result<ProbeHeader, String> {
        let mut deserializer = serde_json::Deserializer::from_slice(header_bytes);
        let object = StrictHeaderObject::deserialize(&mut deserializer)
            .map_err(|error| format!("invalid safetensors header: {error}"))?
            .0;
        deserializer
            .end()
            .map_err(|error| format!("trailing data in safetensors header: {error}"))?;

        let data_len = file_len
            .checked_sub(header_len + 8)
            .ok_or_else(|| "safetensors header exceeds the file".to_string())?;
        let mut tensors = BTreeMap::new();
        for (name, value) in &object {
            if name == "__metadata__" {
                continue;
            }
            if name.is_empty() || name.len() > MAX_TENSOR_KEY_BYTES {
                return Err("safetensors header has an empty or oversized tensor key".into());
            }
            #[derive(Deserialize)]
            struct RawTensor {
                dtype: String,
                shape: Vec<usize>,
                data_offsets: [u64; 2],
            }
            let raw: RawTensor = serde_json::from_value(value.clone())
                .map_err(|error| format!("invalid tensor header {name:?}: {error}"))?;
            if raw.shape.len() > MAX_TENSOR_RANK {
                return Err(format!("tensor {name:?} exceeds the rank bound"));
            }
            if raw.data_offsets[0] > raw.data_offsets[1] || raw.data_offsets[1] > data_len {
                return Err(format!("tensor {name:?} has invalid data offsets"));
            }
            let elements = raw
                .shape
                .iter()
                .try_fold(1u64, |total, dimension| {
                    total.checked_mul(*dimension as u64)
                })
                .ok_or_else(|| format!("tensor {name:?} shape overflows"))?;
            let expected = elements
                .checked_mul(dtype_size(&raw.dtype)?)
                .ok_or_else(|| format!("tensor {name:?} byte size overflows"))?;
            if raw.data_offsets[1] - raw.data_offsets[0] != expected {
                return Err(format!(
                    "tensor {name:?} dtype/shape does not match its byte range"
                ));
            }
            tensors.insert(
                name.clone(),
                HeaderTensor {
                    dtype: raw.dtype,
                    shape: raw.shape,
                    data_offsets: raw.data_offsets,
                },
            );
        }
        if tensors.is_empty() {
            return Err("safetensors header contains no tensors".into());
        }
        if tensors.len() > MAX_TENSORS {
            return Err("safetensors tensor count exceeds the header bound".into());
        }

        let mut ordered: Vec<[u64; 2]> = tensors.values().map(|entry| entry.data_offsets).collect();
        ordered.sort_unstable();
        let mut cursor = 0u64;
        for range in &ordered {
            if range[0] != cursor {
                return Err(format!(
                    "safetensors payload is not contiguous at offset {cursor}"
                ));
            }
            cursor = range[1];
        }

        Ok(ProbeHeader {
            header_len,
            file_len,
            payload_end: cursor,
            trailing_bytes: data_len - cursor,
            tensor_count: tensors.len(),
            tensors,
        })
    }

    pub fn read_header(path: &Path) -> Result<(File, ProbeHeader), String> {
        let mut file =
            File::open(path).map_err(|error| format!("failed to open {path:?}: {error}"))?;
        let file_len = file
            .metadata()
            .map_err(|error| format!("failed to stat {path:?}: {error}"))?
            .len();
        let mut length = [0u8; 8];
        file.read_exact(&mut length)
            .map_err(|error| format!("failed to read the header length of {path:?}: {error}"))?;
        let header_len = u64::from_le_bytes(length);
        if header_len == 0
            || header_len > MAX_HEADER_BYTES
            || header_len > file_len.saturating_sub(8)
        {
            return Err(format!("invalid safetensors header length {header_len}"));
        }
        let mut bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut bytes)
            .map_err(|error| format!("failed to read the header of {path:?}: {error}"))?;
        let header = parse_header(&bytes, header_len, file_len)?;
        Ok((file, header))
    }

    /// Read one tensor's raw payload bytes, verifying its declared dtype and shape.
    pub fn read_tensor_bytes(
        file: &mut File,
        header: &ProbeHeader,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<Vec<u8>, String> {
        let entry = header.tensor(name)?;
        if entry.dtype != dtype {
            return Err(format!(
                "tensor {name:?} has dtype {}, expected {dtype}",
                entry.dtype
            ));
        }
        if entry.shape != shape {
            return Err(format!(
                "tensor {name:?} has shape {:?}, expected {shape:?}",
                entry.shape
            ));
        }
        let start = 8 + header.header_len + entry.data_offsets[0];
        let length = (entry.data_offsets[1] - entry.data_offsets[0]) as usize;
        file.seek(SeekFrom::Start(start))
            .map_err(|error| format!("failed to seek to {name:?}: {error}"))?;
        let mut bytes = vec![0u8; length];
        file.read_exact(&mut bytes)
            .map_err(|error| format!("failed to read {name:?}: {error}"))?;
        Ok(bytes)
    }

    /// Byte-identity of one tensor across two open checkpoints.
    ///
    /// Returns `None` when either file does not carry the name, or carries it
    /// at a different dtype or shape, or when the payload exceeds `max_bytes`.
    pub fn tensors_are_byte_identical(
        left_file: &mut File,
        left: &ProbeHeader,
        right_file: &mut File,
        right: &ProbeHeader,
        name: &str,
        max_bytes: u64,
    ) -> Result<Option<bool>, String> {
        let (Some(left_entry), Some(right_entry)) =
            (left.tensors.get(name), right.tensors.get(name))
        else {
            return Ok(None);
        };
        if left_entry.dtype != right_entry.dtype || left_entry.shape != right_entry.shape {
            return Ok(None);
        }
        if left_entry.data_offsets[1] - left_entry.data_offsets[0] > max_bytes {
            return Ok(None);
        }
        let dtype = left_entry.dtype.clone();
        let shape = left_entry.shape.clone();
        let left_bytes = read_tensor_bytes(left_file, left, name, &dtype, &shape)?;
        let right_bytes = read_tensor_bytes(right_file, right, name, &dtype, &shape)?;
        Ok(Some(left_bytes == right_bytes))
    }

    /// One FP8-E4M3 byte as f32. Exponent bias 7; `S.1111.111` is NaN.
    pub fn e4m3_to_f32(byte: u8) -> f32 {
        let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
        let exponent = ((byte >> 3) & 0x0f) as i32;
        let mantissa_bits = byte & 0x07;
        if exponent == 0x0f && mantissa_bits == 0x07 {
            return f32::NAN;
        }
        let mantissa = mantissa_bits as f32;
        if exponent == 0 {
            sign * (mantissa / 8.0) * (1.0 / 64.0)
        } else {
            sign * (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
        }
    }

    /// Index of the swizzled element that holds logical `(row, column)`.
    ///
    /// This mirrors `comfy_quant::unswizzle_nvfp4_scales` exactly. It is
    /// factored out so the probe can address the swizzled storage by byte as
    /// well as by f32, and so the round trip can be unit-tested.
    pub fn swizzled_index(row: usize, column: usize, column_blocks: usize) -> usize {
        let row_block = row / 128;
        let row_in_block = row % 128;
        let quarter = row_in_block / 32;
        let lane = row_in_block % 32;
        let column_block = column / 4;
        let column_in_block = column % 4;
        let swizzled_column = quarter * 4 + column_in_block;
        let tile = row_block * column_blocks + column_block;
        tile * 512 + lane * 16 + swizzled_column
    }

    pub fn unswizzle<T: Copy + Default>(
        swizzled: &[T],
        logical_rows: usize,
        logical_columns: usize,
    ) -> Result<Vec<T>, String> {
        let row_blocks = logical_rows.div_ceil(128);
        let column_blocks = logical_columns.div_ceil(4);
        let expected = row_blocks * 128 * column_blocks * 4;
        if swizzled.len() != expected {
            return Err(format!(
                "swizzled scale storage has {} elements, expected {expected}",
                swizzled.len()
            ));
        }
        let mut natural = vec![T::default(); logical_rows * logical_columns];
        for row in 0..logical_rows {
            for column in 0..logical_columns {
                natural[row * logical_columns + column] =
                    swizzled[swizzled_index(row, column, column_blocks)];
            }
        }
        Ok(natural)
    }

    pub fn swizzle<T: Copy + Default>(
        natural: &[T],
        logical_rows: usize,
        logical_columns: usize,
    ) -> Result<Vec<T>, String> {
        if natural.len() != logical_rows * logical_columns {
            return Err("natural scale storage has the wrong length".into());
        }
        let row_blocks = logical_rows.div_ceil(128);
        let column_blocks = logical_columns.div_ceil(4);
        let mut swizzled = vec![T::default(); row_blocks * 128 * column_blocks * 4];
        for row in 0..logical_rows {
            for column in 0..logical_columns {
                swizzled[swizzled_index(row, column, column_blocks)] =
                    natural[row * logical_columns + column];
            }
        }
        Ok(swizzled)
    }

    /// Host NVFP4 dequantize for one output-row chunk.
    ///
    /// This mirrors `H3ComfyNvfp4AwqLinear::dequantize_rows` operation for
    /// operation, including the `E2M1 * block_scale * tensor_scale` association
    /// order, so the probe can attribute its cost and compare it bit-for-bit
    /// against a device arm.
    #[allow(clippy::too_many_arguments)]
    pub fn dequantize_rows_host(
        packed: &[u8],
        packed_columns: usize,
        natural_scales: &[f32],
        blocks_per_row: usize,
        tensor_scale: f32,
        start: usize,
        rows: usize,
        in_features: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; rows * in_features];
        for row in 0..rows {
            let packed_row =
                &packed[(start + row) * packed_columns..(start + row + 1) * packed_columns];
            let scales =
                &natural_scales[(start + row) * blocks_per_row..(start + row + 1) * blocks_per_row];
            let target = &mut output[row * in_features..(row + 1) * in_features];
            for (column, slot) in target.iter_mut().enumerate() {
                let byte = packed_row[column / 2];
                let nibble = if column.is_multiple_of(2) {
                    byte >> 4
                } else {
                    byte & 0x0f
                };
                *slot =
                    E2M1_LUT[nibble as usize] * scales[column / NVFP4_BLOCK_SIZE] * tensor_scale;
            }
        }
        output
    }

    /// Deterministic standard-normal activations, independent of any backend RNG.
    pub fn seeded_normal(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Open unit interval: Box-Muller must never see exactly zero.
            ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut values = Vec::with_capacity(count);
        while values.len() < count {
            let u1 = next();
            let u2 = next();
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * u2;
            values.push((radius * angle.cos()) as f32);
            if values.len() < count {
                values.push((radius * angle.sin()) as f32);
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::support::*;

    fn header_bytes(object: &serde_json::Value) -> Vec<u8> {
        serde_json::to_vec(object).unwrap()
    }

    #[test]
    fn trailing_bytes_past_the_payload_are_tolerated_and_reported() {
        let object = serde_json::json!({
            "a": { "dtype": "U8", "shape": [4], "data_offsets": [0, 4] },
        });
        let bytes = header_bytes(&object);
        let header = parse_header(&bytes, bytes.len() as u64, 8 + bytes.len() as u64 + 4 + 65)
            .expect("trailing bytes must be tolerated");
        assert_eq!(header.payload_end, 4);
        assert_eq!(header.trailing_bytes, 65);
        assert_eq!(header.tensor_count, 1);
    }

    #[test]
    fn duplicate_tensor_keys_are_rejected() {
        let raw = br#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let error = parse_header(raw, raw.len() as u64, 8 + raw.len() as u64 + 2)
            .expect_err("duplicate keys must be refused");
        assert!(error.contains("duplicate"), "{error}");
    }

    #[test]
    fn non_contiguous_offsets_are_rejected() {
        let object = serde_json::json!({
            "a": { "dtype": "U8", "shape": [4], "data_offsets": [0, 4] },
            "b": { "dtype": "U8", "shape": [4], "data_offsets": [8, 12] },
        });
        let bytes = header_bytes(&object);
        let error = parse_header(&bytes, bytes.len() as u64, 8 + bytes.len() as u64 + 12)
            .expect_err("a payload gap must be refused");
        assert!(error.contains("not contiguous"), "{error}");
    }

    #[test]
    fn byte_range_must_match_dtype_and_shape() {
        let object = serde_json::json!({
            "a": { "dtype": "F32", "shape": [4], "data_offsets": [0, 8] },
        });
        let bytes = header_bytes(&object);
        let error = parse_header(&bytes, bytes.len() as u64, 8 + bytes.len() as u64 + 8)
            .expect_err("a mismatched byte range must be refused");
        assert!(error.contains("does not match"), "{error}");
    }

    #[test]
    fn swizzle_round_trips_on_a_padded_shape() {
        // 130 rows and 5 blocks force both the 128-row and 4-column padding.
        let rows = 130;
        let columns = 5;
        let natural: Vec<f32> = (0..rows * columns).map(|value| value as f32).collect();
        let swizzled = swizzle(&natural, rows, columns).unwrap();
        assert_eq!(unswizzle(&swizzled, rows, columns).unwrap(), natural);
    }

    #[test]
    fn swizzled_index_is_a_permutation_of_the_tile() {
        let rows = 128;
        let columns = 4;
        let mut seen = vec![false; 512];
        for row in 0..rows {
            for column in 0..columns {
                let index = swizzled_index(row, column, 1);
                assert!(!seen[index], "swizzled index {index} repeats");
                seen[index] = true;
            }
        }
        assert!(seen.into_iter().all(|hit| hit));
    }

    /// The probe's own unswizzle must agree with the shipped
    /// `unswizzle_nvfp4_scales`, which is private, so it is exercised through
    /// the only public path that reaches it: constructing an
    /// `H3ComfyNvfp4AwqLinear` and dequantizing it.
    #[test]
    fn probe_dequantize_matches_the_library_on_a_tiny_shape() {
        use candle::{DType, Device, Tensor};
        use mold_candle::minimax_h3::H3ComfyNvfp4AwqLinear;

        let out_features = 16;
        let in_features = 64;
        let blocks_per_row = in_features / NVFP4_BLOCK_SIZE;
        let packed: Vec<u8> = (0..out_features * in_features / 2)
            .map(|index| (index * 37 % 256) as u8)
            .collect();
        // Positive E4M3 normals only: the library refuses a non-finite or
        // negative block scale, and the padded tail is never read back.
        let scale_bytes: Vec<u8> =
            (0..out_features.div_ceil(128) * 128 * blocks_per_row.div_ceil(4) * 4)
                .map(|index| 0x30 + (index % 16) as u8)
                .collect();
        let tensor_scale = 0.001_358_032_2f32;

        let library = H3ComfyNvfp4AwqLinear::new_with_optional_awq(
            Tensor::from_raw_buffer(
                &packed,
                DType::U8,
                &[out_features, in_features / 2],
                &Device::Cpu,
            )
            .unwrap(),
            Tensor::from_raw_buffer(
                &scale_bytes,
                DType::F8E4M3,
                &[
                    out_features.div_ceil(128) * 128,
                    blocks_per_row.div_ceil(4) * 4,
                ],
                &Device::Cpu,
            )
            .unwrap(),
            Tensor::new(&[tensor_scale], &Device::Cpu).unwrap(),
            None,
            out_features,
            in_features,
        )
        .unwrap();
        let expected = library
            .dequantize_weight(DType::F32, &Device::Cpu, 8)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let natural: Vec<f32> = unswizzle(&scale_bytes, out_features, blocks_per_row)
            .unwrap()
            .iter()
            .map(|byte| e4m3_to_f32(*byte))
            .collect();
        let actual = dequantize_rows_host(
            &packed,
            in_features / 2,
            &natural,
            blocks_per_row,
            tensor_scale,
            0,
            out_features,
            in_features,
        );
        assert_eq!(actual.len(), expected.len());
        for (index, (left, right)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "element {index}: {left} vs {right}"
            );
        }
    }

    #[test]
    fn e4m3_matches_its_defining_arithmetic() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0xb8), -1.0);
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert!(e4m3_to_f32(0x7f).is_nan());
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::path::PathBuf;
    use std::time::Instant;

    use candle::{DType, Device, Tensor};
    use mold_candle::minimax_h3::{
        select_h3_int8_linear_kind, H3ComfyInt8ConvRotLinear, H3ComfyNvfp4AwqLinear,
        H3Int8LinearKind, H3_COMFY_NVFP4_BLOCK_SIZE, H3_COMFY_PORTABLE_ROW_CHUNK,
    };

    use support::{
        dequantize_rows_host, e4m3_to_f32, read_header, read_tensor_bytes, seeded_normal,
        tensors_are_byte_identical, unswizzle, ProbeHeader, E2M1_LUT,
    };

    /// The real per-forward video row count of the reviewed H3 geometry.
    ///
    /// It is `REVIEWED_MAX_TARGET_VIDEO_ROWS` in the private server's
    /// `private_server.rs`, which is not in this repository; the same figure is
    /// the released query length the in-repo attention contract is pinned to
    /// (`minimax_h3/attention.rs`, `h3_query_chunk_rows(.., 37_296)`).
    const REVIEWED_MAX_TARGET_VIDEO_ROWS: usize = 37_296;
    /// Reviewed Turbo 8-step wall clock on this class of card, in seconds.
    const TURBO_8_STEP_BASELINE_SECONDS: f64 = 730.0;
    /// Terminal-inclusive step count of the reviewed Turbo 8-step schedule.
    const TURBO_8_STEP_DENOISE_STEPS: usize = 9;
    const H3_BLOCK_COUNT: usize = 50;
    const QUANTIZED_LINEARS_PER_BLOCK: usize = 4;
    const WARMUP_RUNS: usize = 2;
    const TIMED_RUNS: usize = 5;

    struct LinearShape {
        suffix: &'static str,
        out_features: usize,
        in_features: usize,
    }

    const LINEARS: [LinearShape; 4] = [
        LinearShape {
            suffix: "attn.qkv_proj",
            out_features: 21_504,
            in_features: 5_376,
        },
        LinearShape {
            suffix: "attn.out_proj",
            out_features: 5_376,
            in_features: 7_168,
        },
        LinearShape {
            suffix: "mlp.fc1",
            out_features: 28_672,
            in_features: 5_376,
        },
        LinearShape {
            suffix: "mlp.fc2",
            out_features: 5_376,
            in_features: 14_336,
        },
    ];

    fn relative_frobenius(candidate: &Tensor, reference: &Tensor) -> candle::Result<f64> {
        let difference = (candidate - reference)?;
        let numerator = difference.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        let denominator = reference.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        Ok((numerator.sqrt()) / denominator.sqrt())
    }

    fn row_cosines(candidate: &Tensor, reference: &Tensor) -> candle::Result<(f64, f64)> {
        let dot = (candidate * reference)?.sum(1)?;
        let left = candidate.sqr()?.sum(1)?.sqrt()?;
        let right = reference.sqr()?.sum(1)?.sqrt()?;
        let cosine = dot.div(&(left * right)?)?;
        let mut values = cosine.to_vec1::<f32>()?;
        values.sort_by(f32::total_cmp);
        let minimum = values.first().copied().unwrap_or(f32::NAN) as f64;
        let median = values[values.len() / 2] as f64;
        Ok((minimum, median))
    }

    /// Prototype device-side NVFP4 dequantize: two nibble lookup tables and one
    /// FP8-E4M3 lookup table driven by `index_select`, whose CUDA backend does
    /// accept U8 ids directly (`cuda_backend/mod.rs:459`).
    ///
    /// `id_dtype` selects how the packed bytes are presented to those lookups,
    /// and it is not a free choice. `indexing.cu:60` reserves `max_value<I>()`
    /// as a zero-padding sentinel, so with U8 ids the byte `0xff` silently
    /// resolves to `0.0` instead of the 256th table entry. `0xff` is an
    /// ordinary NVFP4 payload byte (two `-6.0` E2M1 nibbles), so the U8 form is
    /// wrong on real weights; the probe measures both and uses U32.
    #[allow(clippy::too_many_arguments)]
    fn device_dequantize(
        packed: &Tensor,
        scale_bytes: &Tensor,
        high_lut: &Tensor,
        low_lut: &Tensor,
        e4m3_lut: &Tensor,
        tensor_scale: &Tensor,
        out_features: usize,
        in_features: usize,
        id_dtype: DType,
    ) -> candle::Result<Tensor> {
        let blocks_per_row = in_features / H3_COMFY_NVFP4_BLOCK_SIZE;
        let flat = packed.flatten_all()?.to_dtype(id_dtype)?;
        let high = high_lut.index_select(&flat, 0)?;
        let low = low_lut.index_select(&flat, 0)?;
        let scales = e4m3_lut
            .index_select(&scale_bytes.flatten_all()?.to_dtype(id_dtype)?, 0)?
            .reshape((out_features, blocks_per_row, 1))?;
        Tensor::stack(&[&high, &low], 1)?
            .reshape((out_features, blocks_per_row, H3_COMFY_NVFP4_BLOCK_SIZE))?
            .broadcast_mul(&scales)?
            .reshape((out_features, in_features))?
            .broadcast_mul(tensor_scale)
    }

    /// Fastest and mean wall clock over `TIMED_RUNS`, in milliseconds.
    ///
    /// The fastest run is the headline figure everywhere below. This box is a
    /// shared development machine, so a run can be preempted by an unrelated
    /// compile; the minimum is the observation least contaminated by that,
    /// while the mean is retained so the spread stays visible.
    fn time_runs<F>(device: &Device, mut run: F) -> Result<(f64, f64), Box<dyn std::error::Error>>
    where
        F: FnMut() -> candle::Result<()>,
    {
        for _ in 0..WARMUP_RUNS {
            run()?;
        }
        device.synchronize()?;
        let mut total = 0.0;
        let mut fastest = f64::INFINITY;
        for _ in 0..TIMED_RUNS {
            let started = Instant::now();
            run()?;
            device.synchronize()?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            total += elapsed;
            fastest = fastest.min(elapsed);
        }
        Ok((fastest, total / TIMED_RUNS as f64))
    }

    let mut positional: Vec<String> = Vec::new();
    let mut rows = 4_096usize;
    let mut cost_rows = REVIEWED_MAX_TARGET_VIDEO_ROWS;
    let mut blocks = vec![0usize, 25, 49];
    let mut gpu = 0usize;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--rows" => rows = arguments.next().ok_or("--rows needs a value")?.parse()?,
            "--cost-rows" => {
                cost_rows = arguments
                    .next()
                    .ok_or("--cost-rows needs a value")?
                    .parse()?;
            }
            "--gpu" => gpu = arguments.next().ok_or("--gpu needs a value")?.parse()?,
            "--blocks" => {
                blocks = arguments
                    .next()
                    .ok_or("--blocks needs a value")?
                    .split(',')
                    .map(str::parse::<usize>)
                    .collect::<Result<Vec<_>, _>>()?;
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other}").into()),
            other => positional.push(other.to_string()),
        }
    }
    if positional.len() != 3 {
        return Err("usage: h3_nvfp4_layout_probe <nvfp4.safetensors> <int8_convrot.safetensors> <pruned_bf16.safetensors> [--rows N] [--cost-rows N] [--blocks 0,25,49] [--gpu N]".into());
    }
    if rows == 0 || cost_rows == 0 || blocks.is_empty() {
        return Err("row counts and the block list must be nonempty".into());
    }
    if blocks.iter().any(|block| *block >= H3_BLOCK_COUNT) {
        return Err(format!("block indices must be below {H3_BLOCK_COUNT}").into());
    }

    let paths: Vec<PathBuf> = positional.iter().map(PathBuf::from).collect();
    let open = |path: &PathBuf| -> Result<(File, ProbeHeader), Box<dyn std::error::Error>> {
        read_header(path).map_err(std::convert::Into::into)
    };
    let (mut nvfp4_file, nvfp4_header) = open(&paths[0])?;
    let (mut int8_file, int8_header) = open(&paths[1])?;
    let (mut bf16_file, bf16_header) = open(&paths[2])?;

    let describe = |path: &PathBuf, header: &ProbeHeader| {
        serde_json::json!({
            "path": path.display().to_string(),
            "file_len": header.file_len,
            "header_len": header.header_len,
            "tensor_count": header.tensor_count,
            "payload_end": header.payload_end,
            "trailing_bytes": header.trailing_bytes,
        })
    };

    let device = Device::new_cuda(gpu)?;

    // --- self checks -------------------------------------------------------
    // The FP8-E4M3 lookup table this probe uses to unswizzle scale bytes must
    // agree with candle's own cast on every one of the 256 byte values, and the
    // local swizzle index math must reproduce the library's unconditional
    // unswizzle on a real H3-shaped tensor. Both are asserted, not assumed.
    let all_bytes: Vec<u8> = (0..=255u8).collect();
    let candle_e4m3 = Tensor::from_raw_buffer(&all_bytes, DType::F8E4M3, &[256], &Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut e4m3_lut_values = vec![0.0f32; 256];
    let mut e4m3_agreements = 0usize;
    for (index, byte) in all_bytes.iter().enumerate() {
        let local = e4m3_to_f32(*byte);
        e4m3_lut_values[index] = if local.is_nan() { 0.0 } else { local };
        let reference = candle_e4m3[index];
        if (local.is_nan() && reference.is_nan()) || local.to_bits() == reference.to_bits() {
            e4m3_agreements += 1;
        }
    }
    if e4m3_agreements != 256 {
        return Err(format!(
            "the local FP8-E4M3 table disagrees with candle on {} of 256 bytes",
            256 - e4m3_agreements
        )
        .into());
    }

    // --- shared-base evidence ---------------------------------------------
    // The divergence section reads the BF16 file as the common base both
    // quantizations were derived from. That is an assumption about three files
    // from two publishers, so it is checked rather than asserted: every tensor
    // the three carry at one name, dtype, and shape must be byte-identical.
    // Large enough to cover the two un-quantized `token_refiner` blocks, whose
    // BF16 `mlp.fc1` is 294 MiB, so nothing shared is skipped for size.
    const SHARED_TENSOR_MAX_BYTES: u64 = 512 * 1024 * 1024;
    let mut shared_candidates = 0usize;
    let mut shared_identical = 0usize;
    let shared_names: Vec<String> = nvfp4_header
        .tensors
        .keys()
        .filter(|name| int8_header.tensors.contains_key(*name))
        .filter(|name| bf16_header.tensors.contains_key(*name))
        .cloned()
        .collect();
    for name in &shared_names {
        let against_int8 = tensors_are_byte_identical(
            &mut nvfp4_file,
            &nvfp4_header,
            &mut int8_file,
            &int8_header,
            name,
            SHARED_TENSOR_MAX_BYTES,
        )?;
        let against_bf16 = tensors_are_byte_identical(
            &mut nvfp4_file,
            &nvfp4_header,
            &mut bf16_file,
            &bf16_header,
            name,
            SHARED_TENSOR_MAX_BYTES,
        )?;
        // A quantized weight differs in dtype and shape and answers `None`;
        // only tensors all three carry unchanged are candidates.
        if let (Some(int8_same), Some(bf16_same)) = (against_int8, against_bf16) {
            shared_candidates += 1;
            if int8_same && bf16_same {
                shared_identical += 1;
            }
        }
    }

    // --- shared LUT tensors for the prototype device arm -------------------
    let high_lut = Tensor::from_vec(
        (0..=255u8)
            .map(|byte| E2M1_LUT[(byte >> 4) as usize])
            .collect::<Vec<f32>>(),
        256,
        &device,
    )?;
    let low_lut = Tensor::from_vec(
        (0..=255u8)
            .map(|byte| E2M1_LUT[(byte & 0x0f) as usize])
            .collect::<Vec<f32>>(),
        256,
        &device,
    )?;
    let e4m3_lut = Tensor::from_vec(e4m3_lut_values.clone(), 256, &device)?;

    struct Loaded {
        packed: Vec<u8>,
        swizzled_scale_bytes: Vec<u8>,
        natural_scale_bytes: Vec<u8>,
        natural_scales: Vec<f32>,
        raw_scales: Vec<f32>,
        tensor_scale: f32,
        int8_weight: Vec<u8>,
        int8_scale: Vec<u8>,
        bf16_weight: Vec<u8>,
    }

    let load = |nvfp4_file: &mut File,
                int8_file: &mut File,
                bf16_file: &mut File,
                block: usize,
                shape: &LinearShape|
     -> Result<Loaded, Box<dyn std::error::Error>> {
        let base = format!("blocks.{block}.{}", shape.suffix);
        let blocks_per_row = shape.in_features / H3_COMFY_NVFP4_BLOCK_SIZE;
        let scale_rows = shape.out_features.div_ceil(128) * 128;
        let scale_columns = blocks_per_row.div_ceil(4) * 4;
        let packed = read_tensor_bytes(
            nvfp4_file,
            &nvfp4_header,
            &format!("{base}.weight"),
            "U8",
            &[shape.out_features, shape.in_features / 2],
        )?;
        let swizzled_scale_bytes = read_tensor_bytes(
            nvfp4_file,
            &nvfp4_header,
            &format!("{base}.weight_scale"),
            "F8_E4M3",
            &[scale_rows, scale_columns],
        )?;
        let tensor_scale_bytes = read_tensor_bytes(
            nvfp4_file,
            &nvfp4_header,
            &format!("{base}.weight_scale_2"),
            "F32",
            &[],
        )?;
        let tensor_scale = f32::from_le_bytes(
            tensor_scale_bytes
                .as_slice()
                .try_into()
                .map_err(|_| "weight_scale_2 must be four bytes")?,
        );
        let natural_scale_bytes =
            unswizzle(&swizzled_scale_bytes, shape.out_features, blocks_per_row)?;
        let natural_scales: Vec<f32> = natural_scale_bytes
            .iter()
            .map(|byte| e4m3_to_f32(*byte))
            .collect();
        // The "natural" hypothesis reads the stored scale block as already
        // being in logical [out, blocks] order, dropping the padded tail.
        let raw_scales: Vec<f32> = (0..shape.out_features)
            .flat_map(|row| (0..blocks_per_row).map(move |column| (row, column)))
            .map(|(row, column)| e4m3_to_f32(swizzled_scale_bytes[row * scale_columns + column]))
            .collect();
        let int8_weight = read_tensor_bytes(
            int8_file,
            &int8_header,
            &format!("{base}.weight"),
            "I8",
            &[shape.out_features, shape.in_features],
        )?;
        let int8_scale = read_tensor_bytes(
            int8_file,
            &int8_header,
            &format!("{base}.weight_scale"),
            "F32",
            &[shape.out_features, 1],
        )?;
        let bf16_weight = read_tensor_bytes(
            bf16_file,
            &bf16_header,
            &format!("{base}.weight"),
            "BF16",
            &[shape.out_features, shape.in_features],
        )?;
        Ok(Loaded {
            packed,
            swizzled_scale_bytes,
            natural_scale_bytes,
            natural_scales,
            raw_scales,
            tensor_scale,
            int8_weight,
            int8_scale,
            bf16_weight,
        })
    };

    let mut layout_rows = Vec::new();
    let mut divergence_rows = Vec::new();
    let mut activation_rows = Vec::new();
    let mut swizzled_total = 0.0f64;
    let mut natural_total = 0.0f64;
    let mut library_parity_max = 0.0f64;
    let mut probed = 0usize;

    for block in &blocks {
        for shape in &LINEARS {
            let loaded = load(
                &mut nvfp4_file,
                &mut int8_file,
                &mut bf16_file,
                *block,
                shape,
            )?;
            let blocks_per_row = shape.in_features / H3_COMFY_NVFP4_BLOCK_SIZE;
            let packed_columns = shape.in_features / 2;

            let bf16 = Tensor::from_raw_buffer(
                &loaded.bf16_weight,
                DType::BF16,
                &[shape.out_features, shape.in_features],
                &device,
            )?
            .to_dtype(DType::F32)?;

            // Hypothesis A: the stored scales are swizzled (the library's rule).
            let swizzled_dense = {
                let host = dequantize_rows_host(
                    &loaded.packed,
                    packed_columns,
                    &loaded.natural_scales,
                    blocks_per_row,
                    loaded.tensor_scale,
                    0,
                    shape.out_features,
                    shape.in_features,
                );
                Tensor::from_vec(host, (shape.out_features, shape.in_features), &device)?
            };
            // Hypothesis B: the stored scales are already in natural order.
            let natural_dense = {
                let host = dequantize_rows_host(
                    &loaded.packed,
                    packed_columns,
                    &loaded.raw_scales,
                    blocks_per_row,
                    loaded.tensor_scale,
                    0,
                    shape.out_features,
                    shape.in_features,
                );
                Tensor::from_vec(host, (shape.out_features, shape.in_features), &device)?
            };

            // Cross-check the probe's own index math against the shipped
            // `H3ComfyNvfp4AwqLinear`, whose construction unswizzles
            // unconditionally, on this exact H3-shaped tensor.
            let library = H3ComfyNvfp4AwqLinear::new_with_optional_awq(
                Tensor::from_raw_buffer(
                    &loaded.packed,
                    DType::U8,
                    &[shape.out_features, packed_columns],
                    &Device::Cpu,
                )?,
                Tensor::from_raw_buffer(
                    &loaded.swizzled_scale_bytes,
                    DType::F8E4M3,
                    &[
                        shape.out_features.div_ceil(128) * 128,
                        blocks_per_row.div_ceil(4) * 4,
                    ],
                    &Device::Cpu,
                )?,
                Tensor::new(&[loaded.tensor_scale], &Device::Cpu)?,
                None,
                shape.out_features,
                shape.in_features,
            )?;
            let library_dense =
                library.dequantize_weight(DType::F32, &device, H3_COMFY_PORTABLE_ROW_CHUNK)?;
            let parity = (&library_dense - &swizzled_dense)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()? as f64;
            library_parity_max = library_parity_max.max(parity);
            drop(library_dense);
            drop(library);

            let int8 = H3ComfyInt8ConvRotLinear::new(
                Tensor::from_raw_buffer(
                    &loaded.int8_weight,
                    DType::U8,
                    &[shape.out_features, shape.in_features],
                    &Device::Cpu,
                )?,
                Tensor::from_raw_buffer(
                    &loaded.int8_scale,
                    DType::F32,
                    &[shape.out_features, 1],
                    &Device::Cpu,
                )?,
            )?;
            let int8_dense =
                int8.dequantize_weight(DType::F32, &device, H3_COMFY_PORTABLE_ROW_CHUNK)?;

            let swizzled_error = relative_frobenius(&swizzled_dense, &bf16)?;
            let natural_error = relative_frobenius(&natural_dense, &bf16)?;
            let int8_error = relative_frobenius(&int8_dense, &bf16)?;
            let nvfp4_vs_int8 = relative_frobenius(&swizzled_dense, &int8_dense)?;
            let (cosine_min, cosine_median) = row_cosines(&swizzled_dense, &bf16)?;
            let (int8_cosine_min, int8_cosine_median) = row_cosines(&int8_dense, &bf16)?;
            swizzled_total += swizzled_error;
            natural_total += natural_error;
            probed += 1;

            layout_rows.push(serde_json::json!({
                "block": block,
                "linear": shape.suffix,
                "out_features": shape.out_features,
                "in_features": shape.in_features,
                "swizzled_hypothesis_vs_bf16_rel_fro": swizzled_error,
                "natural_hypothesis_vs_bf16_rel_fro": natural_error,
                "swizzled_hypothesis_vs_int8_rel_fro": nvfp4_vs_int8,
                "library_unswizzle_max_abs_delta": parity,
            }));
            divergence_rows.push(serde_json::json!({
                "block": block,
                "linear": shape.suffix,
                "nvfp4_vs_bf16_rel_fro": swizzled_error,
                "int8_vs_bf16_rel_fro": int8_error,
                "nvfp4_vs_int8_rel_fro": nvfp4_vs_int8,
                "nvfp4_row_cosine_min": cosine_min,
                "nvfp4_row_cosine_p50": cosine_median,
                "int8_row_cosine_min": int8_cosine_min,
                "int8_row_cosine_p50": int8_cosine_median,
            }));
            drop(natural_dense);

            // --- activation space ------------------------------------------
            let activations = Tensor::from_vec(
                seeded_normal(
                    0x0000_0517 ^ ((*block as u64) << 8) ^ probed as u64,
                    rows * shape.in_features,
                ),
                (rows, shape.in_features),
                &device,
            )?;
            let reference_output = activations.matmul(&bf16.t()?.contiguous()?)?;
            let nvfp4_output = {
                let mut chunks = Vec::new();
                for start in (0..shape.out_features).step_by(H3_COMFY_PORTABLE_ROW_CHUNK) {
                    let width = H3_COMFY_PORTABLE_ROW_CHUNK.min(shape.out_features - start);
                    let weight = swizzled_dense.narrow(0, start, width)?;
                    chunks.push(activations.matmul(&weight.t()?.contiguous()?)?);
                }
                Tensor::cat(&chunks, 1)?
            };
            let nvfp4_activation_error = relative_frobenius(&nvfp4_output, &reference_output)?;
            drop(nvfp4_output);
            let int8_output = int8.forward_reference(
                &activations,
                None,
                DType::F32,
                H3_COMFY_PORTABLE_ROW_CHUNK,
            )?;
            let int8_activation_error = relative_frobenius(&int8_output, &reference_output)?;
            drop(int8_output);
            drop(reference_output);
            drop(activations);

            activation_rows.push(serde_json::json!({
                "block": block,
                "linear": shape.suffix,
                "rows": rows,
                "nvfp4_vs_bf16_rel_l2": nvfp4_activation_error,
                "int8_vs_bf16_rel_l2": int8_activation_error,
                "nvfp4_over_int8": nvfp4_activation_error / int8_activation_error,
            }));

            drop(swizzled_dense);
            drop(int8_dense);
            drop(bf16);
            drop(int8);
        }
    }

    let swizzled_mean = swizzled_total / probed as f64;
    let natural_mean = natural_total / probed as f64;
    let verdict = if swizzled_mean < 0.5 && natural_mean > 0.5 {
        "swizzled"
    } else if natural_mean < 0.5 && swizzled_mean > 0.5 {
        "natural"
    } else {
        "ambiguous"
    };

    let worst_activation_ratio = activation_rows
        .iter()
        .filter_map(|row| row["nvfp4_over_int8"].as_f64())
        .fold(0.0f64, f64::max);

    // --- cost --------------------------------------------------------------
    let cost_shape = &LINEARS[0];
    let cost = load(
        &mut nvfp4_file,
        &mut int8_file,
        &mut bf16_file,
        0,
        cost_shape,
    )?;
    let blocks_per_row = cost_shape.in_features / H3_COMFY_NVFP4_BLOCK_SIZE;
    let packed_columns = cost_shape.in_features / 2;
    let cost_activations = Tensor::from_vec(
        seeded_normal(0x00C0_5701, cost_rows * cost_shape.in_features),
        (cost_rows, cost_shape.in_features),
        &device,
    )?;

    let nvfp4_linear = H3ComfyNvfp4AwqLinear::new_with_optional_awq(
        Tensor::from_raw_buffer(
            &cost.packed,
            DType::U8,
            &[cost_shape.out_features, packed_columns],
            &Device::Cpu,
        )?,
        Tensor::from_raw_buffer(
            &cost.swizzled_scale_bytes,
            DType::F8E4M3,
            &[
                cost_shape.out_features.div_ceil(128) * 128,
                blocks_per_row.div_ceil(4) * 4,
            ],
            &Device::Cpu,
        )?,
        Tensor::new(&[cost.tensor_scale], &Device::Cpu)?,
        None,
        cost_shape.out_features,
        cost_shape.in_features,
    )?;
    let int8_linear = H3ComfyInt8ConvRotLinear::new(
        Tensor::from_raw_buffer(
            &cost.int8_weight,
            DType::U8,
            &[cost_shape.out_features, cost_shape.in_features],
            &Device::Cpu,
        )?,
        Tensor::from_raw_buffer(
            &cost.int8_scale,
            DType::F32,
            &[cost_shape.out_features, 1],
            &Device::Cpu,
        )?,
    )?;
    let int8_kind = select_h3_int8_linear_kind(
        &device,
        cfg!(feature = "cuda"),
        false,
        cost_shape.in_features,
        cost_shape.out_features,
    );

    let (nvfp4_forward_ms, nvfp4_forward_ms_mean) = time_runs(&device, || {
        nvfp4_linear
            .forward_dequantized(
                &cost_activations,
                None,
                DType::F32,
                H3_COMFY_PORTABLE_ROW_CHUNK,
            )
            .map(|_| ())
    })?;
    let (int8_forward_ms, int8_forward_ms_mean) = time_runs(&device, || {
        int8_linear
            .forward_reference(
                &cost_activations,
                None,
                DType::F32,
                H3_COMFY_PORTABLE_ROW_CHUNK,
            )
            .map(|_| ())
    })?;

    // Attribution: a local copy of the shipped chunk loop, with a device
    // synchronize between each stage so the three costs are separable.
    let mut host_ms = f64::INFINITY;
    let mut upload_ms = f64::INFINITY;
    let mut matmul_ms = f64::INFINITY;
    for run in 0..(WARMUP_RUNS + TIMED_RUNS) {
        let timed = run >= WARMUP_RUNS;
        let mut run_host = 0.0;
        let mut run_upload = 0.0;
        let mut run_matmul = 0.0;
        let mut chunks = Vec::new();
        for start in (0..cost_shape.out_features).step_by(H3_COMFY_PORTABLE_ROW_CHUNK) {
            let width = H3_COMFY_PORTABLE_ROW_CHUNK.min(cost_shape.out_features - start);
            let began = Instant::now();
            let host = dequantize_rows_host(
                &cost.packed,
                packed_columns,
                &cost.natural_scales,
                blocks_per_row,
                cost.tensor_scale,
                start,
                width,
                cost_shape.in_features,
            );
            let after_host = Instant::now();
            let weight = Tensor::from_vec(host, (width, cost_shape.in_features), &device)?;
            device.synchronize()?;
            let after_upload = Instant::now();
            chunks.push(cost_activations.matmul(&weight.t()?.contiguous()?)?);
            device.synchronize()?;
            let after_matmul = Instant::now();
            if timed {
                run_host += after_host.duration_since(began).as_secs_f64() * 1000.0;
                run_upload += after_upload.duration_since(after_host).as_secs_f64() * 1000.0;
                run_matmul += after_matmul.duration_since(after_upload).as_secs_f64() * 1000.0;
            }
        }
        drop(Tensor::cat(&chunks, 1)?);
        if timed {
            host_ms = host_ms.min(run_host);
            upload_ms = upload_ms.min(run_upload);
            matmul_ms = matmul_ms.min(run_matmul);
        }
    }

    // --- prototype device arm ---------------------------------------------
    let packed_device = Tensor::from_raw_buffer(
        &cost.packed,
        DType::U8,
        &[cost_shape.out_features, packed_columns],
        &device,
    )?;
    let scale_bytes_device = Tensor::from_raw_buffer(
        &cost.natural_scale_bytes,
        DType::U8,
        &[cost_shape.out_features, blocks_per_row],
        &device,
    )?;
    let tensor_scale_device = Tensor::new(&[[cost.tensor_scale]], &device)?;
    let prototype_dense = device_dequantize(
        &packed_device,
        &scale_bytes_device,
        &high_lut,
        &low_lut,
        &e4m3_lut,
        &tensor_scale_device,
        cost_shape.out_features,
        cost_shape.in_features,
        DType::U32,
    )?;
    let host_dense = dequantize_rows_host(
        &cost.packed,
        packed_columns,
        &cost.natural_scales,
        blocks_per_row,
        cost.tensor_scale,
        0,
        cost_shape.out_features,
        cost_shape.in_features,
    );
    let prototype_host = prototype_dense.flatten_all()?.to_vec1::<f32>()?;
    let mismatched = prototype_host
        .iter()
        .zip(&host_dense)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    if mismatched != 0 {
        return Err(format!(
            "the prototype device dequantize differs from the host loop in {mismatched} of {} elements",
            host_dense.len()
        )
        .into());
    }
    drop(prototype_host);
    drop(prototype_dense);

    // Measured negative result: the same arm with U8 ids, which is what #1317
    // proposed. `indexing.cu:60` zeroes any id equal to `max_value<I>()`, so
    // every `0xff` payload byte decodes to `0.0`.
    let u8_ids_dense = device_dequantize(
        &packed_device,
        &scale_bytes_device,
        &high_lut,
        &low_lut,
        &e4m3_lut,
        &tensor_scale_device,
        cost_shape.out_features,
        cost_shape.in_features,
        DType::U8,
    )?;
    let u8_ids_host = u8_ids_dense.flatten_all()?.to_vec1::<f32>()?;
    let u8_mismatched = u8_ids_host
        .iter()
        .zip(&host_dense)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    let u8_sentinel_bytes = cost.packed.iter().filter(|byte| **byte == u8::MAX).count();
    drop(u8_ids_host);
    drop(u8_ids_dense);
    drop(host_dense);

    let (prototype_ms, prototype_ms_mean) = time_runs(&device, || {
        let weight = device_dequantize(
            &packed_device,
            &scale_bytes_device,
            &high_lut,
            &low_lut,
            &e4m3_lut,
            &tensor_scale_device,
            cost_shape.out_features,
            cost_shape.in_features,
            DType::U32,
        )?;
        cost_activations
            .matmul(&weight.t()?.contiguous()?)
            .map(|_| ())
    })?;
    let (prototype_dequantize_only_ms, prototype_dequantize_only_ms_mean) =
        time_runs(&device, || {
            device_dequantize(
                &packed_device,
                &scale_bytes_device,
                &high_lut,
                &low_lut,
                &e4m3_lut,
                &tensor_scale_device,
                cost_shape.out_features,
                cost_shape.in_features,
                DType::U32,
            )
            .map(|_| ())
        })?;
    // #1317's part 2 casts the dequantized weight to BF16 before the matmul.
    // That is a different GEMM, not a different dequantize, so it is timed
    // separately rather than folded into the F32 number above.
    let cost_activations_bf16 = cost_activations.to_dtype(DType::BF16)?;
    let (prototype_bf16_ms, prototype_bf16_ms_mean) = time_runs(&device, || {
        let weight = device_dequantize(
            &packed_device,
            &scale_bytes_device,
            &high_lut,
            &low_lut,
            &e4m3_lut,
            &tensor_scale_device,
            cost_shape.out_features,
            cost_shape.in_features,
            DType::U32,
        )?
        .to_dtype(DType::BF16)?;
        cost_activations_bf16
            .matmul(&weight.t()?.contiguous()?)
            .map(|_| ())
    })?;

    let per_denoise = |per_linear_ms: f64| {
        per_linear_ms
            * QUANTIZED_LINEARS_PER_BLOCK as f64
            * H3_BLOCK_COUNT as f64
            * TURBO_8_STEP_DENOISE_STEPS as f64
            / 1000.0
    };

    let report = serde_json::json!({
        "schema": "mold.minimax-h3.nvfp4-layout-probe.v1",
        "inputs": {
            "nvfp4": describe(&paths[0], &nvfp4_header),
            "int8_convrot": describe(&paths[1], &int8_header),
            "pruned_bf16": describe(&paths[2], &bf16_header),
        },
        "self_checks": {
            "shared_base": {
                "claim": "the NVFP4 and INT8 ConvRot artifacts are requantizations of this exact pruned BF16 base",
                "tensors_carried_by_all_three_at_one_dtype_and_shape": shared_candidates,
                "of_those_byte_identical_in_all_three": shared_identical,
            },
            "e4m3_table_bytes_agreeing_with_candle": e4m3_agreements,
            "local_unswizzle_vs_library_max_abs_delta": library_parity_max,
            "prototype_device_dequantize_bit_equal_to_host_loop": true,
            "candle_u8_index_select_sentinel": {
                "finding": "candle-kernels/src/indexing.cu:60 zeroes any index_select id equal to max_value<I>(), so a 256-entry F32 lookup table driven by U8 ids can never return its 256th entry",
                "affects": "0xff is an ordinary NVFP4 payload byte: two -6.0 E2M1 nibbles",
                "sentinel_bytes_in_probed_weight": u8_sentinel_bytes,
                "u8_id_arm_mismatched_elements": u8_mismatched,
                "u8_id_arm_total_elements": cost_shape.out_features * cost_shape.in_features,
                "resolution": "the prototype casts the packed bytes to U32 ids on the device; the packed weight itself stays U8 at rest",
            },
        },
        "scale_layout": {
            "question": "are comfy-kitchen NVFP4 block scales stored swizzled on H3-shaped tensors?",
            "note": "every probed linear has out % 128 == 0 and in/16 % 4 == 0, so the swizzled and natural scale shapes coincide and the header cannot answer",
            "per_linear": layout_rows,
            "swizzled_hypothesis_mean_rel_fro": swizzled_mean,
            "natural_hypothesis_mean_rel_fro": natural_mean,
            "verdict": verdict,
        },
        "divergence": {
            "reference": "pruned BF16 FL2VA, the common base both quantizations were derived from",
            "weight_space": divergence_rows,
            "activation_space": {
                "rows": rows,
                "extrapolation_basis_rows": REVIEWED_MAX_TARGET_VIDEO_ROWS,
                "distribution": "fixed-seed N(0,1)",
                "per_linear": activation_rows,
            },
            "go_no_go_rule": "if nvfp4_vs_bf16 > 2 x int8_vs_bf16 in activation space, the layout is not worth shipping as-is",
            "worst_nvfp4_over_int8": worst_activation_ratio,
            "go_no_go": if worst_activation_ratio > 2.0 { "no-go" } else { "go" },
        },
        "cost": {
            "linear": format!("blocks.0.{}", cost_shape.suffix),
            "out_features": cost_shape.out_features,
            "in_features": cost_shape.in_features,
            "rows": cost_rows,
            "reviewed_max_target_video_rows": REVIEWED_MAX_TARGET_VIDEO_ROWS,
            "warmup_runs": WARMUP_RUNS,
            "timed_runs": TIMED_RUNS,
            "statistic": "fastest of the timed runs; this is a shared development machine, so the mean is reported alongside each figure to expose contention",
            "loadavg_1m_at_report": std::fs::read_to_string("/proc/loadavg")
                .ok()
                .and_then(|line| line.split_whitespace().next().and_then(|value| value.parse::<f64>().ok())),
            "int8_linear_kind": format!("{int8_kind:?}"),
            "int8_native_cublaslt": int8_kind == H3Int8LinearKind::NativeCudaInt8,
            "nvfp4_forward_dequantized_ms": nvfp4_forward_ms,
            "nvfp4_forward_dequantized_mean_ms": nvfp4_forward_ms_mean,
            "nvfp4_attribution_ms": {
                "note": "fastest run; a device synchronize separates each stage, so these three serialize work the shipped forward overlaps: their sum therefore exceeds nvfp4_forward_dequantized_ms and each figure is an upper bound on that stage alone",
                "host_dequantize": host_ms,
                "host_to_device": upload_ms,
                "matmul": matmul_ms,
            },
            "int8_forward_reference_ms": int8_forward_ms,
            "int8_forward_reference_mean_ms": int8_forward_ms_mean,
            "prototype_device_dequantize_ms": prototype_ms,
            "prototype_device_dequantize_mean_ms": prototype_ms_mean,
            "prototype_device_dequantize_without_matmul_ms": prototype_dequantize_only_ms,
            "prototype_device_dequantize_without_matmul_mean_ms": prototype_dequantize_only_ms_mean,
            "prototype_device_dequantize_bf16_matmul_ms": prototype_bf16_ms,
            "prototype_device_dequantize_bf16_matmul_mean_ms": prototype_bf16_ms_mean,
            "extrapolated_denoise_seconds": {
                "basis": "per_linear_ms x 4 linears x 50 blocks x 9 terminal-inclusive steps",
                "nvfp4_host_dequantize": per_denoise(nvfp4_forward_ms),
                "int8_native": per_denoise(int8_forward_ms),
                "prototype_device_dequantize": per_denoise(prototype_ms),
                "prototype_device_dequantize_bf16_matmul": per_denoise(prototype_bf16_ms),
                "turbo_8_step_baseline_seconds": TURBO_8_STEP_BASELINE_SECONDS,
            },
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("h3_nvfp4_layout_probe requires --features cuda");
    std::process::exit(2);
}
