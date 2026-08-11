//! Moving one Wan transformer block's quantized weights between host and
//! device, losslessly (#776 item 3).
//!
//! An 81-frame A14B q5 render extrapolates to ~25,461 MiB against a 24,564 MiB
//! card (see `mold-server`'s `wan_admission` calibration), and one of the 40
//! blocks is ~270 MiB at Q5 — so parking a handful of blocks in host RAM is
//! what closes the gap, rather than the stream-everything mode
//! `flux/offload.rs` runs.
//!
//! ## Why this is a byte move and not a dtype round trip
//!
//! The obvious spelling — dequantize to F32, move, re-quantize — is lossy, and
//! silently so: `QTensor::quantize_onto` takes an F32 source, so a parked block
//! would come back subtly different from the one the checkpoint shipped and the
//! render would depend on how many blocks happened to be parked. candle exposes
//! the two halves needed to avoid that entirely:
//!
//! - [`QTensor::data`] hands back the raw quantized bytes, and its CUDA arm is
//!   implemented (`candle-core/src/quantized/mod.rs:249`), so this works from a
//!   resident block rather than only from a file.
//! - [`gguf_file`]'s sibling `ggml_file::qtensor_from_ggml` rebuilds a
//!   `QTensor` on **any** device from those bytes.
//!
//! Composed, a park/unpark cycle is a memcpy in each direction. The bytes that
//! come back are the bytes that went out, which is what
//! [`tests::parking_a_block_is_byte_identical`] asserts.
//!
//! ## Why it works at the tensor level rather than on the block struct
//!
//! `WanLinear` is `Arc<dyn Module + Send + Sync>`, so a block cannot be asked to
//! change device — there is no `to_device` on a trait object and no way to write
//! one generically. Rather than de-erase that type (which would mean a
//! `to_device` arm for the plain, GGUF, fp8, `CastBoundary`, and LoRA-branch
//! variants, and touching every construction site in the family), this module
//! moves the *tensors* and lets the existing, already-correct
//! `WanBlock::load` path rebuild the block from them. That path is the one the
//! initial load uses, so the quantized source and the LoRA branch compose here
//! exactly as they already do.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::{ggml_file, QTensor};
use candle_core::{Device, Result};

/// Checkpoint-name prefix for the tensors belonging to block `index`.
///
/// Both the diffusers export and the shipped GGUF files use a flat
/// `blocks.{i}.` namespace, which is what makes a block's weight set
/// identifiable without a second manifest.
pub(crate) fn block_prefix(index: usize) -> String {
    format!("blocks.{index}.")
}

/// Move one quantized tensor to `device` without changing a byte.
///
/// A tensor already on the target device is returned as a cheap `Arc` clone
/// rather than copied through host memory.
pub(crate) fn qtensor_to_device(src: &Arc<QTensor>, device: &Device) -> Result<Arc<QTensor>> {
    if same_device(&src.device(), device) {
        return Ok(src.clone());
    }
    let dims = src.shape().dims().to_vec();
    let dtype = src.dtype();
    let bytes = src.data()?;
    Ok(Arc::new(ggml_file::qtensor_from_ggml(
        dtype, &bytes, dims, device,
    )?))
}

/// `Device` has no `PartialEq`, and `same_device` is the comparison candle
/// itself uses — location and ordinal, not identity.
fn same_device(a: &Device, b: &Device) -> bool {
    match (a, b) {
        (Device::Cpu, Device::Cpu) => true,
        (Device::Cuda(_), Device::Cuda(_)) | (Device::Metal(_), Device::Metal(_)) => {
            a.same_device(b)
        }
        _ => false,
    }
}

/// One block's weights, held wherever [`WanBlockParking::park`] put them.
#[derive(Clone)]
pub(crate) struct ParkedBlock {
    index: usize,
    /// Keyed by the tensor's full checkpoint name, so the set can be handed
    /// straight back to a var builder without re-deriving names.
    tensors: HashMap<String, Arc<QTensor>>,
}

impl ParkedBlock {
    pub(crate) fn index(&self) -> usize {
        self.index
    }

    /// Bytes this block occupies wherever it currently lives.
    pub(crate) fn size_in_bytes(&self) -> usize {
        self.tensors
            .values()
            .map(|tensor| tensor.storage_size_in_bytes())
            .sum()
    }

    pub(crate) fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Move every tensor in this block to `device`.
    pub(crate) fn to_device(&self, device: &Device) -> Result<Self> {
        let mut tensors = HashMap::with_capacity(self.tensors.len());
        for (name, tensor) in &self.tensors {
            tensors.insert(name.clone(), qtensor_to_device(tensor, device)?);
        }
        Ok(Self {
            index: self.index,
            tensors,
        })
    }

    /// The tensor map, ready for `mold_candle::quantized::VarBuilder::from_qtensors`.
    pub(crate) fn into_tensors(self) -> HashMap<String, Arc<QTensor>> {
        self.tensors
    }
}

/// Splits a checkpoint's quantized tensor map into per-block sets.
pub(crate) struct WanBlockParking;

impl WanBlockParking {
    /// Collect the tensors belonging to block `index`.
    ///
    /// Returns `None` when the checkpoint has no tensors under that prefix,
    /// which is how a caller discovers the real block count rather than
    /// trusting a config field against an unfamiliar export.
    pub(crate) fn park(all: &HashMap<String, Arc<QTensor>>, index: usize) -> Option<ParkedBlock> {
        let prefix = block_prefix(index);
        let tensors: HashMap<String, Arc<QTensor>> = all
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect();
        if tensors.is_empty() {
            return None;
        }
        Some(ParkedBlock { index, tensors })
    }

    /// How many blocks must be parked to free at least `needed_bytes`.
    ///
    /// Deliberately returns a count rather than a set: which blocks to park is
    /// a scheduling decision (the later ones are cheapest to prefetch behind
    /// the earlier ones), while how many is pure arithmetic and is the part
    /// worth pinning in a test.
    pub(crate) fn blocks_to_park(
        needed_bytes: u64,
        bytes_per_block: u64,
        total_blocks: usize,
    ) -> usize {
        if needed_bytes == 0 || bytes_per_block == 0 {
            return 0;
        }
        let exact = needed_bytes.div_ceil(bytes_per_block);
        (exact as usize).min(total_blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::{DType, Tensor};

    fn quantized(name: &str, rows: usize, cols: usize) -> (String, Arc<QTensor>) {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1.0, (rows, cols), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        (
            name.to_string(),
            Arc::new(QTensor::quantize(&tensor, GgmlDType::Q5K).unwrap()),
        )
    }

    fn checkpoint() -> HashMap<String, Arc<QTensor>> {
        let mut all = HashMap::new();
        for block in 0..3 {
            for leaf in ["self_attn.q.weight", "ffn.0.weight"] {
                let (name, tensor) = quantized(&format!("blocks.{block}.{leaf}"), 32, 256);
                all.insert(name, tensor);
            }
        }
        let (name, tensor) = quantized("patch_embedding.weight", 32, 256);
        all.insert(name, tensor);
        all
    }

    /// The whole premise: a park/unpark cycle must return the bytes it was
    /// given. A dequantize/re-quantize round trip would pass a loose tolerance
    /// check and still make the render depend on how many blocks were parked,
    /// so this asserts raw storage equality rather than closeness.
    #[test]
    fn parking_a_block_is_byte_identical() {
        let all = checkpoint();
        let parked = WanBlockParking::park(&all, 1).expect("block 1 exists");
        let moved = parked.to_device(&Device::Cpu).unwrap();

        for name in parked.tensor_names() {
            let before = all.get(name).expect("original tensor");
            let after = moved
                .tensors
                .get(name)
                .expect("moved set keeps every tensor name");
            assert_eq!(before.dtype(), after.dtype(), "{name} changed quantization");
            assert_eq!(before.shape(), after.shape(), "{name} changed shape");
            assert_eq!(
                before.data().unwrap().as_ref(),
                after.data().unwrap().as_ref(),
                "{name} is not byte-identical after a park/unpark cycle"
            );
        }
    }

    #[test]
    fn parking_selects_exactly_one_blocks_tensors() {
        let all = checkpoint();
        let parked = WanBlockParking::park(&all, 1).expect("block 1 exists");
        let mut names: Vec<&str> = parked.tensor_names().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec!["blocks.1.ffn.0.weight", "blocks.1.self_attn.q.weight"]
        );
        assert_eq!(parked.index(), 1);
    }

    /// `blocks.1.` must not also match `blocks.11.`, which a naive prefix over
    /// a 40-block checkpoint would.
    #[test]
    fn block_prefix_does_not_match_a_longer_index() {
        let mut all = HashMap::new();
        let (name, tensor) = quantized("blocks.1.ffn.0.weight", 32, 256);
        all.insert(name, tensor);
        let (name, tensor) = quantized("blocks.11.ffn.0.weight", 32, 256);
        all.insert(name, tensor);

        let parked = WanBlockParking::park(&all, 1).expect("block 1 exists");
        let names: Vec<&str> = parked.tensor_names().map(String::as_str).collect();
        assert_eq!(names, vec!["blocks.1.ffn.0.weight"]);
    }

    #[test]
    fn an_absent_block_parks_nothing() {
        let all = checkpoint();
        assert!(WanBlockParking::park(&all, 99).is_none());
    }

    #[test]
    fn block_count_to_park_covers_the_shortfall() {
        const MIB: u64 = 1024 * 1024;
        // The real shape: ~900 MiB short, ~270 MiB per Q5 block, 40 blocks.
        assert_eq!(WanBlockParking::blocks_to_park(900 * MIB, 270 * MIB, 40), 4);
        // Rounds up rather than leaving a shortfall.
        assert_eq!(WanBlockParking::blocks_to_park(271 * MIB, 270 * MIB, 40), 2);
        // Nothing needed, nothing parked.
        assert_eq!(WanBlockParking::blocks_to_park(0, 270 * MIB, 40), 0);
        // Never asks for more blocks than exist, even when it cannot fit.
        assert_eq!(WanBlockParking::blocks_to_park(u64::MAX, MIB, 40), 40);
        // Degenerate input cannot divide by zero.
        assert_eq!(WanBlockParking::blocks_to_park(900 * MIB, 0, 40), 0);
    }
}
