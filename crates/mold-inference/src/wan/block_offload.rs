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
//!
//! ## What the driver still needs
//!
//! One gap is known and is deliberately not closed here, because it changes
//! `WanTransformer`'s shape rather than this module's: **the transformer does
//! not retain the checkpoint's tensor map.** `from_gguf_with_loras` builds a
//! `VarBuilder`, hands it to `from_weights`, and drops it at the end of the
//! statement — after which only the tensors the blocks themselves hold survive,
//! via `Arc`. So [`WanBlockParking::park`] cannot be called after construction
//! as things stand; the driver will have to either park during construction or
//! give the transformer a retained map. Note the map cannot simply be kept
//! alive to rebuild from later: `VarBuilder::from_gguf` uploads *every* tensor
//! to the device eagerly (`mold-candle/src/quantized.rs`), so holding it would
//! pin the whole expert on the GPU and defeat the purpose.

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
    // `Device` has no `PartialEq`; `same_device` is candle's own comparison —
    // location and ordinal, so two handles to the same GPU compare equal and
    // two ordinals do not.
    if src.device().same_device(device) {
        return Ok(src.clone());
    }
    rebuild_on(src, device)
}

/// [`qtensor_to_device`] with the same-device short circuit removed.
///
/// Split out because the short circuit makes a same-device call untestable:
/// it hands back the input `Arc`, so asserting the result equals the input
/// proves nothing about the byte path — which is the entire correctness claim
/// of this module. CI has no GPU, so this is what lets a CPU-only test drive
/// exactly the serialization a CUDA park/unpark runs. Production callers want
/// the short circuit and should use [`qtensor_to_device`].
pub(crate) fn rebuild_on(src: &QTensor, device: &Device) -> Result<Arc<QTensor>> {
    let dims = src.shape().dims().to_vec();
    let dtype = src.dtype();
    let bytes = src.data()?;
    Ok(Arc::new(ggml_file::qtensor_from_ggml(
        dtype, &bytes, dims, device,
    )?))
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
        self.move_each(|tensor| qtensor_to_device(tensor, device))
    }

    /// [`Self::to_device`] with the same-device short circuit removed — see
    /// [`rebuild_on`] for why that distinction has to be reachable.
    #[cfg(test)]
    fn rebuilt_on(&self, device: &Device) -> Result<Self> {
        self.move_each(|tensor| rebuild_on(tensor, device))
    }

    fn move_each<F>(&self, mut move_one: F) -> Result<Self>
    where
        F: FnMut(&Arc<QTensor>) -> Result<Arc<QTensor>>,
    {
        let mut tensors = HashMap::with_capacity(self.tensors.len());
        for (name, tensor) in &self.tensors {
            tensors.insert(name.clone(), move_one(tensor)?);
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
    ///
    /// Deliberately drives `rebuilt_on`, not `to_device`: CI has no GPU, and a
    /// same-device `to_device` returns the input `Arc` untouched, so this test
    /// would compare a tensor with itself and pass no matter how broken the
    /// byte path was. `rebuilt_on` runs the real serialization.
    #[test]
    fn parking_a_block_is_byte_identical() {
        let all = checkpoint();
        let parked = WanBlockParking::park(&all, 1).expect("block 1 exists");
        let moved = parked.rebuilt_on(&Device::Cpu).unwrap();

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

    /// The fast path must be a genuine short circuit — the same allocation
    /// back, not a copy. Pinned because it is what makes an unpark of an
    /// already-resident block free, and because its existence is exactly what
    /// made the byte-identity test above vacuous until it was split out.
    #[test]
    fn a_same_device_move_returns_the_same_allocation() {
        let (_, tensor) = quantized("blocks.0.ffn.0.weight", 32, 256);
        let moved = qtensor_to_device(&tensor, &Device::Cpu).unwrap();
        assert!(
            Arc::ptr_eq(&tensor, &moved),
            "a same-device move must not copy"
        );

        // And the rebuild path must NOT short circuit, or the test above is
        // testing nothing.
        let rebuilt = rebuild_on(&tensor, &Device::Cpu).unwrap();
        assert!(
            !Arc::ptr_eq(&tensor, &rebuilt),
            "rebuild_on must actually round trip through bytes"
        );
        assert_eq!(
            tensor.data().unwrap().as_ref(),
            rebuilt.data().unwrap().as_ref(),
            "the rebuilt tensor must still be byte-identical"
        );
    }

    /// Every quantization the shipped Wan checkpoints use has to survive the
    /// round trip, not just the one the other tests happen to build with.
    #[test]
    fn every_shipped_quantization_survives_the_round_trip() {
        let device = Device::Cpu;
        for dtype in [
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q8_0,
            GgmlDType::F16,
            GgmlDType::F32,
        ] {
            let src = Tensor::randn(0f32, 1.0, (32, 256), &device).unwrap();
            let quantized = Arc::new(QTensor::quantize(&src, dtype).unwrap());
            let rebuilt = rebuild_on(&quantized, &device)
                .unwrap_or_else(|err| panic!("{dtype:?} failed to rebuild: {err}"));
            assert_eq!(rebuilt.dtype(), dtype, "{dtype:?} changed quantization");
            assert_eq!(
                rebuilt.shape(),
                quantized.shape(),
                "{dtype:?} changed shape"
            );
            assert_eq!(
                quantized.data().unwrap().as_ref(),
                rebuilt.data().unwrap().as_ref(),
                "{dtype:?} is not byte-identical after a round trip"
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

    /// The contract a driver depends on: the map `into_tensors` hands back has
    /// to be directly loadable by the var builder `WanBlock::load` already
    /// takes, at the path the block lives at.
    ///
    /// This is why `park` keeps FULL checkpoint names rather than stripping the
    /// `blocks.{i}.` prefix. `VarBuilder::path` joins its `pp` segments with
    /// `.` and looks the result up verbatim (`mold-candle/src/quantized.rs`),
    /// so a stripped key would need the driver to navigate to the root and a
    /// full key needs `.pp("blocks").pp(i)` — and getting that backwards fails
    /// at runtime with `cannot find tensor`, not at compile time.
    #[test]
    fn a_parked_block_reloads_through_the_var_builder() {
        let all = checkpoint();
        let parked = WanBlockParking::park(&all, 1).expect("block 1 exists");
        let index = parked.index();
        let moved = parked.rebuilt_on(&Device::Cpu).unwrap();

        let vb =
            mold_candle::quantized::VarBuilder::from_qtensors(moved.into_tensors(), &Device::Cpu);
        let block_vb = vb.pp("blocks").pp(index.to_string());

        let weight = block_vb
            .pp("self_attn")
            .pp("q")
            .get((32, 256), "weight")
            .expect("the parked block must be reachable at blocks.{i}.self_attn.q.weight");
        assert_eq!(weight.shape().dims(), &[32, 256]);

        // A sibling block's weights must NOT be reachable from this set.
        assert!(
            vb.pp("blocks")
                .pp("0")
                .pp("self_attn")
                .pp("q")
                .get((32, 256), "weight")
                .is_err(),
            "parking block 1 must not carry block 0's weights"
        );
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
