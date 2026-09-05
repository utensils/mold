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
//! ## When parking happens
//!
//! At construction, not later, and that is forced by how the weights load:
//! `VarBuilder::from_gguf` uploads *every* tensor to the device eagerly
//! (`mold-candle/src/quantized.rs`), and `from_weights` consumes the builder —
//! after which only the tensors the blocks themselves hold survive, via `Arc`.
//! So the transformer cannot retain the map to park from later without pinning
//! the whole expert on the GPU, which is the opposite of the point.
//! `WanTransformer::from_gguf_with_offload` therefore parks while the map is
//! still in scope.
//!
//! Doing it after the weights are resident costs nothing: the *load* peak is
//! one expert (~10.8 GB at Q5) and the peak that exceeds the card is the
//! denoise, so there is room to free blocks at the point parking runs. What
//! must still fit afterwards is the activation budget, which is why
//! [`plan_offload`] compares that against free VRAM rather than comparing
//! weights against capacity.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::{ggml_file, QTensor};
use candle_core::{Device, Result};
use mold_core::GpuBackend;

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

/// Explicit block count from `MOLD_WAN_OFFLOAD_BLOCKS`, or `MOLD_OFFLOAD`.
///
/// `MOLD_WAN_OFFLOAD_BLOCKS=N` pins the count exactly, including `0` to force
/// the policy off, and wins over the family-wide flag.
pub(crate) fn forced_block_count() -> Option<usize> {
    forced_block_count_for_request(None)
}

pub(crate) fn forced_block_count_for_request(offload: Option<bool>) -> Option<usize> {
    resolve_forced_block_count(
        crate::runtime_env::value("MOLD_WAN_OFFLOAD_BLOCKS").as_deref(),
        offload
            .map(|value| if value { "1" } else { "0" })
            .or(crate::runtime_env::value("MOLD_OFFLOAD").as_deref()),
    )
}

/// [`forced_block_count`] over explicit strings, so the policy is testable
/// without mutating process environment shared by parallel tests.
///
/// `--offload` / `MOLD_OFFLOAD=1` parks **every** block. That is the same
/// contract `flux/offload.rs` gives the flag — the user is asking for the
/// lowest-VRAM execution available and accepting the wall clock — and it is
/// what this module's documentation promised from the start while nothing in
/// the wan path read the variable (#776 item 3: "auto-engage under pressure,
/// forceable via `--offload`/`MOLD_OFFLOAD`"). Resolving it here rather than
/// through the factory's `block_offload` argument is deliberate: that argument
/// now carries the scheduler plan's *predicted* disposition, so overloading it
/// would turn an auto-parked render into a park-everything one.
fn resolve_forced_block_count(blocks: Option<&str>, offload: Option<&str>) -> Option<usize> {
    if let Some(raw) = blocks {
        return raw.trim().parse::<usize>().ok();
    }
    // `plan_offload` clamps to the checkpoint's real block count.
    forced_offload_requested(offload).then_some(usize::MAX)
}

fn forced_offload_requested(value: Option<&str>) -> bool {
    matches!(value.map(str::trim), Some("1" | "true" | "yes"))
}

/// Block-parking authority admission may rely on for a selected backend.
///
/// Automatic parking needs a device-local free-VRAM reading, which the Wan
/// runtime currently has only for CUDA. An explicit block count or the generic
/// offload override still engages parking on Metal because it does not need to
/// infer a shortfall from that missing reading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionPolicy {
    Disabled,
    Automatic,
    ForcedFinite(usize),
    ForcedAll,
}

impl AdmissionPolicy {
    pub fn from_values(backend: GpuBackend, blocks: Option<&str>, offload: Option<&str>) -> Self {
        if let Some(raw) = blocks {
            return match raw.trim().parse::<usize>() {
                Ok(0) => Self::Disabled,
                Ok(blocks) => Self::ForcedFinite(blocks),
                Err(_) if backend == GpuBackend::Cuda => Self::Automatic,
                Err(_) => Self::Disabled,
            };
        }
        if forced_offload_requested(offload) {
            Self::ForcedAll
        } else if backend == GpuBackend::Cuda {
            Self::Automatic
        } else {
            Self::Disabled
        }
    }

    /// Whether this policy parks any blocks for the given fit check.
    pub fn will_park(self, predicted_peak_bytes: u64, available_bytes: u64) -> bool {
        match self {
            Self::Disabled => false,
            Self::Automatic => predicted_peak_bytes > available_bytes,
            Self::ForcedFinite(blocks) => blocks > 0,
            Self::ForcedAll => true,
        }
    }

    /// Whether admission may credit the measured maximum parking relief.
    ///
    /// A finite count engages parking but only proves the measured maximum
    /// when it covers every checkpoint block. Crediting that maximum for a
    /// partial count can admit a request the pinned count cannot fit.
    pub fn supports_max_relief(self, total_blocks: Option<usize>) -> bool {
        match self {
            Self::Automatic | Self::ForcedAll => true,
            Self::ForcedFinite(blocks) => {
                total_blocks.is_some_and(|total| total > 0 && blocks >= total)
            }
            Self::Disabled => false,
        }
    }
}

/// Whether a render at this shape will park blocks at all.
///
/// The server needs the answer to name the disposition in its scheduler plan
/// (#776) without restating the policy: [`plan_offload`] decides *how many*
/// blocks from the shortfall, and this answers only whether that count is
/// non-zero. Both read one authority, so a plan that says "block offload" and
/// an engine that parks nothing cannot drift apart.
pub fn will_park(predicted_peak_bytes: u64, available_bytes: u64) -> bool {
    match forced_block_count() {
        Some(blocks) => blocks > 0,
        None => predicted_peak_bytes > available_bytes,
    }
}

/// Usable VRAM actually returned per byte of parked weight, measured.
///
/// 6,272 MiB came back from parking ~8.1 GB of Q5 blocks (see [`plan_offload`]),
/// i.e. 77% at that count, and only 13% at 8 blocks — relief is non-linear and
/// worst when few blocks are parked, because the allocator has already grown
/// its pool to the unparked high-water mark.
///
/// Carried at 35%, which is lower than any single measurement and is set by the
/// one that matters: at 81 frames / 832x480 a 58% assumption planned ~12 blocks
/// and **OOM'd**, while 20 blocks rendered the clip at a 22,218 MiB peak. 35%
/// is what turns that shortfall into ~19 blocks. Under-parking OOMs
/// mid-denoise; over-parking only costs wall-clock, so the asymmetry is
/// deliberate.
const RELIEF_EFFICIENCY_NUMERATOR: u64 = 35;
const RELIEF_EFFICIENCY_DENOMINATOR: u64 = 100;

/// Most of a transformer's weight bytes that block offload can be relied on to
/// free, for admission (#776 item 3).
///
/// Admission has to answer "is this shape feasible at all", which parking can
/// change — but it must not assume the whole expert can leave the device: the
/// blocks still take turns being resident, and the non-block weights never
/// move. This is the measured relief fraction applied to the transformer's own
/// bytes, so a shape is admitted only when parking can genuinely reach it.
pub fn max_block_offload_relief_bytes(transformer_bytes: u64) -> u64 {
    transformer_bytes.saturating_mul(RELIEF_EFFICIENCY_NUMERATOR) / RELIEF_EFFICIENCY_DENOMINATOR
}

/// What the offload policy decided for one load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WanOffloadPlan {
    /// Everything fits; keep every block resident. The no-offload path must
    /// stay byte-for-byte the behaviour it had before this existed.
    None,
    /// Park this many trailing blocks in host RAM.
    Park { blocks: usize },
}

impl WanOffloadPlan {
    pub(crate) fn blocks(self) -> usize {
        match self {
            Self::None => 0,
            Self::Park { blocks } => blocks,
        }
    }
}

/// Decide how many blocks to park for a render.
///
/// `MOLD_WAN_OFFLOAD_BLOCKS=N` pins the count and `MOLD_OFFLOAD=1` forces the
/// policy on; otherwise this engages only when the predicted peak does not fit,
/// so a render that already fits keeps exactly the execution it had before.
///
/// `headroom_bytes` is subtracted from what is free before the comparison —
/// the denoise needs room for its own activations after the weights land, and
/// parking blocks until the *weights* fit while leaving nothing for activations
/// would trade one OOM for another.
pub(crate) fn plan_offload(
    predicted_peak_bytes: u64,
    available_bytes: u64,
    bytes_per_block: u64,
    total_blocks: usize,
    forced_blocks: Option<usize>,
) -> WanOffloadPlan {
    if let Some(blocks) = forced_blocks {
        let blocks = blocks.min(total_blocks);
        return if blocks == 0 {
            WanOffloadPlan::None
        } else {
            WanOffloadPlan::Park { blocks }
        };
    }
    if predicted_peak_bytes <= available_bytes {
        return WanOffloadPlan::None;
    }
    // Parking N blocks does not return N block-sizes of usable VRAM. Measured
    // on an RTX 4090 (`wan22-t2v-a14b:q5`, 53f/832x480): parking 30 of 40
    // blocks — ~8.1 GB of Q5 weights — moved the peak 21,354 -> 15,082 MiB, so
    // 6.27 GB of 8.1 came back, and parking 8 returned only 288 MiB of the
    // ~2.2 GB they occupy. The allocator has already grown its pool to the
    // unparked high-water mark, so relief is non-linear and worst at small
    // counts. Scaling the shortfall by the measured efficiency is what stops
    // the policy parking the arithmetic minimum and still OOMing.
    let shortfall = predicted_peak_bytes - available_bytes;
    let needed = shortfall
        .saturating_mul(RELIEF_EFFICIENCY_DENOMINATOR)
        .div_ceil(RELIEF_EFFICIENCY_NUMERATOR);
    let blocks = WanBlockParking::blocks_to_park(needed, bytes_per_block, total_blocks);
    if blocks == 0 {
        WanOffloadPlan::None
    } else {
        WanOffloadPlan::Park { blocks }
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

    #[test]
    fn admission_policy_matches_backend_and_override_precedence() {
        assert_eq!(
            AdmissionPolicy::from_values(GpuBackend::Cuda, None, None),
            AdmissionPolicy::Automatic
        );
        assert_eq!(
            AdmissionPolicy::from_values(GpuBackend::Metal, None, None),
            AdmissionPolicy::Disabled
        );
        assert_eq!(
            AdmissionPolicy::from_values(GpuBackend::Metal, Some("12"), None),
            AdmissionPolicy::ForcedFinite(12)
        );
        assert_eq!(
            AdmissionPolicy::from_values(GpuBackend::Metal, None, Some("1")),
            AdmissionPolicy::ForcedAll
        );
        assert_eq!(
            AdmissionPolicy::from_values(GpuBackend::Cuda, Some("0"), Some("1")),
            AdmissionPolicy::Disabled
        );
        assert_eq!(
            AdmissionPolicy::from_values(GpuBackend::Metal, Some("invalid"), Some("1")),
            AdmissionPolicy::Disabled
        );

        assert!(!AdmissionPolicy::ForcedFinite(1).supports_max_relief(Some(40)));
        assert!(AdmissionPolicy::ForcedFinite(1).will_park(1, 2));
        assert!(AdmissionPolicy::ForcedFinite(40).supports_max_relief(Some(40)));
        assert!(AdmissionPolicy::ForcedFinite(999).supports_max_relief(Some(40)));
        assert!(!AdmissionPolicy::ForcedFinite(40).supports_max_relief(None));
        assert!(AdmissionPolicy::ForcedAll.supports_max_relief(None));
    }

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
    fn a_render_that_fits_parks_nothing() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // Fits with room to spare, and fits exactly: both keep every block
        // resident, because the no-offload path must stay untouched.
        assert_eq!(
            plan_offload(20 * GIB, 24 * GIB, 270 * 1024 * 1024, 40, None),
            WanOffloadPlan::None
        );
        assert_eq!(
            plan_offload(24 * GIB, 24 * GIB, 270 * 1024 * 1024, 40, None),
            WanOffloadPlan::None
        );
    }

    #[test]
    fn a_render_that_does_not_fit_parks_the_shortfall() {
        const MIB: u64 = 1024 * 1024;
        // The real shape: 25,461 MiB predicted against 24,564 MiB usable is a
        // 897 MiB shortfall, and a Q5 block is ~270 MiB.
        // 897 MiB short, scaled by the measured 35% relief efficiency is
        // 2,563 MiB of weights to park, so 10 blocks rather than the naive 4 —
        // parking the arithmetic minimum is what the measurement showed does
        // not actually come back.
        assert_eq!(
            plan_offload(25_461 * MIB, 24_564 * MIB, 270 * MIB, 40, None),
            WanOffloadPlan::Park { blocks: 10 }
        );
    }

    #[test]
    fn admission_relief_never_assumes_the_whole_expert_leaves() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // A14B q5's expert is ~10.05 GiB; relief is the measured 58% of it,
        // never all of it — the blocks still take turns being resident and the
        // non-block weights never move.
        let expert = 10 * GIB;
        let relief = max_block_offload_relief_bytes(expert);
        assert!(relief < expert, "relief must not free the whole expert");
        assert_eq!(relief, expert * 35 / 100);
        assert_eq!(max_block_offload_relief_bytes(0), 0);
    }

    #[test]
    fn a_forced_count_overrides_the_fit_check() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // Forced on even though it fits.
        assert_eq!(
            plan_offload(GIB, 24 * GIB, 270 * 1024 * 1024, 40, Some(8)),
            WanOffloadPlan::Park { blocks: 8 }
        );
        // Forced off even though it does not fit.
        assert_eq!(
            plan_offload(99 * GIB, 24 * GIB, 270 * 1024 * 1024, 40, Some(0)),
            WanOffloadPlan::None
        );
        // Never more blocks than exist.
        assert_eq!(
            plan_offload(99 * GIB, 24 * GIB, 270 * 1024 * 1024, 40, Some(999)),
            WanOffloadPlan::Park { blocks: 40 }
        );
    }

    /// `--offload` sets `MOLD_OFFLOAD=1`, and until this landed the wan path
    /// read neither — the flag was documented three times in this module and
    /// implemented nowhere, so the one lever a user has for "fit it at any
    /// speed" was silently inert on the family that needed it most.
    #[test]
    fn the_family_wide_offload_flag_parks_every_block() {
        // Park-everything, clamped by `plan_offload` to the real block count.
        assert_eq!(
            resolve_forced_block_count(None, Some("1")),
            Some(usize::MAX)
        );
        for spelling in ["1", "true", "yes"] {
            assert!(forced_offload_requested(Some(spelling)), "{spelling}");
        }
        for spelling in ["0", "false", "no", "", "maybe"] {
            assert!(!forced_offload_requested(Some(spelling)), "{spelling}");
        }
        assert!(!forced_offload_requested(None));

        // An explicit count wins over the family-wide flag in both directions:
        // it is the more specific instruction, including `0` for "off".
        assert_eq!(resolve_forced_block_count(Some("8"), Some("1")), Some(8));
        assert_eq!(resolve_forced_block_count(Some("0"), Some("1")), Some(0));
        // Neither set leaves the automatic shortfall policy in charge.
        assert_eq!(resolve_forced_block_count(None, None), None);
        // A malformed count is not silently read as "park everything".
        assert_eq!(resolve_forced_block_count(Some("many"), None), None);
    }

    /// The server names the disposition in its plan from the same predicate the
    /// engine parks by, so the two cannot disagree.
    #[test]
    fn will_park_matches_the_planners_own_fit_check() {
        const GIB: u64 = 1024 * 1024 * 1024;
        assert!(!will_park(20 * GIB, 24 * GIB));
        // Exactly fitting is fitting — same boundary `plan_offload` applies.
        assert!(!will_park(24 * GIB, 24 * GIB));
        assert!(will_park(25 * GIB, 24 * GIB));
        for (peak, available) in [(20 * GIB, 24 * GIB), (25 * GIB, 24 * GIB)] {
            let parked =
                plan_offload(peak, available, 270 * 1024 * 1024, 40, None) != WanOffloadPlan::None;
            assert_eq!(
                parked,
                will_park(peak, available),
                "{peak} against {available}"
            );
        }
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
