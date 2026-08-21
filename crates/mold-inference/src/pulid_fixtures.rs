//! Test-only fixture plumbing shared by the PuLID encoders.
//!
//! The parity goldens under `crates/mold-inference/testdata/pulid/` were
//! captured from upstream `ToTheBeginning/PuLID` on CPU in float32 (see that
//! directory's `README.md` and `capture_eva_goldens.py`). Their *inputs* are never
//! committed as bytes — both the capture script and this module generate them
//! from the same deterministic value stream, so a fixture tensor of any size
//! costs nothing in the repository.

// The PuLID pipeline that consumes this module lands with the FLUX
// integration (milestone "PuLID-FLUX: functional"); issue #1229 delivers the
// encoders and their parity coverage on their own. Until that consumer exists
// every item here is reachable only from tests, so the dead-code lint would
// otherwise force either a premature `pub` surface or a stub caller.
#![allow(dead_code)]

use candle_core::{DType, Device, Tensor};
use std::path::{Path, PathBuf};

/// `xorshift64*`, chosen because it is four lines in both Python and Rust and
/// leaves no room for a library version to change the numbers underneath a
/// golden. Mirrors `capture_eva_goldens.py::DeterministicStream`.
pub(crate) struct DeterministicStream {
    state: u64,
}

const MULTIPLIER: u64 = 0x2545_F491_4F6C_DD1D;

/// Seeds. Each fixture draws from its own stream so adding one never shifts
/// another's values. The ASCII in the comments is what the constant spells.
pub(crate) const SEED_TOWER_INPUT: u64 = 0x50554C49_44544F57; // PULIDTOW
pub(crate) const SEED_TOWER_PROBE: u64 = 0x50554C49_44505242; // PULIDPRB
pub(crate) const SEED_IDFORMER_ID: u64 = 0x50554C49_44494446; // PULIDIDF
pub(crate) const SEED_IDFORMER_VIT: u64 = 0x50554C49_44564954; // PULIDVIT
pub(crate) const SEED_IMAGE: u64 = 0x50554C49_44494D47; // PULIDIMG

/// How many scattered elements each large-tensor golden pins.
pub(crate) const PROBE_COUNT: usize = 512;

impl DeterministicStream {
    pub(crate) fn new(seed: u64) -> Self {
        assert_ne!(seed, 0, "xorshift64* has a fixed point at zero");
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(MULTIPLIER)
    }

    /// One sample in `[-1, 1)`.
    fn next_unit(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 11) as f64;
        ((mantissa / (1_u64 << 53) as f64) * 2.0 - 1.0) as f32
    }

    pub(crate) fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next_unit()).collect()
    }

    pub(crate) fn tensor(&mut self, shape: &[usize], device: &Device) -> Tensor {
        let count = shape.iter().product::<usize>();
        Tensor::from_vec(self.values(count), shape, device).expect("fixture tensor")
    }

    /// Flat element indices into a tensor of `modulo` elements.
    pub(crate) fn indices(&mut self, count: usize, modulo: usize) -> Vec<u32> {
        (0..count)
            .map(|_| (self.next_u64() % modulo as u64) as u32)
            .collect()
    }
}

/// The committed golden directory.
pub(crate) fn testdata_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/pulid")
}

/// Load one named array out of `goldens.safetensors` as f32 on the CPU.
pub(crate) fn golden(name: &str) -> Tensor {
    let path = testdata_dir().join("goldens.safetensors");
    let tensors = candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    tensors
        .get(name)
        .unwrap_or_else(|| panic!("golden {name} is missing from {}", path.display()))
        .to_dtype(DType::F32)
        .expect("golden is numeric")
}

/// Gather the flat elements a probe golden pins.
pub(crate) fn gather_probe(tensor: &Tensor, seed: u64) -> Vec<f32> {
    let flat = tensor.flatten_all().expect("flatten");
    let count = flat.dim(0).expect("rank 1");
    let indices = DeterministicStream::new(seed).indices(PROBE_COUNT, count);
    let values = flat.to_vec1::<f32>().expect("f32 golden comparison");
    indices
        .into_iter()
        .map(|index| values[index as usize])
        .collect()
}

/// Largest absolute difference, and the largest difference relative to the
/// per-element reference magnitude (floored at 1 so a near-zero reference
/// cannot manufacture an enormous ratio).
pub(crate) fn max_errors(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len(), "golden length mismatch");
    let mut absolute = 0.0_f32;
    let mut relative = 0.0_f32;
    for (a, e) in actual.iter().zip(expected) {
        let difference = (a - e).abs();
        absolute = absolute.max(difference);
        relative = relative.max(difference / e.abs().max(1.0));
    }
    (absolute, relative)
}

/// A golden's recorded whole-tensor statistics: mean, standard deviation,
/// minimum, maximum, and peak magnitude, in that order. Written by
/// `capture_eva_goldens.py::stats_tensor`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GoldenStats {
    pub(crate) mean: f32,
    pub(crate) std: f32,
    pub(crate) min: f32,
    pub(crate) max: f32,
    pub(crate) peak: f32,
}

impl GoldenStats {
    pub(crate) fn load(name: &str) -> Self {
        let values = golden(name).to_vec1::<f32>().expect("stats golden");
        assert_eq!(values.len(), 5, "{name} is not a five-slot stats golden");
        Self {
            mean: values[0],
            std: values[1],
            min: values[2],
            max: values[3],
            peak: values[4],
        }
    }

    /// Recompute the same statistics from a candidate tensor.
    ///
    /// The probe goldens pin 512 scattered elements; these pin the whole
    /// tensor at once, which is what catches a defect that misses every probe
    /// index — a wrong value in one token, say, or a scale error confined to
    /// the tail.
    pub(crate) fn measure(tensor: &Tensor) -> Self {
        let values = tensor
            .flatten_all()
            .expect("flatten")
            .to_dtype(DType::F32)
            .expect("f32")
            .to_vec1::<f32>()
            .expect("f32 values");
        let count = values.len() as f64;
        let mean = values.iter().map(|&v| v as f64).sum::<f64>() / count;
        let variance = values
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / count;
        Self {
            mean: mean as f32,
            std: variance.sqrt() as f32,
            min: values.iter().copied().fold(f32::INFINITY, f32::min),
            max: values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            peak: values.iter().fold(0.0_f32, |peak, &v| peak.max(v.abs())),
        }
    }

    /// Assert agreement, every slot judged against the tensor's own peak
    /// magnitude so one tolerance covers a residual stream whose values span
    /// four orders of magnitude.
    pub(crate) fn assert_matches(&self, other: &Self, tolerance: f32, label: &str) {
        let scale = self.peak.max(1.0);
        for (name, expected, actual) in [
            ("std", self.std, other.std),
            ("min", self.min, other.min),
            ("max", self.max, other.max),
            ("peak", self.peak, other.peak),
            ("mean", self.mean, other.mean),
        ] {
            let error = (expected - actual).abs() / scale;
            assert!(
                error < tolerance,
                "{label} {name}: expected {expected}, got {actual} \
                 ({error:.3e} of the {scale} scale, tolerance {tolerance:.3e})"
            );
        }
    }
}

/// Largest absolute difference as a fraction of `peak`.
///
/// This is the metric the EVA02 hidden-state goldens are judged on, and the
/// per-element ratio above is not. The residual stream carries genuine
/// outliers — the tapped states peak at 40, 49, 58, 120 and 257 — so f32
/// rounding is proportional to the *tensor's* scale, not to whichever element
/// it happened to land on. Judging a 3e-2 deviation against a neighbouring
/// probe element of magnitude 1 would read as a 3% error when it is 1.3e-4 of
/// the signal. `peak` therefore comes from the whole-tensor [`GoldenStats`],
/// never from the 512-element probe subsample.
pub(crate) fn scale_relative_error(actual: &[f32], expected: &[f32], peak: f32) -> f32 {
    let (absolute, _) = max_errors(actual, expected);
    if peak > 0.0 {
        absolute / peak
    } else {
        absolute
    }
}

/// Root of the pinned PuLID checkpoints for weight-gated tests. Point
/// `MOLD_TEST_PULID_ASSETS` at a directory holding
/// `EVA02_CLIP_L_336_psz14_s6B.pt` and `pulid_flux_v0.9.1.safetensors`
/// (searched one level deep, so the `hf download --local-dir` layout works).
pub(crate) fn pulid_asset(filename: &str) -> PathBuf {
    let root = std::env::var_os("MOLD_TEST_PULID_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set MOLD_TEST_PULID_ASSETS to the PuLID checkpoint root"));
    let direct = root.join(filename);
    if direct.is_file() {
        return direct;
    }
    let nested = std::fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", root.display()))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path().join(filename))
        .find(|candidate| candidate.is_file());
    nested.unwrap_or_else(|| {
        panic!(
            "{filename} is not under {} (nor one level below it)",
            root.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the stream itself. If this drifts every golden below it is
    /// comparing against different inputs than the capture script used, which
    /// would look like a model bug rather than a fixture bug.
    #[test]
    fn the_value_stream_is_pinned() {
        let mut stream = DeterministicStream::new(SEED_TOWER_INPUT);
        let values = stream.values(4);
        // Captured from `capture_eva_goldens.py::DeterministicStream` under the
        // same seed; see `testdata/pulid/README.md`.
        let expected = [0.803_524_6_f32, -0.701_746_34, -0.431_918_62, 0.244_076_33];
        let (absolute, _) = max_errors(&values, &expected);
        assert!(absolute < 1e-6, "stream drifted: {values:?}");
    }

    #[test]
    fn the_stream_is_reproducible_across_calls() {
        let first = DeterministicStream::new(SEED_IDFORMER_ID).values(64);
        let second = DeterministicStream::new(SEED_IDFORMER_ID).values(64);
        assert_eq!(first, second);
        let other = DeterministicStream::new(SEED_IDFORMER_VIT).values(64);
        assert_ne!(first, other, "distinct seeds must not collide");
    }
}
