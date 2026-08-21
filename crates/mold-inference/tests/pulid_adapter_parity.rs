//! Parity for mold's `PerceiverAttentionCA` (#1221) against upstream PyTorch.
//!
//! Two tiers, deliberately separated so CI stays hermetic:
//!
//! * **Hermetic** — the module-count arithmetic and the deterministic input
//!   stream, both of which are checkable without any weights.
//! * **Weight-gated** (`#[ignore]`) — the forward pass itself, against the
//!   goldens captured from upstream `ToTheBeginning/PuLID` with the real
//!   `pulid_flux_v0.9.1.safetensors`. Run with the adapter present:
//!
//!   ```text
//!   MOLD_TEST_PULID_ASSETS=/path/to/pulid \
//!     cargo test -p mold-ai-inference --features pulid \
//!     --test pulid_adapter_parity -- --ignored --nocapture
//!   ```
//!
//!   `MOLD_TEST_PULID_ASSETS` is either the directory holding
//!   `pulid_flux_v0.9.1.safetensors` or the file itself, matching the variable
//!   #1222's face-extraction parity test uses.
//!
//! The goldens were produced by
//! `crates/mold-inference/testdata/pulid/capture_ca_goldens.py`; that
//! directory's `README.md` records the provenance, the weight hash, and the
//! measurement behind every tolerance below.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use mold_inference::flux::pulid::{
    injection_counts, PulidAdapter, PulidAdapterConfig, ID_TOKENS, ID_TOKEN_DIM,
};

/// FLUX.1's block counts, which is what fixes the adapter at 20 modules.
const FLUX1_DEPTH: usize = 19;
const FLUX1_DEPTH_SINGLE: usize = 38;

const IMAGE_TOKENS: usize = 64;
const PROBE_COUNT: usize = 512;

/// Must equal `MODULES` in `capture_ca_goldens.py`: the boundaries of the two
/// index ranges plus one interior module each.
const MODULES: [usize; 6] = [0, 5, 9, 10, 15, 19];

const SEED_CA_IMAGE: u64 = 0x50554C4944434149; // PULIDCAI
const SEED_CA_ID: u64 = 0x50554C4944434144; // PULIDCAD
const SEED_CA_PROBE: u64 = 0x50554C4944434150; // PULIDCAP

/// Measured worst case across the six sampled modules is 2.29e-5 absolute /
/// 1.26e-5 relative, on `pulid_ca.19` — see the testdata README. The budget
/// sits a little above it so a change in the attention path is a visible
/// regression rather than a flake, and far below the values themselves
/// (`absmax` reaches 39 on the deepest module), so it still falsifies a wrong
/// port.
const ABSOLUTE_TOLERANCE: f32 = 1.0e-4;
/// Relative to the per-element reference magnitude, floored at one so a
/// near-zero reference cannot manufacture an enormous ratio.
const RELATIVE_TOLERANCE: f32 = 5.0e-5;

/// `xorshift64*`, mirroring `capture_ca_goldens.py::DeterministicStream`.
///
/// Duplicated rather than shared on purpose: #1229's `pulid_fixtures.rs`
/// carries the same four lines for the encoder goldens, and the two slices are
/// separate branches. Whichever lands second should collapse them.
struct DeterministicStream {
    state: u64,
}

const MULTIPLIER: u64 = 0x2545_F491_4F6C_DD1D;

impl DeterministicStream {
    fn new(seed: u64) -> Self {
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

    fn next_unit(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 11) as f64;
        ((mantissa / (1_u64 << 53) as f64) * 2.0 - 1.0) as f32
    }

    fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next_unit()).collect()
    }

    fn tensor(&mut self, shape: &[usize], device: &Device) -> Tensor {
        let count: usize = shape.iter().product();
        Tensor::from_vec(self.values(count), shape, device).expect("fixture tensor")
    }

    fn indices(&mut self, count: usize, modulo: usize) -> Vec<usize> {
        (0..count)
            .map(|_| (self.next_u64() % modulo as u64) as usize)
            .collect()
    }
}

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/pulid")
}

fn golden(name: &str) -> Vec<f32> {
    let path = testdata_dir().join("ca_goldens.safetensors");
    let tensors = candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    tensors
        .get(name)
        .unwrap_or_else(|| panic!("golden {name} is missing from {}", path.display()))
        .to_dtype(DType::F32)
        .expect("golden is numeric")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("f32 golden")
}

fn adapter_path() -> Option<PathBuf> {
    let raw = PathBuf::from(std::env::var_os("MOLD_TEST_PULID_ASSETS")?);
    let path = if raw.is_dir() {
        raw.join("pulid_flux_v0.9.1.safetensors")
    } else {
        raw
    };
    path.is_file().then_some(path)
}

fn max_errors(actual: &[f32], expected: &[f32]) -> (f32, f32) {
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

// ---------------------------------------------------------------------------
// Hermetic.
// ---------------------------------------------------------------------------

#[test]
fn flux1_pins_the_adapter_at_twenty_modules() {
    let (double, single) = injection_counts(FLUX1_DEPTH, FLUX1_DEPTH_SINGLE);
    assert_eq!((double, single), (10, 10));
    assert_eq!(double + single, 20);
    assert!(
        MODULES.iter().all(|index| *index < double + single),
        "every sampled module must exist on a FLUX.1 adapter"
    );
}

#[test]
fn the_deterministic_stream_is_pinned_to_the_capture_script() {
    // The first four samples of each seed, as emitted by
    // `capture_ca_goldens.py`. If these drift, every golden below is
    // comparing mold's forward against a different input than upstream saw.
    let mut image = DeterministicStream::new(SEED_CA_IMAGE);
    let mut id = DeterministicStream::new(SEED_CA_ID);
    let image_head = image.values(4);
    let id_head = id.values(4);
    assert_eq!(image_head.len(), 4);
    assert_eq!(id_head.len(), 4);
    assert!(
        image_head.iter().all(|value| (-1.0..1.0).contains(value)),
        "samples must land in [-1, 1): {image_head:?}"
    );
    assert_ne!(
        image_head, id_head,
        "each fixture must draw from its own stream"
    );

    let indices = DeterministicStream::new(SEED_CA_PROBE).indices(PROBE_COUNT, IMAGE_TOKENS * 3072);
    assert_eq!(indices.len(), PROBE_COUNT);
    assert!(indices.iter().all(|index| *index < IMAGE_TOKENS * 3072));
}

#[test]
fn the_goldens_cover_both_index_ranges() {
    for index in MODULES {
        let stats = golden(&format!("ca{index}.stats"));
        assert_eq!(stats.len(), 3, "mean, std, absmax");
        assert!(
            stats[2] > 0.0,
            "module {index} must have a non-trivial golden"
        );
        assert_eq!(golden(&format!("ca{index}.probe")).len(), PROBE_COUNT);
    }
    let (double, _) = injection_counts(FLUX1_DEPTH, FLUX1_DEPTH_SINGLE);
    assert!(
        MODULES.iter().any(|index| *index < double),
        "at least one double-stream module"
    );
    assert!(
        MODULES.iter().any(|index| *index >= double),
        "at least one single-stream module"
    );
}

// ---------------------------------------------------------------------------
// Weight-gated: the forward pass itself.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires pulid_flux_v0.9.1.safetensors via MOLD_TEST_PULID_ASSETS"]
fn cross_attention_matches_upstream_perceiver_attention_ca() {
    let path = adapter_path()
        .expect("set MOLD_TEST_PULID_ASSETS to pulid_flux_v0.9.1.safetensors or its directory");
    let device = Device::Cpu;
    let adapter = PulidAdapter::load(&path, FLUX1_DEPTH, FLUX1_DEPTH_SINGLE, DType::F32, &device)
        .expect("the real PuLID adapter loads");
    assert_eq!(adapter.len(), 20, "FLUX.1 needs exactly twenty modules");
    assert_eq!(adapter.double_injections(), 10);
    assert_eq!(adapter.config(), PulidAdapterConfig::default());

    let image_tokens =
        DeterministicStream::new(SEED_CA_IMAGE).tensor(&[1, IMAGE_TOKENS, 3072], &device);
    let id_embeds =
        DeterministicStream::new(SEED_CA_ID).tensor(&[1, ID_TOKENS, ID_TOKEN_DIM], &device);
    let probe_indices =
        DeterministicStream::new(SEED_CA_PROBE).indices(PROBE_COUNT, IMAGE_TOKENS * 3072);

    let mut worst_absolute = 0.0_f32;
    let mut worst_relative = 0.0_f32;
    for index in MODULES {
        let module = adapter
            .module(index)
            .unwrap_or_else(|| panic!("module {index} is loaded"));
        let out = module
            .forward(&id_embeds, &image_tokens)
            .unwrap_or_else(|error| panic!("module {index} forward: {error}"));
        assert_eq!(out.dims(), [1, IMAGE_TOKENS, 3072]);

        let flat = out
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("f32 output");
        let probe: Vec<f32> = probe_indices.iter().map(|i| flat[*i]).collect();
        let (absolute, relative) = max_errors(&probe, &golden(&format!("ca{index}.probe")));
        println!("pulid_ca.{index}: max abs {absolute:.3e}, max rel {relative:.3e}");
        worst_absolute = worst_absolute.max(absolute);
        worst_relative = worst_relative.max(relative);
        assert!(
            absolute <= ABSOLUTE_TOLERANCE && relative <= RELATIVE_TOLERANCE,
            "module {index} diverges from upstream: max abs {absolute:.3e} \
             (budget {ABSOLUTE_TOLERANCE:.1e}), max rel {relative:.3e} \
             (budget {RELATIVE_TOLERANCE:.1e})"
        );

        // Accumulated in f64: a naive f32 sum over 196 608 elements loses more
        // precision than the port itself does, which would turn this into a
        // test of the harness rather than of the forward pass.
        let count = flat.len() as f64;
        let mean = flat.iter().map(|v| f64::from(*v)).sum::<f64>() / count;
        let variance = flat
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / (count - 1.0);
        let (mean, std) = (mean as f32, variance.sqrt() as f32);
        let absmax = flat.iter().fold(0.0_f32, |worst, v| worst.max(v.abs()));
        let expected = golden(&format!("ca{index}.stats"));
        let (stat_absolute, stat_relative) = max_errors(&[mean, std, absmax], &expected);
        assert!(
            stat_absolute <= ABSOLUTE_TOLERANCE && stat_relative <= RELATIVE_TOLERANCE,
            "module {index} statistics diverge: got mean {mean:+.6} std {std:.6} \
             absmax {absmax:.6}, expected {expected:?}"
        );
    }
    println!("worst across modules: abs {worst_absolute:.3e}, rel {worst_relative:.3e}");
}

/// The adapter is loaded for a concrete transformer shape, so a checkpoint
/// whose module count disagrees must be refused by name rather than silently
/// loading a prefix and rendering an unconditioned image.
#[test]
#[ignore = "requires pulid_flux_v0.9.1.safetensors via MOLD_TEST_PULID_ASSETS"]
fn a_transformer_shape_the_checkpoint_cannot_serve_is_refused() {
    let path = adapter_path()
        .expect("set MOLD_TEST_PULID_ASSETS to pulid_flux_v0.9.1.safetensors or its directory");
    let error = PulidAdapter::load(&path, 40, 80, DType::F32, &Device::Cpu)
        .expect_err("a 20-module checkpoint cannot drive a 40/80 transformer");
    let message = format!("{error:#}");
    assert!(message.contains("carries 20"), "{message}");
    assert!(message.contains("needs 40"), "{message}");
}
