//! Device parity and the charged device peak for #1227 phase 2.
//!
//! Phase 2 moved the whole extraction — detector, recognizer, parser, EVA
//! tower, IDFormer — from `Device::Cpu` at admission onto the render's leased
//! device (`docs/architecture/pulid-perf.md` §5). Two things have to be true
//! for that to be a performance change rather than a behaviour change:
//!
//! 1. **The identity is the same person.** The device path must agree with the
//!    host path on the final `[1, 32, 2048]` tokens FLUX is conditioned on, to
//!    a tolerance stated and measured rather than assumed.
//! 2. **The memory charge is honest.** Admission adds
//!    `IDENTITY_EXTRACTION_VRAM_OVERHEAD_BYTES` to the peak estimate for a
//!    conditioned request. A charge that is far above the truth parks renders
//!    the card could run; one below it OOMs a card that was admitted.
//!
//! Weight-gated and `#[ignore]`d, like every other test that needs the pinned
//! checkpoints:
//!
//! ```text
//! MOLD_TEST_PULID_ASSETS=/path/to/pulid-assets \
//!   cargo test -p mold-ai-inference --features pulid,metal \
//!   --test pulid_device_parity -- --ignored --nocapture
//! ```
//!
//! Every test here SKIPS rather than fails when no accelerator is present: a
//! CPU-only build and a CI runner without a GPU are both legitimate, and a
//! device test that fails on them would be reporting the runner.

#![cfg(feature = "pulid")]

use std::path::PathBuf;

use candle_core::Device;
use mold_core::pulid_assets::PulidPaths;
use mold_inference::identity::extraction::compose_identity_tokens_observed;

/// Largest deviation between the device and host token sets, as a fraction of
/// the host tensor's own peak.
///
/// **No new tolerance is invented.** This is the same `5e-5` that
/// `identity::extraction`'s `the_identity_matches_upstream_end_to_end` and
/// `crates/mold-inference/testdata/pulid/README.md`'s "final identity, relative
/// to its own peak" row already state over the whole stack, and it holds even
/// though the device tower runs f16 where the host tower runs f32
/// (`identity::extraction::eva_working_dtype` — which is upstream's own choice:
/// `PuLID/pulid/pipeline_flux.py:60` casts the tower to `weight_dtype`, bf16 in
/// `PuLID/app_flux.py:45`, so the host's f32 is the wide outlier).
///
/// That it holds is a measurement, not an assumption, and it is less
/// surprising than it looks: the IDFormer attends over 577 tokens per scale
/// and the CLS projection is L2-normalized, so the tower's per-element f16
/// error averages down rather than accumulating. Measured worst across the
/// four committed portraits on halcyon (M4 Max, Metal): **3.82e-5**.
const DEVICE_RELATIVE_BUDGET: f32 = 5e-5;

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/pulid")
}

/// Resolve one pinned asset under `MOLD_TEST_PULID_ASSETS`, searching one
/// level down exactly as the other weight-gated suites do — the bundle is
/// laid out per-component on a dev box and flat in the installed root.
fn pulid_asset(name: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MOLD_TEST_PULID_ASSETS")?);
    let direct = dir.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path().join(name))
        .find(|path| path.is_file())
}

fn paths() -> Option<PulidPaths> {
    Some(PulidPaths {
        adapter: pulid_asset("pulid_flux_v0.9.1.safetensors")?,
        vision_encoder_source: pulid_asset("EVA02_CLIP_L_336_psz14_s6B.pt")?,
        face_detector: pulid_asset("scrfd_10g_bnkps.onnx")?,
        face_recognizer: pulid_asset("glintr100.onnx")?,
        face_parser_source: pulid_asset("parsing_bisenet.pth")?,
    })
}

/// The accelerator this build and this machine actually have, if any.
fn accelerator() -> Option<Device> {
    if let Ok(device) = Device::new_cuda(0) {
        return Some(device);
    }
    // Through the crate's own memo rather than `Device::new_metal`, for the
    // reason `production_code_never_constructs_a_metal_device_directly` pins:
    // two devices for one GPU have a split identity.
    mold_inference::device::metal_device(0).ok()
}

/// Every committed face fixture's aligned crop and raw ArcFace embedding.
fn fixtures() -> Vec<(String, Vec<f32>, image::RgbImage)> {
    let faces = testdata_dir().join("faces");
    let sources = std::fs::read_to_string(faces.join("sources.json"))
        .expect("the #1222 face fixtures are committed");
    sources
        .split("\"file\": \"")
        .skip(1)
        .filter_map(|rest| rest.split('"').next())
        .map(|file| file.trim_end_matches(".jpg").to_string())
        .map(|stem| {
            let golden: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(faces.join(format!("{stem}.golden.json"))).unwrap(),
            )
            .unwrap();
            let arcface: Vec<f32> = golden["embedding"]
                .as_array()
                .expect("the golden carries the raw embedding")
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            let crop = image::open(faces.join(format!("{stem}.eva512.png")))
                .unwrap()
                .to_rgb8();
            (stem, arcface, crop)
        })
        .collect()
}

/// The identity a device produces must be the same person the host produces.
///
/// Stated over the FINAL tokens rather than an intermediate, because that is
/// the value the adapter injects and the value the frozen fingerprint is taken
/// over. An intermediate agreeing proves nothing about what FLUX sees.
#[test]
#[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
fn the_device_path_matches_the_host_path_within_the_recorded_tolerance() {
    let Some(paths) = paths() else {
        eprintln!("skipping: MOLD_TEST_PULID_ASSETS is unset or incomplete");
        return;
    };
    let Some(device) = accelerator() else {
        eprintln!("skipping: this build and machine have no CUDA or Metal device");
        return;
    };

    let mut worst = 0.0f32;
    for (stem, arcface, crop) in fixtures() {
        let host =
            compose_identity_tokens_observed(&paths, &arcface, &crop, &Device::Cpu, &mut |_, _| {})
                .expect("the host path composes");
        let accelerated =
            compose_identity_tokens_observed(&paths, &arcface, &crop, &device, &mut |_, _| {})
                .expect("the device path composes");
        assert_eq!(host.len(), accelerated.len());

        let peak = host.iter().fold(0.0f32, |peak, v| peak.max(v.abs()));
        assert!(peak > 0.0, "{stem}: the host tokens are all zero");
        let relative = host
            .iter()
            .zip(accelerated.iter())
            .fold(0.0f32, |worst, (h, d)| worst.max((h - d).abs() / peak));
        eprintln!("{stem}: device relative error {relative:.3e} of peak {peak:.3}");
        worst = worst.max(relative);
    }
    assert!(
        worst <= DEVICE_RELATIVE_BUDGET,
        "worst device relative error {worst:.3e} exceeds {DEVICE_RELATIVE_BUDGET:.1e}"
    );
}

/// What the extraction actually peaks at on the device, against what admission
/// charges for it.
///
/// **CUDA only, and that is the point of the test rather than a limitation.**
/// On Metal there is no second pool to measure: unified memory means the
/// extraction's bytes ARE host bytes, mold's rule is that Metal reserves no
/// host RAM separately (CLAUDE.md), and `free_vram_bytes` on macOS reports
/// system availability, which on a shared box moves by gigabytes for reasons
/// that have nothing to do with this phase. A number taken there would be
/// noise wearing a measurement's clothes.
///
/// The charge itself is a `mold_core` constant so the admission gate — which
/// compiles without the `pulid` feature — can read it;
/// `EXTRACTION_DEVICE_PEAK_BYTES`'s own doc carries the per-term derivation
/// from the artifacts' pinned sizes, and this is the live check on it.
/// Sampling is at the composer's stage boundaries, which is where the peak
/// lives: the tower's weights are resident across `EvaConstruct` and
/// `EvaForward`, and the IDFormer's across `IdFormerBuild`.
#[test]
#[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS and a CUDA device"]
fn the_measured_device_peak_is_within_ten_percent_of_the_charged_term() {
    let Some(paths) = paths() else {
        eprintln!("skipping: MOLD_TEST_PULID_ASSETS is unset or incomplete");
        return;
    };
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skipping: a device peak is only separable from host memory on CUDA");
        return;
    };
    let charged = mold_core::identity::EXTRACTION_DEVICE_PEAK_BYTES;

    let Some((stem, arcface, crop)) = fixtures().into_iter().next() else {
        panic!("the face fixtures are committed");
    };
    // Warm: the derived artifacts are converted on first use, and a
    // conversion's allocations are not what this phase costs on a warm host.
    compose_identity_tokens_observed(&paths, &arcface, &crop, &device, &mut |_, _| {})
        .expect("the device path composes");

    let Some(idle) = mold_inference::device::free_vram_bytes(0) else {
        eprintln!("skipping: this device reports no VRAM accounting");
        return;
    };
    let mut low = idle;
    compose_identity_tokens_observed(&paths, &arcface, &crop, &device, &mut |_, _| {
        if let Some(free) = mold_inference::device::free_vram_bytes(0) {
            low = low.min(free);
        }
    })
    .expect("the device path composes");

    let measured = idle.saturating_sub(low);
    eprintln!(
        "{stem}: measured device peak {measured} bytes, charged {charged} bytes ({:.1}% of charge)",
        measured as f64 / charged as f64 * 100.0
    );
    assert!(
        measured <= charged,
        "the charge {charged} must cover the measured {measured}; a device admitted on \
         an under-charge OOMs mid-extraction"
    );
    assert!(
        measured * 10 >= charged * 9,
        "the charge {charged} is more than 10% above the measured {measured}; an \
         over-charge parks renders the device could actually run"
    );
}

/// A repeat render of the same face must not re-run a 300 GFLOP tower.
///
/// This is `pulid-perf.md` §2's whole point, stated over the production entry
/// point rather than over the cache's own internals: a second
/// `extract_identity_embeddings` for byte-identical photographs opens no
/// detector, no recognizer, no parser, no tower, and no adapter, and returns a
/// `FrozenIdentityEmbedding` byte-identical to the first — because if it did
/// not, the fingerprint would stop identifying an identity and every
/// provenance claim built on it would break.
///
/// Timing is asserted only as an order of magnitude (a hit must be at least
/// ten times faster), because a threshold in milliseconds is a property of the
/// machine and its load; the hit COUNTER is the exact statement.
#[test]
#[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
fn a_repeat_photograph_is_served_from_the_cache_without_loading_anything() {
    use mold_inference::identity::extraction::{
        extract_identity_embeddings, forget_cached_identities, identity_cache_hit_count,
    };

    let Some(paths) = paths() else {
        eprintln!("skipping: MOLD_TEST_PULID_ASSETS is unset or incomplete");
        return;
    };
    let device = accelerator().unwrap_or(Device::Cpu);
    let photo = std::fs::read(
        testdata_dir()
            .join("faces")
            .join("frank-rubio-official-portrait.jpg"),
    )
    .expect("the portrait fixture is committed");

    forget_cached_identities();
    let hits = identity_cache_hit_count();
    let started = std::time::Instant::now();
    let first = extract_identity_embeddings(&paths, &[&photo], true, &device)
        .expect("the first extraction runs");
    let cold = started.elapsed();
    assert_eq!(
        identity_cache_hit_count(),
        hits,
        "a cold extraction must not report a hit"
    );

    let started = std::time::Instant::now();
    let second = extract_identity_embeddings(&paths, &[&photo], true, &device)
        .expect("the second extraction runs");
    let warm = started.elapsed();
    assert_eq!(
        identity_cache_hit_count(),
        hits + 1,
        "the second extraction did not hit the cache"
    );
    assert_eq!(
        first.embedding.fingerprint(),
        second.embedding.fingerprint(),
        "a cache hit must produce the identical frozen identity"
    );
    eprintln!(
        "cold {:.1} ms -> warm {:.1} ms",
        cold.as_secs_f64() * 1000.0,
        warm.as_secs_f64() * 1000.0
    );
    assert!(
        warm * 10 < cold,
        "a hit that costs a tenth of a miss is not a hit: cold {cold:?}, warm {warm:?}"
    );

    // A different photograph is an ordinary miss, not an eviction problem.
    let other = std::fs::read(
        testdata_dir()
            .join("faces")
            .join("kayla-barron-official-portrait.jpg"),
    )
    .expect("the second portrait fixture is committed");
    let hits = identity_cache_hit_count();
    let different = extract_identity_embeddings(&paths, &[&other], false, &device)
        .expect("a different photograph extracts");
    assert_eq!(identity_cache_hit_count(), hits, "a new face must miss");
    assert_ne!(
        first.embedding.fingerprint(),
        different.embedding.fingerprint()
    );
}
