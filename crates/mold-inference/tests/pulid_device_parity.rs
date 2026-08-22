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
/// that have nothing to do with this phase.
///
/// ## Two ways to measure this wrongly, both of which the first version did
///
/// The first version of this test reported a peak of **0 bytes** on an L40S,
/// and the charge passed anyway because zero is under any ceiling. Both flaws
/// are worth naming, because both are easy to reintroduce:
///
/// 1. **It warmed up first.** candle's CUDA allocator does not return freed
///    blocks to the driver, so a warm-up run leaves the whole peak already
///    reserved: the baseline is sampled with the memory outstanding and the
///    measured run simply reuses it, moving `free_vram_bytes` by nothing. The
///    only run that can be measured is the FIRST one on a fresh context.
/// 2. **It measured the wrong function.** `compose_identity_tokens_observed`
///    takes an already-computed ArcFace vector and an already-aligned crop, so
///    SCRFD and ArcFace never ran and their weights were never placed. The
///    charge covers the whole extraction, so the measurement has to go through
///    `extract_identity_embeddings`, which is what admission's job actually
///    calls.
///
/// ## The band nets out what the charge deliberately reserves
///
/// This measures ONE photograph; the charge budgets for `ID_IMAGES_MAX` of
/// them. Comparing the two directly puts an honest measurement permanently at
/// the bottom of the ±10% band — plato's three cold runs came in at
/// 637,534,208 / 643,825,664 / 643,825,664 against a naive floor of
/// 630,000,000, i.e. 7.5 MB from a false failure. Raising the charge makes it
/// worse, not better: 710,000,000 floors at 639,000,000 and fails on the
/// 637.5 MB run already observed.
///
/// So the over-charge half subtracts
/// `EXTRACTION_DEVICE_MULTI_IMAGE_ALLOWANCE_BYTES` first, leaving a real ±10%
/// band around a single-photograph run (floor 597,600,000, ~40 MB of margin).
/// The coverage half is unchanged and still uses the full charge, because
/// under-charging is what OOMs a card mid-extraction.
///
/// Measured properly on plato: **637,534,208–643,825,664 bytes** against a
/// 700,000,000 charge.
#[test]
#[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS and a CUDA device"]
fn the_measured_device_peak_is_within_ten_percent_of_the_charged_term() {
    use mold_inference::identity::extraction::{
        extract_identity_embeddings, forget_cached_identities,
    };

    let Some(paths) = paths() else {
        eprintln!("skipping: MOLD_TEST_PULID_ASSETS is unset or incomplete");
        return;
    };
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skipping: a device peak is only separable from host memory on CUDA");
        return;
    };
    let charged = mold_core::identity::EXTRACTION_DEVICE_PEAK_BYTES;
    let recorded = mold_core::identity::EXTRACTION_DEVICE_PEAK_MEASURED_BYTES;
    // What the charge holds back for photographs this run does not supply.
    let allowance = mold_core::identity::EXTRACTION_DEVICE_MULTI_IMAGE_ALLOWANCE_BYTES;
    let single_image_charge = charged - allowance;

    // The baseline must be taken on a context that has never run this phase.
    // There is deliberately NO warm-up: see the doc comment above.
    let Some(idle) = mold_inference::device::free_vram_bytes(0) else {
        eprintln!("skipping: this device reports no VRAM accounting");
        return;
    };
    forget_cached_identities();

    let photo = std::fs::read(
        testdata_dir()
            .join("faces")
            .join("frank-rubio-official-portrait.jpg"),
    )
    .expect("the portrait fixture is committed");

    // High-water across the whole extraction, sampled from a peer thread: the
    // allocator's peak is what the charge has to cover, and it lives inside
    // the call rather than at its edges.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let low = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(idle));
    let sampler = {
        let stop = stop.clone();
        let low = low.clone();
        std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(free) = mold_inference::device::free_vram_bytes(0) {
                    low.fetch_min(free, std::sync::atomic::Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        })
    };
    let extraction = extract_identity_embeddings(&paths, &[photo.as_slice()], true, &device)
        .expect("the cold device extraction runs");
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    sampler.join().expect("the sampler thread joins");
    assert!(
        extraction.extracted,
        "the measured run must be the cold one, not a cache hit"
    );

    let measured = idle.saturating_sub(low.load(std::sync::atomic::Ordering::Relaxed));
    eprintln!(
        "cold device peak {measured} bytes, charged {charged}, recorded {recorded} \
         ({:.1}% of charge)",
        measured as f64 / charged as f64 * 100.0
    );
    assert!(
        measured > 0,
        "a peak of zero means the allocator was already warm — this test must \
         run on a fresh context, with no warm-up, through the production entry \
         point"
    );
    assert!(
        measured <= charged,
        "the charge {charged} must cover the measured {measured}; a device \
         admitted on an under-charge OOMs mid-extraction"
    );
    assert!(
        measured * 10 >= single_image_charge * 9,
        "the charge {charged} less its {allowance}-byte multi-photograph \
         allowance is more than 10% above the measured {measured}; an \
         over-charge parks renders the device could actually run"
    );
}

/// The charge covers the largest admissible photograph set, not just the one
/// the measurement above supplies.
///
/// Stated arithmetically rather than by extracting four photographs, because
/// the four-photograph peak is the one-photograph peak plus a term the
/// composer's own structure fixes: the tower is built once and run per
/// photograph, so only the retained hidden states scale.
#[test]
fn the_charge_covers_the_largest_admissible_photograph_set() {
    let charged = mold_core::identity::EXTRACTION_DEVICE_PEAK_BYTES;
    let measured = mold_core::identity::EXTRACTION_DEVICE_PEAK_MEASURED_BYTES;
    let allowance = mold_core::identity::EXTRACTION_DEVICE_MULTI_IMAGE_ALLOWANCE_BYTES;
    assert_eq!(
        allowance,
        mold_inference::identity::extraction::EXTRACTION_RETAINED_BYTES_PER_IMAGE
            * (mold_core::identity::ID_IMAGES_MAX as u64 - 1),
        "the allowance must be the per-photograph charge, not a second number"
    );
    assert!(
        charged >= measured + allowance,
        "the charge {charged} must cover the largest set ({measured} + {allowance})"
    );
    // And not by so much that the single-photograph band above cannot hold.
    assert!(
        (charged - allowance) * 9 <= measured * 10,
        "the netted charge is still more than 10% above one photograph"
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

/// The batch case, end to end: N threads resolving ONE photograph at once must
/// compose it once and all receive the identical frozen identity.
///
/// This is the failure the per-photograph cache alone does not prevent. A
/// `batch_size = 4` parent is prepared before fan-out and its children are
/// dispatched to several GPU worker threads, each resolving its own identity —
/// so without the single flight all four miss the cold cache together, run the
/// whole stack four times, and (on different devices) end up conditioned on
/// four faces that differ at the device tolerance. The fingerprints below are
/// the check that matters: equal-within-tolerance is not equal.
#[test]
#[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
fn concurrent_siblings_of_one_parent_extract_once_and_agree_exactly() {
    use mold_inference::identity::extraction::{
        extract_identity_embeddings, forget_cached_identities, identity_extraction_count,
    };

    const SIBLINGS: usize = 4;
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
    let before = identity_extraction_count();
    let fingerprints: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..SIBLINGS)
            .map(|_| {
                let paths = &paths;
                let device = &device;
                let photo = &photo;
                scope.spawn(move || {
                    extract_identity_embeddings(paths, &[photo.as_slice()], true, device)
                        .expect("every sibling resolves")
                        .embedding
                        .fingerprint()
                        .to_string()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    assert_eq!(
        identity_extraction_count() - before,
        1,
        "{SIBLINGS} concurrent siblings must compose the identity exactly once"
    );
    let unique: std::collections::BTreeSet<&String> = fingerprints.iter().collect();
    assert_eq!(
        unique.len(),
        1,
        "every sibling must carry the byte-identical identity: {fingerprints:?}"
    );
}

/// A true-CFG request whose photograph is already cached needs the IDFormer
/// and nothing else. Loading SCRFD and ArcFace for it places ~278 MB on the
/// device to run neither.
///
/// Observed through the composer's own stage callback, which is the only
/// evidence that distinguishes "did not need them" from "needed them and was
/// fast": a face-stack load would have to emit its work somewhere, and the
/// stage the caller sees is `IdFormerBuild` alone.
#[test]
#[ignore = "requires the pinned PuLID checkpoints via MOLD_TEST_PULID_ASSETS"]
fn an_uncond_only_miss_opens_neither_face_graph() {
    use mold_inference::identity::extraction::{
        extract_identity_embeddings, forget_cached_identities, identity_extraction_count,
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
    // An ordinary identity render first: the photograph lands in the cache and
    // no unconditional identity is computed, because nothing asked for one.
    let plain = extract_identity_embeddings(&paths, &[photo.as_slice()], false, &device)
        .expect("the ordinary render resolves");
    assert!(plain.extracted);
    assert!(!plain.embedding.has_uncond());

    // Now the same face with true CFG. Only the uncond is missing.
    let before = identity_extraction_count();
    let started = std::time::Instant::now();
    let branched = extract_identity_embeddings(&paths, &[photo.as_slice()], true, &device)
        .expect("the true-CFG render resolves");
    let elapsed = started.elapsed();
    assert!(branched.embedding.has_uncond(), "the uncond was computed");
    assert!(branched.extracted, "an uncond miss is a real computation");
    assert_eq!(identity_extraction_count() - before, 1);
    assert_eq!(
        branched.embedding.source_sha256s(),
        plain.embedding.source_sha256s(),
        "the cached photograph must be reused, not re-detected"
    );

    // The face stack is ~278 MB of graph decode plus two forwards. Its absence
    // is what makes this fast; the bound is generous because the IDFormer's
    // own 605 MB build is still paid.
    eprintln!("uncond-only miss: {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    assert!(
        elapsed < std::time::Duration::from_millis(1500),
        "an uncond-only miss loaded more than the IDFormer: {elapsed:?}"
    );

    // And a third call needs nothing at all.
    let before = identity_extraction_count();
    let warm = extract_identity_embeddings(&paths, &[photo.as_slice()], true, &device)
        .expect("the repeat resolves");
    assert!(!warm.extracted, "everything was cached");
    assert_eq!(identity_extraction_count(), before);
    assert_eq!(
        warm.embedding.fingerprint(),
        branched.embedding.fingerprint()
    );
}
