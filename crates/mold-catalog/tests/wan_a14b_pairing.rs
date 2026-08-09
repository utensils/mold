//! End-to-end catalog behaviour for Civitai's separately-published Wan 2.2
//! A14B experts, over fixture bodies captured from the real Civitai API
//! (`/api/v1/models?baseModels=Wan Video 2.2 {T2V,I2V}-A14B`, February 2026
//! shapes captured 2026-08-08).
//!
//! The invariants under test:
//! - one confidently-paired High/Low version pair emits ONE supported row
//!   (from the high-noise version) whose recipe carries both experts;
//! - versions whose counterpart cannot be identified stay visible but
//!   unsupported — never a silent single-expert install;
//! - GGUF-only publications stay out entirely (safetensors-only path);
//! - intent synthesis, the installed sidecar, and completeness checks all
//!   fail closed on a half-pair and resume the missing half on repair.

use mold_catalog::entry::{CatalogEntry, Kind, RecipeFileRole};
use mold_catalog::normalizer::{from_civitai_search_entries, CivitaiItem};
use mold_catalog::synthesis::{synthesize_intent, SynthesisError};
use mold_catalog::wan_a14b;
use std::path::Path;

fn fixture_items(name: &str) -> Vec<CivitaiItem> {
    let raw = std::fs::read_to_string(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("fixture readable");
    let value: serde_json::Value = serde_json::from_str(&raw).expect("fixture parses");
    serde_json::from_value(value["items"].clone()).expect("items parse as CivitaiItem")
}

fn search_entries(name: &str) -> Vec<CatalogEntry> {
    fixture_items(name)
        .into_iter()
        .flat_map(from_civitai_search_entries)
        .collect()
}

fn paired_t2v_entry() -> CatalogEntry {
    search_entries("civitai_wan22_a14b_t2v_search.json")
        .into_iter()
        .find(|entry| entry.id.as_str() == "cv:2057171")
        .expect("official repack high-noise fp8 pair present")
}

/// The captured T2V page: the official repack pairs (fp8 + fp16) and the
/// KJ FP8 pair each emit one supported row from their high-noise version;
/// low-noise versions are subsumed; merged/unmarked republications stay
/// visible but unsupported; GGUF-only models emit nothing.
#[test]
fn t2v_search_page_offers_one_runnable_install_per_pair() {
    let entries = search_entries("civitai_wan22_a14b_t2v_search.json");

    let supported: Vec<&str> = entries
        .iter()
        .filter(|e| e.supported)
        .map(|e| e.id.as_str())
        .collect();
    assert_eq!(
        supported,
        vec!["cv:2057171", "cv:2057999", "cv:2545187"],
        "exactly the high-noise halves of the three safetensors pairs"
    );

    // Low-noise siblings never emit their own row — one pair, one install.
    for low_id in ["cv:2057100", "cv:2057683", "cv:2545318"] {
        assert!(
            entries.iter().all(|e| e.id.as_str() != low_id),
            "{low_id} is subsumed by its high-noise sibling"
        );
    }

    // Merged / unmarked republications (Magic-Wan-Image) stay visible but
    // un-installable, with the unpaired predicate naming the reason.
    let unpaired: Vec<&CatalogEntry> = entries.iter().filter(|e| !e.supported).collect();
    assert!(!unpaired.is_empty(), "unmarked versions stay visible");
    for entry in &unpaired {
        assert!(
            wan_a14b::entry_is_unpaired_a14b(entry),
            "{} must read as unpaired",
            entry.id.as_str()
        );
        assert!(wan_a14b::unpaired_reason(&entry.name).contains("pair"));
    }

    // GGUF-only publications ("Wan 2.2 T2V A14B GGUF", "Rapid WAN 2.2 T2V
    // GGUF") never surface on the safetensors-only Civitai path.
    assert!(entries
        .iter()
        .all(|e| !e.name.to_ascii_lowercase().contains("gguf")));

    for entry in &entries {
        assert_eq!(entry.sub_family.as_deref(), Some("wan22-t2v-a14b"));
        assert_eq!(entry.family.as_str(), "wan");
    }
    for entry in entries.iter().filter(|e| e.supported) {
        assert!(
            entry.companions.contains(&"wan-umt5".to_string())
                && entry.companions.contains(&"wan21-vae".to_string()),
            "A14B pairs take the UMT5 encoder and the 2.1 VAE"
        );
    }
}

/// The captured I2V page: the TrueVision pairs match by exact name key,
/// the official repack fp8 pair survives its real `_sd`/`_scd` suffix typo
/// through the size-band fallback, and every GGUF model stays out.
#[test]
fn i2v_search_page_pairs_across_naming_styles() {
    let entries = search_entries("civitai_wan22_a14b_i2v_search.json");

    let supported: Vec<&str> = entries
        .iter()
        .filter(|e| e.supported)
        .map(|e| e.id.as_str())
        .collect();
    assert_eq!(
        supported,
        vec![
            "cv:2959309", // SnatchKiss High v11
            "cv:2769496", // BoundBite High v10
            "cv:2557969", // SynthSeduction High v9
            "cv:2057465", // official repack i2v fp8 (suffix typo)
            "cv:2058318", // official repack i2v fp16
        ]
    );

    for entry in entries.iter().filter(|e| e.supported) {
        assert_eq!(entry.sub_family.as_deref(), Some("wan22-i2v-a14b"));
        let files = &entry.download_recipe.files;
        assert_eq!(files.len(), 2, "{}", entry.id.as_str());
        assert!(files[0].role.is_none());
        assert_eq!(files[1].role, Some(RecipeFileRole::LowNoiseTransformer));
    }
}

/// The paired recipe drives everything downstream: primary = high expert,
/// the low expert keeps its own version's directory, sizes are per-file,
/// and the entry size is the full download.
#[test]
fn paired_recipe_carries_both_experts_with_honest_sizes() {
    let entry = paired_t2v_entry();
    let files = &entry.download_recipe.files;
    assert_eq!(files.len(), 2);
    assert!(files[0].role.is_none());
    assert_eq!(
        files[0].dest,
        "{family}/civitai/2057171/wanVideo22_t2vHighNoise14B.safetensors"
    );
    assert_eq!(files[1].role, Some(RecipeFileRole::LowNoiseTransformer));
    assert_eq!(
        files[1].dest,
        "{family}/civitai/2057100/wanVideo22_t2vLowNoise14B.safetensors"
    );
    assert!(files[1]
        .url
        .contains("civitai.com/api/download/models/2057100"));
    let sum: u64 = files.iter().filter_map(|f| f.size_bytes).sum();
    assert!(entry.size_bytes.unwrap().abs_diff(sum) <= 2);
}

/// Synthesis records the low-noise expert path; a half-pair entry cannot
/// synthesize an intent at all — the fail-closed backstop behind the
/// normalizer's `supported: false`.
#[test]
fn synthesis_requires_both_experts_for_a14b() {
    let models_dir = Path::new("/tmp/mold-test-models");
    let entry = paired_t2v_entry();
    let intent = synthesize_intent(&entry, models_dir).expect("paired entry synthesizes");
    assert_eq!(
        intent.primary_recipe_path,
        models_dir.join("cv-2057171/wan/civitai/2057171/wanVideo22_t2vHighNoise14B.safetensors")
    );
    assert_eq!(
        intent.low_noise_recipe_path.as_deref(),
        Some(
            models_dir
                .join("cv-2057171/wan/civitai/2057100/wanVideo22_t2vLowNoise14B.safetensors")
                .as_path()
        )
    );

    // Strip the low-noise half: synthesis must refuse, naming the missing
    // expert.
    let mut half = entry.clone();
    half.download_recipe
        .files
        .retain(|file| file.role.is_none());
    let err = synthesize_intent(&half, models_dir).expect_err("half-pair must not synthesize");
    match err {
        SynthesisError::MissingExpertCounterpart { id, missing } => {
            assert_eq!(id, "cv:2057171");
            assert_eq!(missing, "low-noise");
        }
        other => panic!("expected MissingExpertCounterpart, got {other:?}"),
    }
}

/// A Wan LoRA published against an A14B base model is a single adapter
/// file, not an expert pair — the fail-closed check must not apply.
#[test]
fn a14b_loras_are_not_forced_into_pairs() {
    let mut entry = paired_t2v_entry();
    entry.kind = Kind::Lora;
    entry
        .download_recipe
        .files
        .retain(|file| file.role.is_none());
    entry.companions.clear();
    assert!(synthesize_intent(&entry, Path::new("/tmp")).is_ok());
}

/// Serialized intents from before pairing (no `low_noise_recipe_path`)
/// keep deserializing — the field is additive.
#[test]
fn pre_pairing_intents_deserialize_with_absent_low_noise() {
    let entry = paired_t2v_entry();
    let intent = synthesize_intent(&entry, Path::new("/tmp")).unwrap();
    let mut value = serde_json::to_value(&intent).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("low_noise_recipe_path");
    let read: mold_catalog::synthesis::CatalogModelIntent =
        serde_json::from_value(value).expect("older intent shape deserializes");
    assert_eq!(read.low_noise_recipe_path, None);
}

/// The sidecar records both halves of the pair and reports the install
/// complete only when both files are on disk at their declared sizes; a
/// half-pair is "not installed", which is exactly what routes a re-install
/// into resuming the missing half.
#[test]
fn sidecar_pair_is_installed_only_when_both_experts_are_on_disk() {
    let entry = paired_t2v_entry();
    let sidecar = mold_catalog::sidecar::sidecar_from_entry(
        &entry,
        "wan/civitai/2057171/wanVideo22_t2vHighNoise14B.safetensors".into(),
    );
    assert_eq!(
        sidecar.low_noise_filename_rel.as_deref(),
        Some("wan/civitai/2057100/wanVideo22_t2vLowNoise14B.safetensors")
    );
    let high_size = entry.download_recipe.files[0].size_bytes.unwrap();
    let low_size = entry.download_recipe.files[1].size_bytes.unwrap();
    assert_eq!(sidecar.primary_size_bytes, Some(high_size));
    assert_eq!(sidecar.low_noise_size_bytes, Some(low_size));

    // Shrink the declared sizes so the test can write real files.
    let mut sidecar = sidecar;
    sidecar.primary_size_bytes = Some(4);
    sidecar.low_noise_size_bytes = Some(3);
    sidecar.size_bytes = Some(7);

    let dir = tempfile::tempdir().unwrap();
    let install = dir.path().join("cv-2057171");
    let high = install.join("wan/civitai/2057171/wanVideo22_t2vHighNoise14B.safetensors");
    let low = install.join("wan/civitai/2057100/wanVideo22_t2vLowNoise14B.safetensors");
    std::fs::create_dir_all(high.parent().unwrap()).unwrap();
    std::fs::create_dir_all(low.parent().unwrap()).unwrap();

    // High-noise only: a half-pair is not installed.
    std::fs::write(&high, b"high").unwrap();
    assert!(
        mold_catalog::sidecar::primary_path_if_present(&install, &sidecar).is_none(),
        "a half-pair must never report installed"
    );

    // Truncated low half: still incomplete.
    std::fs::write(&low, b"lo").unwrap();
    assert!(mold_catalog::sidecar::primary_path_if_present(&install, &sidecar).is_none());

    // Both complete: installed, and the primary path is the high expert.
    std::fs::write(&low, b"low").unwrap();
    assert_eq!(
        mold_catalog::sidecar::primary_path_if_present(&install, &sidecar),
        Some(high.clone())
    );

    // Round-trips through disk with the pair fields intact.
    let path = install.join(mold_catalog::sidecar::SIDECAR_FILENAME);
    mold_catalog::sidecar::write_sidecar(&path, &sidecar).unwrap();
    assert_eq!(mold_catalog::sidecar::read_sidecar(&path).unwrap(), sidecar);
}

/// Sidecars written before pairing (no pair fields) keep deserializing and
/// keep their original single-file completeness semantics.
#[test]
fn pre_pairing_sidecars_deserialize_and_stay_single_file() {
    let entry = paired_t2v_entry();
    let sidecar = mold_catalog::sidecar::sidecar_from_entry(&entry, "x.safetensors".into());
    let mut value = serde_json::to_value(&sidecar).unwrap();
    let object = value.as_object_mut().unwrap();
    object.remove("primary_size_bytes");
    object.remove("low_noise_filename_rel");
    object.remove("low_noise_size_bytes");
    let read: mold_catalog::sidecar::CatalogSidecar =
        serde_json::from_value(value).expect("older sidecar shape deserializes");
    assert_eq!(read.primary_size_bytes, None);
    assert_eq!(read.low_noise_filename_rel, None);
    assert_eq!(read.low_noise_size_bytes, None);
}

/// The low-noise declaration obeys the same containment rules as the
/// primary: escapes are dropped, which then reads as an incomplete (never
/// installed) pair.
#[test]
fn low_noise_declaration_cannot_escape_the_install_directory() {
    let entry = paired_t2v_entry();
    let mut sidecar = mold_catalog::sidecar::sidecar_from_entry(&entry, "x.safetensors".into());
    let dir = tempfile::tempdir().unwrap();
    for invalid in ["/tmp/escape.safetensors", "../escape.safetensors", ""] {
        sidecar.low_noise_filename_rel = Some(invalid.into());
        assert!(
            mold_catalog::sidecar::low_noise_path(dir.path(), &sidecar).is_none(),
            "accepted invalid low-noise rel {invalid:?}"
        );
        assert!(mold_catalog::sidecar::primary_path_if_present(dir.path(), &sidecar).is_none());
    }
}
