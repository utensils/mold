//! Pairing of Civitai's separately-published Wan 2.2 A14B experts.
//!
//! A14B is a *pair*: two complete transformers the sampler alternates
//! between at a fixed timestep (`mold-inference`'s `wan/experts.rs`).
//! Civitai publishes the high- and low-noise experts as **separate model
//! versions** of one model — `"HIGH Q5_0"` / `"LOW Q5_0"`,
//! `"t2v_high_noise_14B"` / `"t2v_low_noise_14B"`, `"SnatchKiss High v11"`
//! / `"SnatchKiss Low v11"` (all real, captured in
//! `tests/fixtures/civitai_wan22_*`). Installing one version alone yields a
//! model that cannot run: the I2V half fails loudly at load, and the T2V
//! half is shape-indistinguishable from a single-expert Wan 2.1 14B and
//! would render silently wrong.
//!
//! This module locates the counterpart version on the same Civitai model
//! with confidence, or fails closed:
//!
//! 1. classify the clicked version as high- or low-noise from unambiguous
//!    `high`/`low` name markers (version name first; the primary file name
//!    is consulted only when the version name carries *no* marker — a
//!    version named with BOTH markers is an explicit high+low bundle and
//!    never falls back);
//! 2. among the model's *other* public versions with the same `baseModel`
//!    and a loadable safetensors Model file, keep only those that classify
//!    as the opposite expert;
//! 3. prefer the candidate whose name matches the clicked version's name
//!    with the marker swapped (`"HIGH Q5_0"` ↔ `"LOW Q5_0"`);
//! 4. otherwise fall back to the unique candidate whose file size is
//!    within ±10% of the primary — the two experts of one precision are
//!    near-identical in size, while fp16↔fp8 and GGUF quant rungs differ
//!    by ≥25% — AND whose names are semantically compatible with the
//!    clicked version's (same numeric tokens, near-identical text modulo
//!    the marker). The band alone absorbs real uploader typos such as
//!    `i2v_low_noise_14B_fp8_scd` ↔ `i2v_high_noise_14B_fp8_sd`, but a
//!    same-precision sibling from an *unrelated* finetune revision is also
//!    nearly identical in size, so size can never pair on its own;
//! 5. anything else — no marker, both markers, zero or multiple surviving
//!    candidates — is an [`A14bPairingError`], never a guess.
//!
//! The experts of a pair are byte-different files with identical tensor
//! shapes, so no header sniff can tell high from low; the name markers are
//! the only signal Civitai carries. `WanExperts` validates at bind time
//! that both files declare the same architecture, which is the final gate
//! behind this heuristic.

use crate::normalizer::{CivitaiFile, CivitaiItem, CivitaiVersion};

/// Civitai `baseModel` → A14B sub-family slug. `None` for every other base
/// model, including single-expert Wan checkpoints.
pub fn a14b_sub_family(base_model: &str) -> Option<&'static str> {
    match base_model {
        "Wan Video 2.2 T2V-A14B" => Some("wan22-t2v-a14b"),
        "Wan Video 2.2 I2V-A14B" => Some("wan22-i2v-a14b"),
        _ => None,
    }
}

/// Whether a catalog `sub_family` slug names an A14B expert pair. The
/// consumers that must fail closed on a half-pair (`synthesize_intent`,
/// `installed_intent_from_sidecar`) key off this.
pub fn is_a14b_sub_family(sub_family: Option<&str>) -> bool {
    matches!(sub_family, Some("wan22-t2v-a14b" | "wan22-i2v-a14b"))
}

/// Which expert a version publishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertRole {
    HighNoise,
    LowNoise,
}

impl ExpertRole {
    pub fn counterpart(self) -> Self {
        match self {
            Self::HighNoise => Self::LowNoise,
            Self::LowNoise => Self::HighNoise,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::HighNoise => "high-noise",
            Self::LowNoise => "low-noise",
        }
    }
}

/// Why a confident pairing could not be built. Every variant is a
/// fail-closed outcome: the entry stays un-installable with this reason
/// rather than becoming a silent single-expert install.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum A14bPairingError {
    /// Neither the version name nor its primary file name carries an
    /// unambiguous `high`/`low` marker (or it carries both). Covers
    /// merged "all-in-one" republications ("Q4_K_M Rapid Base") that mold
    /// cannot verify are a complete pair.
    #[error(
        "cannot tell which A14B expert this version is: no unambiguous \
         high/low marker in {name:?}"
    )]
    UnclassifiedExpert { name: String },

    /// The version classified cleanly but no sibling version of the same
    /// model publishes the counterpart expert.
    #[error(
        "the {missing} expert counterpart of {name:?} was not found on the \
         same Civitai model"
    )]
    NoCounterpart { name: String, missing: &'static str },

    /// More than one sibling version could be the counterpart and neither
    /// the name-key match nor the size band disambiguates.
    #[error(
        "multiple sibling versions could be the {missing} expert counterpart \
         of {name:?}; refusing to guess"
    )]
    AmbiguousCounterpart { name: String, missing: &'static str },

    /// A counterpart matched by name, but its file size is too far from
    /// the primary's to be the same-precision expert.
    #[error(
        "the {missing} counterpart candidate for {name:?} has a file size \
         incompatible with the primary expert"
    )]
    CounterpartSizeMismatch { name: String, missing: &'static str },
}

/// One side of a resolved pair: the version that owns the file, and the
/// file itself.
#[derive(Clone, Debug)]
pub struct PairedExpert<'a> {
    pub version_id: u64,
    pub file: &'a CivitaiFile,
}

/// A confidently-paired A14B install: the high-noise expert is the
/// primary (`ModelConfig.transformer`), the low-noise expert fills
/// `low_noise_transformer`.
#[derive(Clone, Debug)]
pub struct A14bPair<'a> {
    pub high: PairedExpert<'a>,
    pub low: PairedExpert<'a>,
    /// The role the *requested* version plays in the pair. Search emits
    /// the pair only from the high-noise version so one pair is one row;
    /// a direct `cv:<low-id>` fetch still resolves to the full pair.
    pub requested_role: ExpertRole,
}

/// Counterpart file-size band: same-precision experts are near-identical
/// (every captured pair differs by <0.1%), while fp16↔fp8 differ ~2× and
/// adjacent GGUF quant rungs differ ≥25%.
const SIZE_BAND: f64 = 0.10;

/// What one name says about the expert role. `NoMarker` and `Ambiguous`
/// are deliberately distinct: an unmarked *version* name may still be
/// resolved by the file name, but a name carrying BOTH markers is an
/// explicit high+low bundle — falling back to a one-sided file name would
/// classify a bundle as a single expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NameMarker {
    High,
    Low,
    NoMarker,
    Ambiguous,
}

/// Classify a version as one expert of the pair from its name markers.
/// The primary file's name is consulted only when the version name carries
/// no marker at all; an ambiguous (both-marker) version name fails closed.
/// `None` when no unambiguous marker exists.
pub fn classify_expert(version: &CivitaiVersion, file: &CivitaiFile) -> Option<ExpertRole> {
    let version_marker = version
        .name
        .as_deref()
        .map(classify_name)
        .unwrap_or(NameMarker::NoMarker);
    let marker = match version_marker {
        NameMarker::NoMarker => classify_name(&file.name),
        decided => decided,
    };
    match marker {
        NameMarker::High => Some(ExpertRole::HighNoise),
        NameMarker::Low => Some(ExpertRole::LowNoise),
        NameMarker::NoMarker | NameMarker::Ambiguous => None,
    }
}

fn classify_name(name: &str) -> NameMarker {
    let lower = name.to_ascii_lowercase();
    match (lower.contains("high"), lower.contains("low")) {
        (true, false) => NameMarker::High,
        (false, true) => NameMarker::Low,
        // Merged republications, plain "v2.0-fp8" — no marker at all.
        (false, false) => NameMarker::NoMarker,
        // "high+low bundle" — explicitly both experts, never one.
        (true, true) => NameMarker::Ambiguous,
    }
}

/// Normalized comparison key: lowercase with every `high`/`low` marker
/// replaced by a shared placeholder, so `"HIGH Q5_0"` and `"LOW Q5_0"`
/// (or `"t2v_high_noise_14B"` / `"t2v_low_noise_14B"`) produce the same
/// key.
fn expert_key(name: &str) -> String {
    name.to_ascii_lowercase()
        .replace("high", "{expert}")
        .replace("low", "{expert}")
}

fn version_key(version: &CivitaiVersion, file: &CivitaiFile) -> (String, String) {
    (
        expert_key(version.name.as_deref().unwrap_or_default()),
        expert_key(&file.name),
    )
}

/// Maximum Levenshtein distance between two expert keys for the size-band
/// fallback to consider them the same artifact. 1–2 characters covers real
/// uploader typos (`…_fp8_sd` ↔ `…_fp8_scd`); different finetune names
/// ("BoundBite" vs "SynthSeduction") are far beyond it.
const KEY_EDIT_DISTANCE_MAX: usize = 2;

/// The digit runs of a key, in order. Two keys naming the same artifact
/// modulo the high/low marker carry identical numeric tokens (`14B`,
/// `fp8`, `v10`); a different finetune *version* differs here even when
/// the letters are one typo apart — `v10` vs `v9` must never pair.
fn digit_runs(key: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current = String::new();
    for ch in key.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut row = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let substitute = prev[j] + usize::from(ca != cb);
            row[j + 1] = substitute.min(prev[j + 1] + 1).min(row[j] + 1);
        }
        std::mem::swap(&mut prev, &mut row);
    }
    prev[b.len()]
}

/// Whether two marker-normalized keys name the same artifact: identical
/// numeric tokens AND near-identical text. This is what keeps the
/// size-band fallback from pairing a high-noise expert with a same-size,
/// same-precision low-noise expert of an *unrelated* finetune revision
/// when the intended counterpart version has vanished upstream.
fn keys_semantically_compatible(a: &str, b: &str) -> bool {
    digit_runs(a) == digit_runs(b) && levenshtein(a, b) <= KEY_EDIT_DISTANCE_MAX
}

/// Semantic name compatibility over both the version-name and file-name
/// keys — the same finetune identity and version modulo the high/low
/// marker.
fn version_keys_semantically_compatible(own: &(String, String), other: &(String, String)) -> bool {
    keys_semantically_compatible(&own.0, &other.0) && keys_semantically_compatible(&own.1, &other.1)
}

fn sizes_compatible(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) if a > 0.0 && b > 0.0 => {
            let ratio = a / b;
            (1.0 - SIZE_BAND..=1.0 + SIZE_BAND).contains(&ratio)
        }
        // A missing size cannot raise confidence; only the exact-name key
        // match may pair without it.
        _ => false,
    }
}

/// The safetensors Model file a version would install — the normalizer's
/// own primary pick, so pairing candidates and installs can never select
/// different files. GGUF-only versions return `None` and are never
/// pairing candidates (the catalog's Civitai path is safetensors-only),
/// and neither are 4-bit (NF4/NVFP4) safetensors the Wan loader cannot
/// read — pairing one in would download a counterpart that fails at load.
fn safetensors_model_file(version: &CivitaiVersion) -> Option<&CivitaiFile> {
    crate::normalizer::pick_safetensors(&version.files)
        .filter(|file| crate::normalizer::wan_precision_is_loadable(file))
}

fn version_is_public(version: &CivitaiVersion) -> bool {
    crate::normalizer::version_is_public(version)
}

/// Locate the counterpart expert for `version` among `item`'s versions and
/// return the resolved pair, or a typed fail-closed reason.
///
/// `version` must already have a safetensors Model file (`primary_file`) —
/// the normalizer's `pick_safetensors` guarantees that before calling.
pub fn pair_experts<'a>(
    item: &'a CivitaiItem,
    version: &'a CivitaiVersion,
    primary_file: &'a CivitaiFile,
) -> Result<A14bPair<'a>, A14bPairingError> {
    let display_name = version
        .name
        .clone()
        .unwrap_or_else(|| primary_file.name.clone());
    let role = classify_expert(version, primary_file).ok_or_else(|| {
        A14bPairingError::UnclassifiedExpert {
            name: display_name.clone(),
        }
    })?;
    let missing = role.counterpart().label();

    let candidates: Vec<(&CivitaiVersion, &CivitaiFile)> = item
        .model_versions
        .iter()
        .filter(|sibling| sibling.id != version.id)
        .filter(|sibling| sibling.base_model == version.base_model)
        .filter(|sibling| version_is_public(sibling))
        .filter_map(|sibling| safetensors_model_file(sibling).map(|file| (sibling, file)))
        .filter(|(sibling, file)| classify_expert(sibling, file) == Some(role.counterpart()))
        .collect();

    if candidates.is_empty() {
        return Err(A14bPairingError::NoCounterpart {
            name: display_name,
            missing,
        });
    }

    // Stage 1: exact name-key match (version name and file name both
    // reduce to the same key with the marker swapped).
    let own_key = version_key(version, primary_file);
    let key_matches: Vec<&(&CivitaiVersion, &CivitaiFile)> = candidates
        .iter()
        .filter(|(sibling, file)| version_key(sibling, file) == own_key)
        .collect();
    let chosen = match key_matches.as_slice() {
        [one] => {
            // A name-key match at a wildly different size is a different
            // artifact wearing the same name — refuse it.
            if one.1.size_kb.is_some()
                && primary_file.size_kb.is_some()
                && !sizes_compatible(one.1.size_kb, primary_file.size_kb)
            {
                return Err(A14bPairingError::CounterpartSizeMismatch {
                    name: display_name,
                    missing,
                });
            }
            **one
        }
        [] => {
            // Stage 2: unique candidate within the size band AND with
            // semantically compatible names. Covers real uploader typos
            // where the two version names differ slightly beyond the
            // marker (`…_fp8_scd` ↔ `…_fp8_sd`) while refusing a
            // same-size sibling from an unrelated finetune revision —
            // one precision's experts are all near-identical in size, so
            // the band alone cannot tell counterpart from stranger when
            // the intended counterpart version has vanished upstream.
            let mut in_band = candidates.iter().filter(|(sibling, file)| {
                sizes_compatible(file.size_kb, primary_file.size_kb)
                    && version_keys_semantically_compatible(&own_key, &version_key(sibling, file))
            });
            match (in_band.next(), in_band.next()) {
                (Some(one), None) => *one,
                (None, _) => {
                    return Err(A14bPairingError::NoCounterpart {
                        name: display_name,
                        missing,
                    })
                }
                (Some(_), Some(_)) => {
                    return Err(A14bPairingError::AmbiguousCounterpart {
                        name: display_name,
                        missing,
                    })
                }
            }
        }
        _ => {
            return Err(A14bPairingError::AmbiguousCounterpart {
                name: display_name,
                missing,
            })
        }
    };

    let own = PairedExpert {
        version_id: version.id,
        file: primary_file,
    };
    let other = PairedExpert {
        version_id: chosen.0.id,
        file: chosen.1,
    };
    let (high, low) = match role {
        ExpertRole::HighNoise => (own, other),
        ExpertRole::LowNoise => (other, own),
    };
    Ok(A14bPair {
        high,
        low,
        requested_role: role,
    })
}

/// Which counterpart a lone expert file is missing, for error messages.
/// `"expert"` when the file name itself carries no confident marker.
pub fn missing_counterpart_label(primary_name: &str) -> &'static str {
    match classify_name(primary_name) {
        NameMarker::High => "low-noise",
        NameMarker::Low => "high-noise",
        NameMarker::NoMarker | NameMarker::Ambiguous => "expert",
    }
}

/// True for a normalized A14B checkpoint entry whose recipe does NOT carry
/// the low-noise counterpart — i.e. pairing failed and the entry is
/// deliberately un-installable. Lets the download surfaces name the real
/// reason instead of a generic "unsupported".
pub fn entry_is_unpaired_a14b(entry: &crate::entry::CatalogEntry) -> bool {
    entry.family == crate::families::Family::Wan
        && entry.kind == crate::entry::Kind::Checkpoint
        && is_a14b_sub_family(entry.sub_family.as_deref())
        && !entry
            .download_recipe
            .files
            .iter()
            .any(|file| file.role == Some(crate::entry::RecipeFileRole::LowNoiseTransformer))
}

/// User-facing reason for an un-installable A14B entry whose counterpart
/// could not be identified. Derived from the entry's own metadata so the
/// download route can surface it without re-running the live pairing.
pub fn unpaired_reason(entry_name: &str) -> String {
    format!(
        "Wan 2.2 A14B denoises with a pair of experts (high- and low-noise), and mold \
         could not confidently identify the counterpart expert for '{entry_name}' on its \
         Civitai model page. Installing a single expert would produce a model that cannot \
         generate correctly, so this version is not installable. Pick a version published \
         as a matching High/Low pair, or install the built-in `wan22-t2v-a14b` / \
         `wan22-i2v-a14b` models."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizer::CivitaiItem;

    fn load_item(fixture: &str) -> CivitaiItem {
        let path = format!("{}/tests/fixtures/{fixture}", env!("CARGO_MANIFEST_DIR"));
        let raw = std::fs::read_to_string(path).expect("fixture readable");
        serde_json::from_str(&raw).expect("fixture parses as CivitaiItem")
    }

    fn load_search_item(fixture: &str, model_id: u64) -> CivitaiItem {
        let path = format!("{}/tests/fixtures/{fixture}", env!("CARGO_MANIFEST_DIR"));
        let raw = std::fs::read_to_string(path).expect("fixture readable");
        let value: serde_json::Value = serde_json::from_str(&raw).expect("fixture parses");
        let items: Vec<CivitaiItem> =
            serde_json::from_value(value["items"].clone()).expect("items parse");
        items
            .into_iter()
            .find(|item| item.id == model_id)
            .unwrap_or_else(|| panic!("model {model_id} in fixture"))
    }

    fn version(item: &CivitaiItem, id: u64) -> &CivitaiVersion {
        item.model_versions
            .iter()
            .find(|v| v.id == id)
            .unwrap_or_else(|| panic!("version {id} in fixture"))
    }

    fn pair_for(item: &CivitaiItem, id: u64) -> Result<A14bPair<'_>, A14bPairingError> {
        let v = version(item, id);
        let file = safetensors_model_file(v).expect("version has a safetensors model file");
        pair_experts(item, v, file)
    }

    #[test]
    fn base_model_strings_map_to_a14b_sub_families() {
        assert_eq!(
            a14b_sub_family("Wan Video 2.2 T2V-A14B"),
            Some("wan22-t2v-a14b")
        );
        assert_eq!(
            a14b_sub_family("Wan Video 2.2 I2V-A14B"),
            Some("wan22-i2v-a14b")
        );
        assert_eq!(a14b_sub_family("Wan Video 2.2 TI2V-5B"), None);
        assert_eq!(a14b_sub_family("Wan Video 14B t2v"), None);

        assert!(is_a14b_sub_family(Some("wan22-t2v-a14b")));
        assert!(is_a14b_sub_family(Some("wan22-i2v-a14b")));
        assert!(!is_a14b_sub_family(Some("wan22-ti2v-5b")));
        assert!(!is_a14b_sub_family(None));
    }

    /// Marker vocabulary from the captured fixtures: version-name styles
    /// from four real models. `NoMarker` and `Ambiguous` are distinct
    /// outcomes — only the former may consult the file name.
    #[test]
    fn classification_covers_the_published_marker_styles() {
        for (name, want) in [
            ("HIGH Q5_0", NameMarker::High),
            ("LOW Q5_0", NameMarker::Low),
            ("t2v_high_noise_14B", NameMarker::High),
            ("t2v_low_noise_14B_fp16", NameMarker::Low),
            ("T2V A14B HIGH", NameMarker::High),
            ("SnatchKiss Low v11", NameMarker::Low),
            ("i2v_high_noise_14B_fp8_sd", NameMarker::High),
            // Merged republications carry no marker.
            ("Q4_K_M Rapid Base", NameMarker::NoMarker),
            ("v2.0-fp8", NameMarker::NoMarker),
            // Both markers — an explicit high+low bundle.
            ("high and low bundle", NameMarker::Ambiguous),
        ] {
            assert_eq!(classify_name(name), want, "{name:?}");
        }
    }

    fn version_with_file(version_name: Option<&str>, file_name: &str) -> CivitaiVersion {
        serde_json::from_value(serde_json::json!({
            "id": 1,
            "name": version_name,
            "baseModel": "Wan Video 2.2 T2V-A14B",
            "files": [{
                "id": 10,
                "name": file_name,
                "type": "Model",
                "metadata": { "format": "SafeTensor" },
            }],
        }))
        .expect("synthetic version parses")
    }

    /// The file-name fallback exists for versions with *no* marker; a
    /// version explicitly named as a high+low bundle must never be
    /// classified as one expert just because its primary file name is
    /// one-sided.
    #[test]
    fn ambiguous_version_name_never_falls_back_to_the_filename() {
        let bundle = version_with_file(
            Some("high and low bundle"),
            "wan22_bundle_highNoise.safetensors",
        );
        assert_eq!(classify_expert(&bundle, &bundle.files[0]), None);

        // No marker at all in the version name: the one-sided file name
        // still decides.
        let unmarked = version_with_file(Some("v2.0-fp8"), "wan22_t2v_lowNoise.safetensors");
        assert_eq!(
            classify_expert(&unmarked, &unmarked.files[0]),
            Some(ExpertRole::LowNoise)
        );
        // ... and an ambiguous file name behind an unmarked version name
        // stays unclassified.
        let both = version_with_file(Some("v2.0-fp8"), "wan22_high_and_low.safetensors");
        assert_eq!(classify_expert(&both, &both.files[0]), None);
    }

    /// The official Wan 2.2 repack (model 1817671): the fp8-scaled T2V
    /// pair resolves high↔low across sibling versions via the exact name
    /// key, and never crosses precisions (an fp16 pair coexists on the
    /// same model).
    #[test]
    fn official_repack_t2v_pair_resolves_across_versions() {
        let item = load_item("civitai_wan22_model_1817671.json");

        let pair = pair_for(&item, 2057171).expect("high version pairs");
        assert_eq!(pair.requested_role, ExpertRole::HighNoise);
        assert_eq!(pair.high.version_id, 2057171);
        assert_eq!(pair.low.version_id, 2057100);
        assert_eq!(pair.low.file.name, "wanVideo22_t2vLowNoise14B.safetensors");

        // The same pair resolves when the *low* version is the entry point.
        let pair = pair_for(&item, 2057100).expect("low version pairs");
        assert_eq!(pair.requested_role, ExpertRole::LowNoise);
        assert_eq!(pair.high.version_id, 2057171);
        assert_eq!(pair.low.version_id, 2057100);

        // The fp16 twin pair stays within its own precision.
        let pair = pair_for(&item, 2057999).expect("fp16 high pairs");
        assert_eq!(pair.low.version_id, 2057683);
    }

    /// Real uploader typo on the same model: the I2V fp8 versions are
    /// named `…_fp8_sd` (high) and `…_fp8_scd` (low), so the exact name
    /// key cannot match. The size band (identical sizes) resolves it, and
    /// the fp16 sibling (2× larger) is never mistaken for the counterpart.
    #[test]
    fn size_band_fallback_absorbs_the_fp8_suffix_typo() {
        let item = load_item("civitai_wan22_model_1817671.json");

        let pair = pair_for(&item, 2057465).expect("i2v fp8 high pairs despite the typo");
        assert_eq!(pair.high.version_id, 2057465);
        assert_eq!(pair.low.version_id, 2057270);

        let pair = pair_for(&item, 2058318).expect("i2v fp16 high pairs");
        assert_eq!(pair.low.version_id, 2058116);
    }

    /// Merged "all-in-one" republications and unmarked versions must fail
    /// closed as unclassifiable — even though they might run, mold cannot
    /// verify they are a complete pair.
    #[test]
    fn unmarked_versions_fail_closed() {
        let item = load_item("civitai_wan22_model_1817671.json");
        // Synthesize an unmarked version list: strip the low sibling so the
        // high version has no counterpart at all.
        let mut lone = item.clone();
        lone.model_versions
            .retain(|v| ![2057100, 2057683].contains(&v.id));
        let err = pair_for(&lone, 2057171).unwrap_err();
        assert_eq!(
            err,
            A14bPairingError::NoCounterpart {
                name: "t2v_high_noise_14B".into(),
                missing: "low-noise",
            }
        );
        assert!(err.to_string().contains("low-noise"), "{err}");
    }

    /// A non-public counterpart is not a counterpart.
    #[test]
    fn early_access_counterpart_is_not_used() {
        let mut item = load_item("civitai_wan22_model_1817671.json");
        for v in &mut item.model_versions {
            if v.id == 2057100 {
                v.availability = Some("EarlyAccess".into());
            }
        }
        // The fp16 low remains but is outside the size band and has a
        // different name key → no confident pairing.
        let err = pair_for(&item, 2057171).unwrap_err();
        assert!(
            matches!(err, A14bPairingError::NoCounterpart { .. }),
            "{err:?}"
        );
    }

    /// Two same-key candidates (a re-upload) must refuse rather than pick.
    #[test]
    fn duplicate_counterparts_are_ambiguous() {
        let mut item = load_item("civitai_wan22_model_1817671.json");
        let mut dup = version(&item, 2057100).clone();
        dup.id = 999_999;
        item.model_versions.push(dup);
        let err = pair_for(&item, 2057171).unwrap_err();
        assert!(
            matches!(err, A14bPairingError::AmbiguousCounterpart { .. }),
            "{err:?}"
        );
    }

    /// A name-key match whose file size is wildly different is a different
    /// artifact wearing the counterpart's name.
    #[test]
    fn key_match_with_incompatible_size_is_refused() {
        let mut item = load_item("civitai_wan22_model_1817671.json");
        for v in &mut item.model_versions {
            if v.id == 2057100 {
                for f in &mut v.files {
                    f.size_kb = Some(1_000.0);
                }
            }
        }
        let err = pair_for(&item, 2057171).unwrap_err();
        assert!(
            matches!(err, A14bPairingError::CounterpartSizeMismatch { .. }),
            "{err:?}"
        );
    }

    /// The reviewer scenario: the DaSiWa TrueVision model publishes three
    /// finetune pairs, and two of them (BoundBite v10 and SynthSeduction
    /// v9) differ in size by ~0.01% — well inside the ±10% band. When the
    /// intended counterpart version vanishes upstream (deleted/private),
    /// the size band alone would silently pair a high-noise expert of one
    /// finetune with a low-noise expert of an unrelated one. Semantic
    /// name compatibility must fail this closed instead.
    #[test]
    fn vanished_counterpart_never_pairs_with_an_unrelated_finetune() {
        let item = load_search_item("civitai_wan22_a14b_i2v_search.json", 2272580);

        // Intact model: BoundBite High v10 pairs with BoundBite Low v10.
        let pair = pair_for(&item, 2769496).expect("intact BoundBite pair resolves");
        assert_eq!(pair.low.version_id, 2769497);

        // Remove the intended counterpart. SynthSeduction Low v9 remains
        // within the size band but is a different finetune revision.
        let mut lone = item.clone();
        lone.model_versions.retain(|v| v.id != 2769497);
        let err = pair_for(&lone, 2769496).unwrap_err();
        assert_eq!(
            err,
            A14bPairingError::NoCounterpart {
                name: "BoundBite High v10".into(),
                missing: "low-noise",
            },
            "a same-size stranger must never be adopted as the counterpart"
        );
    }

    /// The semantic gate must not break the real uploader typo the size
    /// band exists for: differing numeric tokens fail, tiny letter-level
    /// typos pass.
    #[test]
    fn semantic_key_compatibility_separates_typos_from_strangers() {
        // The official repack's real I2V typo: one character apart, same
        // numeric tokens.
        assert!(keys_semantically_compatible(
            &expert_key("i2v_high_noise_14B_fp8_sd"),
            &expert_key("i2v_low_noise_14B_fp8_scd"),
        ));
        // Different finetune revisions: numeric tokens differ (v10 vs v9)
        // and the text is far apart.
        assert!(!keys_semantically_compatible(
            &expert_key("BoundBite High v10"),
            &expert_key("SynthSeduction Low v9"),
        ));
        // Same finetune, different version: the letters are one edit
        // apart, but v9 vs v8 is a different artifact.
        assert!(!keys_semantically_compatible(
            &expert_key("SynthSeduction High v9"),
            &expert_key("SynthSeduction Low v8"),
        ));
    }

    /// A 4-bit (NF4) safetensors sibling is not a pairing candidate: the
    /// Wan loader has no NF4/NVFP4 load path, so adopting one as the
    /// counterpart would download an expert that fails at load.
    #[test]
    fn unloadable_precision_siblings_are_not_candidates() {
        let mut item = load_item("civitai_wan22_model_1817671.json");
        for v in &mut item.model_versions {
            if v.id == 2057100 {
                for f in &mut v.files {
                    f.metadata.fp = Some("nf4".into());
                }
            }
        }
        let err = pair_for(&item, 2057171).unwrap_err();
        assert!(
            matches!(err, A14bPairingError::NoCounterpart { .. }),
            "{err:?}"
        );
    }

    /// GGUF sibling versions are never candidates: the Civitai catalog
    /// path is safetensors-only, so a GGUF "counterpart" would install a
    /// file the recipe layer refuses.
    #[test]
    fn gguf_siblings_are_not_candidates() {
        let mut item = load_item("civitai_wan22_model_1817671.json");
        for v in &mut item.model_versions {
            if v.id == 2057100 {
                for f in &mut v.files {
                    f.name = "wanVideo22_t2vLowNoise14B.gguf".into();
                    f.metadata.format = Some("GGUF".into());
                }
            }
        }
        let err = pair_for(&item, 2057171).unwrap_err();
        assert!(
            matches!(err, A14bPairingError::NoCounterpart { .. }),
            "{err:?}"
        );
    }
}
