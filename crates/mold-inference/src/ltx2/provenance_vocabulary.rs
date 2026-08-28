//! Exact provenance vocabulary the LTX-2 CUDA qualification harness greps for.
//!
//! `scripts/fixtures/ltx25-cuda-matrix.json` names, per row, the INFO log
//! lines the retained server log must contain and the additive saved-metadata
//! fields the Library row must carry. Those strings are pinned HERE so the
//! fixture and the emitters cannot drift apart silently: a matrix row may
//! only expect a member of [`KNOWN_PROVENANCE_LINES`] / [`KNOWN_METADATA_FIELDS`],
//! and the emitters must log these exact prefixes.
//!
//! The emitting side is `ltx2::provenance` (attention / residency / audio),
//! `ltx2::convrot` (the INT8 arm), and `ltx2::gguf` (the linear kind); the
//! `vocabulary_matches_the_emitting_constants` test pins every pinned string
//! here to the constant that actually emits it, so the harness's read-side
//! pin and the runtime cannot drift apart.
//!
//! Vocabulary (one line per decision, values `|`-separated):
//!
//! - `ltx2 attention path=ltx2-bf16-math|ltx2-bf16-flash|ltx2-f32-chunked|ltx2-metal-sdpa`
//! - `ltx2 residency mode=streaming|adaptive|eager resident=N streamed=M`
//!   (the `resident=`/`streamed=` counts vary per checkpoint and canvas, so
//!   the pinned strings stop at the mode)
//! - `ltx2 int8 arm=native-w8a8|dequant-cuda|dequant-metal|dequant-host`
//! - `ltx2 linear kind=qmatmul|dequant` (GGUF tiers)
//! - `ltx2 audio branch=run|skipped`
//! - `attention backend selected backend=Math|Flash` — the shared dispatcher's
//!   existing line (`crate::attention::AttentionBackend::resolve`), emitted
//!   once per process.

#![allow(dead_code)]

/// `ltx2 attention path=...` — which self-attention arm the video transformer ran.
pub(crate) const ATTENTION_PATH_BF16_MATH: &str = "ltx2 attention path=ltx2-bf16-math";
pub(crate) const ATTENTION_PATH_BF16_FLASH: &str = "ltx2 attention path=ltx2-bf16-flash";
pub(crate) const ATTENTION_PATH_F32_CHUNKED: &str = "ltx2 attention path=ltx2-f32-chunked";
pub(crate) const ATTENTION_PATH_METAL_SDPA: &str = "ltx2 attention path=ltx2-metal-sdpa";

/// `ltx2 residency mode=...` — the transformer residency the planner selected.
pub(crate) const RESIDENCY_MODE_STREAMING: &str = "ltx2 residency mode=streaming";
pub(crate) const RESIDENCY_MODE_ADAPTIVE: &str = "ltx2 residency mode=adaptive";
pub(crate) const RESIDENCY_MODE_EAGER: &str = "ltx2 residency mode=eager";

/// `ltx2 int8 arm=...` — how INT8 ConvRot linears were executed.
pub(crate) const INT8_ARM_NATIVE_W8A8: &str = "ltx2 int8 arm=native-w8a8";
pub(crate) const INT8_ARM_DEQUANT_CUDA: &str = "ltx2 int8 arm=dequant-cuda";
pub(crate) const INT8_ARM_DEQUANT_METAL: &str = "ltx2 int8 arm=dequant-metal";
pub(crate) const INT8_ARM_DEQUANT_HOST: &str = "ltx2 int8 arm=dequant-host";

/// `ltx2 linear kind=...` — the GGUF linear arm.
pub(crate) const LINEAR_KIND_QMATMUL: &str = "ltx2 linear kind=qmatmul";
pub(crate) const LINEAR_KIND_DEQUANT: &str = "ltx2 linear kind=dequant";

/// `ltx2 audio branch=...` — whether the audio branch ran (#1037 opt-in skip).
pub(crate) const AUDIO_BRANCH_RUN: &str = "ltx2 audio branch=run";
pub(crate) const AUDIO_BRANCH_SKIPPED: &str = "ltx2 audio branch=skipped";

/// The shared attention dispatcher's own selection line.
pub(crate) const DISPATCHER_BACKEND_MATH: &str = "attention backend selected backend=Math";
pub(crate) const DISPATCHER_BACKEND_FLASH: &str = "attention backend selected backend=Flash";

/// Every log line a matrix row may expect, verbatim.
pub(crate) const KNOWN_PROVENANCE_LINES: &[&str] = &[
    ATTENTION_PATH_BF16_MATH,
    ATTENTION_PATH_BF16_FLASH,
    ATTENTION_PATH_F32_CHUNKED,
    ATTENTION_PATH_METAL_SDPA,
    RESIDENCY_MODE_STREAMING,
    RESIDENCY_MODE_ADAPTIVE,
    RESIDENCY_MODE_EAGER,
    INT8_ARM_NATIVE_W8A8,
    INT8_ARM_DEQUANT_CUDA,
    INT8_ARM_DEQUANT_METAL,
    INT8_ARM_DEQUANT_HOST,
    LINEAR_KIND_QMATMUL,
    LINEAR_KIND_DEQUANT,
    AUDIO_BRANCH_RUN,
    AUDIO_BRANCH_SKIPPED,
    DISPATCHER_BACKEND_MATH,
    DISPATCHER_BACKEND_FLASH,
];

/// Additive `OutputMetadata` fields (`generations.metadata_json`) a matrix row
/// may assert, with the values each admits.
pub(crate) const METADATA_ATTENTION_PATH: &str = "attention_path";
pub(crate) const METADATA_VIDEO_ONLY: &str = "video_only";
pub(crate) const METADATA_INT8_ARM: &str = "int8_arm";
pub(crate) const KNOWN_METADATA_FIELDS: &[(&str, &[&str])] = &[
    (
        METADATA_ATTENTION_PATH,
        &[
            "ltx2-bf16-math",
            "ltx2-bf16-flash",
            "ltx2-f32-chunked",
            "ltx2-metal-sdpa",
        ],
    ),
    (METADATA_VIDEO_ONLY, &["true", "false"]),
    (
        METADATA_INT8_ARM,
        &[
            "native-w8a8",
            "dequant-cuda",
            "dequant-metal",
            "dequant-host",
        ],
    ),
];

/// Environment profiles the harness restarts the scratch server under. Every
/// variable here except the shared `MOLD_ATTN` is an LTX-2 engine-shaping
/// knob and must be listed in `crate::runtime_env::ENGINE_SHAPING_VARIABLES`
/// once its emitter lands, or it silently reads as unset.
pub(crate) const KNOWN_PROFILE_VARIABLES: &[&str] = &[
    "MOLD_ATTN",
    "MOLD_LTX2_ATTN_F32",
    "MOLD_LTX2_INT8",
    "MOLD_LTX2_QMATMUL",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pinned read-side string is byte-equal to (or a prefix of, for
    /// the residency line whose counts vary) what the emitters compose from
    /// their own constants.
    #[test]
    fn vocabulary_matches_the_emitting_constants() {
        use crate::ltx2::provenance as emit;
        assert_eq!(
            ATTENTION_PATH_BF16_MATH,
            emit::attention_path_line(emit::ATTENTION_PATH_BF16_MATH)
        );
        assert_eq!(
            ATTENTION_PATH_BF16_FLASH,
            emit::attention_path_line(emit::ATTENTION_PATH_BF16_FLASH)
        );
        assert_eq!(
            ATTENTION_PATH_F32_CHUNKED,
            emit::attention_path_line(emit::ATTENTION_PATH_F32_CHUNKED)
        );
        assert_eq!(
            ATTENTION_PATH_METAL_SDPA,
            emit::attention_path_line(emit::ATTENTION_PATH_METAL_SDPA)
        );
        for (pinned, mode) in [
            (RESIDENCY_MODE_STREAMING, emit::RESIDENCY_MODE_STREAMING),
            (RESIDENCY_MODE_ADAPTIVE, emit::RESIDENCY_MODE_ADAPTIVE),
            (RESIDENCY_MODE_EAGER, emit::RESIDENCY_MODE_EAGER),
        ] {
            assert!(emit::residency_mode_line(mode, 0, 0).starts_with(pinned));
        }
        for (pinned, arm) in [
            (
                INT8_ARM_NATIVE_W8A8,
                crate::ltx2::convrot::INT8_ARM_NATIVE_W8A8,
            ),
            (
                INT8_ARM_DEQUANT_CUDA,
                crate::ltx2::convrot::INT8_ARM_DEQUANT_CUDA,
            ),
            (
                INT8_ARM_DEQUANT_METAL,
                crate::ltx2::convrot::INT8_ARM_DEQUANT_METAL,
            ),
            (
                INT8_ARM_DEQUANT_HOST,
                crate::ltx2::convrot::INT8_ARM_DEQUANT_HOST,
            ),
        ] {
            assert_eq!(pinned, format!("ltx2 int8 arm={arm}"));
        }
        assert_eq!(LINEAR_KIND_QMATMUL, crate::ltx2::gguf::LINEAR_KIND_QMATMUL);
        assert_eq!(LINEAR_KIND_DEQUANT, crate::ltx2::gguf::LINEAR_KIND_DEQUANT);
        assert_eq!(AUDIO_BRANCH_RUN, emit::audio_branch_line(true));
        assert_eq!(AUDIO_BRANCH_SKIPPED, emit::audio_branch_line(false));
        // Metadata values are the bare emitter literals.
        let (_, attention_values) = KNOWN_METADATA_FIELDS
            .iter()
            .find(|(name, _)| *name == METADATA_ATTENTION_PATH)
            .unwrap();
        assert_eq!(attention_values, &emit::ATTENTION_PATHS);
    }

    const MATRIX: &str = include_str!("../../../../scripts/fixtures/ltx25-cuda-matrix.json");

    fn matrix() -> serde_json::Value {
        serde_json::from_str(MATRIX).expect("scripts/fixtures/ltx25-cuda-matrix.json parses")
    }

    #[test]
    fn every_matrix_provenance_expectation_is_a_known_line() {
        let matrix = matrix();
        let rows = matrix["rows"].as_array().expect("rows array");
        assert!(rows.len() >= 50, "matrix lost rows: {}", rows.len());
        let mut used = std::collections::BTreeSet::new();
        for row in rows {
            let id = row["id"].as_str().expect("row id");
            let Some(expected) = row["expect"]["provenance"].as_array() else {
                assert_eq!(
                    row["kind"].as_str(),
                    Some("deferred"),
                    "{id}: only deferred rows omit expect.provenance"
                );
                continue;
            };
            for line in expected {
                let line = line.as_str().expect("provenance string");
                assert!(
                    KNOWN_PROVENANCE_LINES.contains(&line),
                    "{id} expects a provenance line the vocabulary does not pin: {line:?}"
                );
                used.insert(line);
            }
        }
        // Every CUDA-reachable line is exercised by at least one row; the
        // Metal SDPA path, the host/Metal dequant arms, and eager residency
        // are the only members a 24 GB CUDA campaign never observes.
        for line in KNOWN_PROVENANCE_LINES {
            let cuda_unreachable = matches!(
                *line,
                ATTENTION_PATH_METAL_SDPA
                    | INT8_ARM_DEQUANT_METAL
                    | INT8_ARM_DEQUANT_HOST
                    | RESIDENCY_MODE_EAGER
            );
            assert!(
                cuda_unreachable || used.contains(line),
                "{line:?} is pinned but no matrix row expects it"
            );
        }
    }

    #[test]
    fn every_matrix_metadata_expectation_is_a_known_field_and_value() {
        let matrix = matrix();
        for row in matrix["rows"].as_array().expect("rows array") {
            let id = row["id"].as_str().expect("row id");
            let Some(metadata) = row["expect"]["metadata"].as_object() else {
                continue;
            };
            for (field, value) in metadata {
                let (_, admitted) = KNOWN_METADATA_FIELDS
                    .iter()
                    .find(|(name, _)| name == field)
                    .unwrap_or_else(|| panic!("{id} expects unknown metadata field {field:?}"));
                let rendered = match value {
                    serde_json::Value::String(text) => text.clone(),
                    serde_json::Value::Bool(flag) => flag.to_string(),
                    other => panic!("{id}: metadata {field} has a non-scalar value {other}"),
                };
                assert!(
                    admitted.contains(&rendered.as_str()),
                    "{id}: metadata {field}={rendered:?} is not an admitted value"
                );
            }
        }
    }

    #[test]
    fn every_profile_variable_is_known_and_every_row_names_a_profile() {
        let matrix = matrix();
        let profiles = matrix["common"]["profiles"]
            .as_object()
            .expect("profiles object");
        for (name, env) in profiles {
            for variable in env.as_object().expect("profile env object").keys() {
                assert!(
                    KNOWN_PROFILE_VARIABLES.contains(&variable.as_str()),
                    "profile {name} sets an unknown variable {variable}"
                );
            }
        }
        for row in matrix["rows"].as_array().expect("rows array") {
            let profile = row["profile"].as_str().expect("row profile");
            assert!(
                profiles.contains_key(profile),
                "{} names an undefined profile {profile}",
                row["id"]
            );
        }
    }

    #[test]
    fn known_lines_are_unique_and_share_the_documented_shape() {
        let mut seen = std::collections::BTreeSet::new();
        for line in KNOWN_PROVENANCE_LINES {
            assert!(seen.insert(*line), "duplicate provenance line {line:?}");
            assert!(
                line.starts_with("ltx2 ") || line.starts_with("attention backend selected "),
                "{line:?} is not an ltx2 or dispatcher line"
            );
            assert!(line.contains('='), "{line:?} carries no key=value decision");
            assert!(
                !line.contains(['\n', '"']),
                "{line:?} must be a single grep-safe line"
            );
        }
    }
}
