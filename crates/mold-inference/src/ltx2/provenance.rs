//! The LTX-2 runtime's provenance vocabulary: every value a render stamps
//! into `VideoData` / `OutputMetadata` and every INFO line the qualification
//! harness is allowed to grep for lives here as a literal.
//!
//! One module, on purpose. The CUDA qualification harness (#1398) reads the
//! metadata fields first and the log lines second, and both must be spelled
//! from these constants — a harness that greps for prose it composed itself
//! is a second vocabulary that drifts. A fixture string that is not one of
//! these literals is a test failure, not a new spelling.
//!
//! What is recorded and why:
//!
//! * **Attention path** — which arithmetic the video/audio self-attention
//!   took (#735). Output-changing: the BF16 dispatcher and the F32 chunked
//!   path agree to BF16 precision, not bit-for-bit, so a print must say
//!   which one rendered it.
//! * **Residency mode** — how the transformer's blocks were placed on the
//!   device. Not output-changing, but the single largest wall-clock and
//!   peak-VRAM lever, so a measurement without it is unreproducible.
//! * **Audio branch** — whether the audio-video transformer ran its audio
//!   half (#1037). Output-changing for the video: the branch feeds the video
//!   stream through the a2v cross-attention, so a silent export rendered
//!   with it skipped is a different print from one rendered with it on.
//!
//! The INT8 ConvRot arm (`ltx2 int8 arm=native|dequant`) is defined beside the
//! arm that selects it, in `ltx2/convrot.rs`, and follows the same rule: a
//! literal, emitted once at INFO, never composed at the call site.

/// Metadata value: unmasked self-attention ran through `crate::attention`'s
/// math backend in BF16 (the CUDA default since #735).
pub const ATTENTION_PATH_BF16_MATH: &str = "ltx2-bf16-math";
/// Metadata value: unmasked self-attention ran through `candle-flash-attn`
/// (`MOLD_ATTN=flash` on a build that compiles it).
pub const ATTENTION_PATH_BF16_FLASH: &str = "ltx2-bf16-flash";
/// Metadata value: every self-attention took the hand-rolled F32 chunked
/// path — CPU, or `MOLD_LTX2_ATTN_F32=1` on CUDA.
pub const ATTENTION_PATH_F32_CHUNKED: &str = "ltx2-f32-chunked";
/// Metadata value: Candle's fused Metal SDPA.
pub const ATTENTION_PATH_METAL_SDPA: &str = "ltx2-metal-sdpa";

/// Every attention-path value a render may stamp, for harness parity tests.
pub const ATTENTION_PATHS: &[&str] = &[
    ATTENTION_PATH_BF16_MATH,
    ATTENTION_PATH_BF16_FLASH,
    ATTENTION_PATH_F32_CHUNKED,
    ATTENTION_PATH_METAL_SDPA,
];

/// Residency vocabulary for the INFO line.
pub const RESIDENCY_MODE_STREAMING: &str = "streaming";
pub const RESIDENCY_MODE_ADAPTIVE: &str = "adaptive";
pub const RESIDENCY_MODE_EAGER: &str = "eager";

/// Audio-branch vocabulary for the INFO line.
pub const AUDIO_BRANCH_RUN: &str = "run";
pub const AUDIO_BRANCH_SKIPPED: &str = "skipped";

/// `tracing` target every provenance line is emitted under.
pub const LOG_TARGET: &str = "mold::ltx2";

/// `ltx2 attention path=<value>` — emitted once per render, right after the
/// real render path returns.
pub fn attention_path_line(path: &str) -> String {
    format!("ltx2 attention path={path}")
}

/// `ltx2 residency mode=<mode> resident=<n> streamed=<m>` — emitted once per
/// transformer load, from whichever arm chose the mode. Eager and streaming
/// report their block split too so the line is one shape for every mode.
pub fn residency_mode_line(mode: &str, resident: usize, streamed: usize) -> String {
    format!("ltx2 residency mode={mode} resident={resident} streamed={streamed}")
}

/// `ltx2 audio branch=run|skipped` — emitted once per prepared run, where the
/// execution graph's `run_audio_branch` is read.
pub fn audio_branch_line(run: bool) -> String {
    let state = if run {
        AUDIO_BRANCH_RUN
    } else {
        AUDIO_BRANCH_SKIPPED
    };
    format!("ltx2 audio branch={state}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The harness greps these exact shapes; a reworded line is a silent
    /// harness failure, so the shapes are pinned here.
    #[test]
    fn provenance_lines_are_pinned_literals() {
        assert_eq!(
            attention_path_line(ATTENTION_PATH_BF16_MATH),
            "ltx2 attention path=ltx2-bf16-math"
        );
        assert_eq!(
            residency_mode_line(RESIDENCY_MODE_ADAPTIVE, 28, 20),
            "ltx2 residency mode=adaptive resident=28 streamed=20"
        );
        assert_eq!(audio_branch_line(true), "ltx2 audio branch=run");
        assert_eq!(audio_branch_line(false), "ltx2 audio branch=skipped");
    }

    /// Every stampable attention path is distinct and carries the family
    /// prefix, so a value can never be confused with another family's.
    #[test]
    fn attention_paths_are_distinct_family_scoped_values() {
        let mut seen = std::collections::BTreeSet::new();
        for path in ATTENTION_PATHS {
            assert!(path.starts_with("ltx2-"), "{path}");
            assert!(seen.insert(*path), "duplicate attention path {path}");
        }
        assert_eq!(ATTENTION_PATHS.len(), 4);
    }
}
