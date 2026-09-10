//! The progress stage names Hunyuan3D Paint reports, as a contract.
//!
//! Shape and paint run inside ONE queue job — the engine builds geometry, drops
//! the shape checkpoint, and then textures — so the durable mesh workflow only
//! ever learns that the job as a whole started and finished. The one live
//! signal that separates the two halves is the job's progress stage name, which
//! `paint_runtime` emits and the workflow runner reads. Both sides take the
//! names from HERE so the runner never matches a string the runtime stopped
//! emitting: a renamed stage would silently put the workflow back to showing
//! "Build geometry" and "Paint PBR materials" running at the same time, which
//! is what this module exists to prevent.
//!
//! Always compiled, unlike `paint_runtime`, because the server reads it whether
//! or not it was built with `mesh-texture`.

pub const UNWRAPPING_MESH: &str = "Unwrapping mesh";
pub const PREPARING_PAINT_MESH: &str = "Preparing paint mesh";
pub const PREPARING_PAINT_VIEWS: &str = "Preparing paint views";
pub const ENCODING_PAINT_APPEARANCE: &str = "Encoding paint appearance";
pub const ENCODING_PAINT_REFERENCE: &str = "Encoding paint reference";
pub const ENCODING_PAINT_NORMALS: &str = "Encoding paint normals";
pub const ENCODING_PAINT_POSITIONS: &str = "Encoding paint positions";
pub const GENERATING_PBR_VIEWS: &str = "Generating PBR views";
pub const DECODING_PBR_VIEWS: &str = "Decoding PBR views";
pub const UPSCALING_PBR_VIEWS: &str = "Upscaling PBR views";
pub const BAKING_PBR_TEXTURES: &str = "Baking PBR textures";
pub const FILLING_PBR_TEXTURES: &str = "Filling PBR textures";
pub const WRITING_TEXTURED_GLB: &str = "Writing textured GLB";

/// Every stage the paint half of a textured render reports, in order.
pub const PAINT_STAGES: &[&str] = &[
    UNWRAPPING_MESH,
    PREPARING_PAINT_MESH,
    PREPARING_PAINT_VIEWS,
    ENCODING_PAINT_APPEARANCE,
    ENCODING_PAINT_REFERENCE,
    ENCODING_PAINT_NORMALS,
    ENCODING_PAINT_POSITIONS,
    GENERATING_PBR_VIEWS,
    DECODING_PBR_VIEWS,
    UPSCALING_PBR_VIEWS,
    BAKING_PBR_TEXTURES,
    FILLING_PBR_TEXTURES,
    WRITING_TEXTURED_GLB,
];

/// True when `stage` is one the paint half reports — meaning the geometry
/// half of the same job has already finished.
pub fn is_paint_stage(stage: &str) -> bool {
    PAINT_STAGES.contains(&stage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_paint_stage_means_geometry_is_done_and_a_shape_stage_does_not() {
        assert!(is_paint_stage(UNWRAPPING_MESH));
        assert!(is_paint_stage(GENERATING_PBR_VIEWS));
        assert!(is_paint_stage(WRITING_TEXTURED_GLB));
        for shape in [
            "Sampling",
            "Decoding volume",
            "Extracting surface",
            "Simplifying mesh",
            "Writing mesh",
            "Removing background",
            "Loading UNet (GPU)",
        ] {
            assert!(!is_paint_stage(shape), "{shape} is not a paint stage");
        }
    }

    #[test]
    fn the_paint_stage_list_carries_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for stage in PAINT_STAGES {
            assert!(seen.insert(*stage), "{stage} listed twice");
        }
    }
}
