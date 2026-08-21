//! The identity fidelity gate (#1222's measurement, #1223's UAT tool).
//!
//! A PuLID render either preserves the reference person's face or it does not,
//! and "it looks right" is not a result anyone can re-check next quarter. So
//! the gate is an ArcFace cosine between the reference photograph and the face
//! in the rendered print, measured with the same `glintr100` graph the
//! conditioning itself used.
//!
//! Weight-gated and `#[ignore]`d, because it needs the antelopev2 models and a
//! real render:
//!
//! ```text
//! MOLD_TEST_PULID_ASSETS=/path/to/pulid-assets \
//! MOLD_TEST_IDENTITY_REFERENCE=/path/to/portrait.jpg \
//! MOLD_TEST_IDENTITY_RENDER=/path/to/render.png \
//!   cargo test -p mold-ai-inference --features pulid \
//!   --test pulid_identity_fidelity -- --ignored --nocapture
//! ```
//!
//! `MOLD_TEST_IDENTITY_RENDER` may name several comma-separated files; each is
//! reported on its own line so one invocation can score a whole UAT sweep.
//!
//! ## The tolerance
//!
//! ArcFace cosine on `glintr100` is conventionally read as: **0.28** is the
//! same-person threshold InsightFace itself ships for verification at a 1e-4
//! false-accept rate, and PuLID's own paper reports face similarity in the
//! 0.6-0.8 band against its reference photographs. Mold's gate is deliberately
//! the conservative end — [`SAME_PERSON_THRESHOLD`] — because the question this
//! test answers is "did the conditioning reach the print at all", not "how good
//! is it". A render that scores below it is a bug in the pipeline, not a
//! quality complaint. Record the number either way; a run that lands just above
//! the threshold is evidence about the pipeline, not a pass to be filed away.

#![cfg(feature = "pulid")]

use std::path::{Path, PathBuf};

use mold_inference::identity::IdentityExtractor;

/// InsightFace's own same-person decision threshold for this recognizer.
const SAME_PERSON_THRESHOLD: f32 = 0.28;

fn extractor() -> Option<IdentityExtractor> {
    let dir = PathBuf::from(std::env::var_os("MOLD_TEST_PULID_ASSETS")?);
    let direct = |name: &str| -> PathBuf {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return candidate;
        }
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path().join(name))
            .find(|path| path.is_file())
            .unwrap_or(candidate)
    };
    Some(
        IdentityExtractor::from_paths(
            &direct("scrfd_10g_bnkps.onnx"),
            &direct("glintr100.onnx"),
        )
        .expect("the antelopev2 models load"),
    )
}

fn embed(extractor: &IdentityExtractor, path: &Path) -> mold_inference::identity::arcface::ArcFaceEmbedding {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    extractor
        .extract(&bytes)
        .unwrap_or_else(|error| panic!("no face in {}: {error}", path.display()))
        .arcface
}

#[test]
#[ignore = "requires antelopev2 via MOLD_TEST_PULID_ASSETS plus a reference and a render"]
fn a_conditioned_render_preserves_the_reference_identity() {
    let Some(extractor) = extractor() else {
        panic!("set MOLD_TEST_PULID_ASSETS to the antelopev2 directory");
    };
    let reference = PathBuf::from(
        std::env::var_os("MOLD_TEST_IDENTITY_REFERENCE")
            .expect("set MOLD_TEST_IDENTITY_REFERENCE to the portrait that conditioned the render"),
    );
    let renders = std::env::var("MOLD_TEST_IDENTITY_RENDER")
        .expect("set MOLD_TEST_IDENTITY_RENDER to one or more comma-separated rendered prints");

    let reference_embedding = embed(&extractor, &reference);
    println!("reference: {}", reference.display());

    let mut failures = Vec::new();
    for render in renders.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let path = PathBuf::from(render);
        let cosine = reference_embedding.cosine_similarity(&embed(&extractor, &path));
        let verdict = if cosine >= SAME_PERSON_THRESHOLD {
            "PASS"
        } else {
            failures.push(format!("{render}: {cosine:.4}"));
            "FAIL"
        };
        println!(
            "{verdict}  cosine {cosine:.4}  (threshold {SAME_PERSON_THRESHOLD})  {}",
            path.display()
        );
    }

    assert!(
        failures.is_empty(),
        "renders below the same-person threshold: {}",
        failures.join("; ")
    );
}
