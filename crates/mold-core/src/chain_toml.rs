//! TOML script serialisation for chained generation.
//!
//! The canonical file format is `mold.chain.v1`:
//!
//! ```toml
//! schema = "mold.chain.v1"
//!
//! [chain]
//! model = "ltx-2-19b-distilled:fp8"
//! width = 1216
//! ...
//!
//! [[stage]]
//! prompt = "..."
//! frames = 97
//! # Attach a starting image — either as a path relative to the script file,
//! # or as inline base64:
//! # source_image_path = "./hero.png"
//! # source_image_b64  = "iVBORw0KGgo..."
//! ```
//!
//! Round-trip invariant: `read(write(script)) == script` for every script
//! that `ChainRequest::normalise` accepts.

use std::path::Path;

use base64::Engine as _;

use crate::chain::ChainScript;
use crate::error::{MoldError, Result};

/// Deserialise a TOML string into a [`ChainScript`]. Rejects unknown
/// schema versions with a clear error pointing at the current mold
/// version's supported schema.
pub fn read_script(toml_str: &str) -> Result<ChainScript> {
    #[derive(serde::Deserialize)]
    struct SchemaPeek {
        #[serde(default)]
        schema: Option<String>,
    }
    let peek: SchemaPeek = toml::from_str(toml_str)
        .map_err(|e| MoldError::Validation(format!("chain TOML parse failed: {e}")))?;
    let schema = peek.schema.as_deref().unwrap_or("mold.chain.v1");
    if schema != "mold.chain.v1" {
        return Err(MoldError::Validation(format!(
            "chain TOML schema '{schema}' is not supported by this mold version \
             (supported: 'mold.chain.v1')"
        )));
    }
    // If the file omitted the schema key, inject it so `ChainScript`
    // (which has a required `schema: String` field) can deserialise.
    let augmented;
    let parse_target = if peek.schema.is_none() {
        augmented = format!("schema = \"mold.chain.v1\"\n{toml_str}");
        augmented.as_str()
    } else {
        toml_str
    };
    toml::from_str(parse_target)
        .map_err(|e| MoldError::Validation(format!("chain TOML parse failed: {e}")))
}

/// Like [`read_script`], but additionally accepts two TOML-only per-stage
/// fields for attaching starting images:
///
/// - `source_image_path = "<path>"` — path to an image file, resolved
///   relative to `script_dir`. The file is read and base64-encoded at load
///   time.
/// - `source_image_b64 = "<base64>"` — inline base64 bytes. Equivalent to
///   the canonical `source_image` field; provided so the web composer can
///   round-trip through TOML without hand-editing, and so authors can
///   distinguish "inline bytes" from "file path" at a glance.
///
/// If both `source_image_path` and `source_image_b64` (or `source_image`)
/// are set on the same stage, returns a validation error rather than
/// silently picking one.
///
/// The canonical `source_image` field (base64 bytes, inherited from the
/// wire format) is still accepted for backwards compatibility.
pub fn read_script_resolving_paths(toml_str: &str, script_dir: &Path) -> Result<ChainScript> {
    let mut doc: toml::Value = toml::from_str(toml_str)
        .map_err(|e| MoldError::Validation(format!("chain TOML parse failed: {e}")))?;

    if let Some(stages) = doc
        .as_table_mut()
        .and_then(|t| t.get_mut("stage"))
        .and_then(|v| v.as_array_mut())
    {
        for (idx, stage) in stages.iter_mut().enumerate() {
            resolve_stage_source_image(stage, idx, script_dir)?;
        }
    }

    let rewritten = toml::to_string(&doc).map_err(|e| {
        MoldError::Other(anyhow::anyhow!(
            "chain TOML re-serialise after path resolution failed: {e}"
        ))
    })?;

    read_script(&rewritten)
}

/// In-place: move `source_image_path` / `source_image_b64` into the
/// canonical `source_image` base64 field. See [`read_script_resolving_paths`]
/// for the rules.
fn resolve_stage_source_image(
    stage: &mut toml::Value,
    idx: usize,
    script_dir: &Path,
) -> Result<()> {
    let Some(table) = stage.as_table_mut() else {
        return Ok(());
    };

    let path = table.remove("source_image_path");
    let b64 = table.remove("source_image_b64");
    let canonical_present = table.get("source_image").is_some();

    let mut specified = 0;
    if path.is_some() {
        specified += 1;
    }
    if b64.is_some() {
        specified += 1;
    }
    if canonical_present {
        specified += 1;
    }
    if specified > 1 {
        return Err(MoldError::Validation(format!(
            "stage {idx}: set at most one of source_image, source_image_path, source_image_b64"
        )));
    }

    if let Some(path_value) = path {
        let rel = path_value.as_str().ok_or_else(|| {
            MoldError::Validation(format!("stage {idx}: source_image_path must be a string"))
        })?;
        let abs = if Path::new(rel).is_absolute() {
            std::path::PathBuf::from(rel)
        } else {
            script_dir.join(rel)
        };
        let bytes = std::fs::read(&abs).map_err(|e| {
            MoldError::Validation(format!(
                "stage {idx}: failed to read source_image_path '{}': {e}",
                abs.display(),
            ))
        })?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        table.insert("source_image".into(), toml::Value::String(encoded));
    } else if let Some(b64_value) = b64 {
        let s = b64_value.as_str().ok_or_else(|| {
            MoldError::Validation(format!("stage {idx}: source_image_b64 must be a string"))
        })?;
        // Validate the base64 payload here so callers get a stage-indexed
        // error instead of a generic `base64_opt` decode failure later.
        base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(|e| {
                MoldError::Validation(format!(
                    "stage {idx}: source_image_b64 is not valid base64: {e}"
                ))
            })?;
        table.insert("source_image".into(), toml::Value::String(s.to_string()));
    }

    Ok(())
}

/// Serialise a [`ChainScript`] to a TOML string.
pub fn write_script(script: &ChainScript) -> Result<String> {
    let body = toml::to_string_pretty(script)
        .map_err(|e| MoldError::Other(anyhow::anyhow!("chain TOML serialise failed: {e}")))?;
    // toml-rs sorts table keys alphabetically; force schema header up top.
    if body.starts_with("schema") {
        Ok(body)
    } else {
        Ok(format!(
            "schema = \"{}\"\n\n{}",
            script.schema,
            body.replace("schema = \"mold.chain.v1\"\n", "")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{ChainScript, ChainScriptChain, ChainStage, TransitionMode};
    use crate::types::OutputFormat;

    fn sample_script() -> ChainScript {
        ChainScript {
            schema: "mold.chain.v1".into(),
            chain: ChainScriptChain {
                model: "ltx-2-19b-distilled:fp8".into(),
                width: 1216,
                height: 704,
                fps: 24,
                seed: Some(42),
                steps: 8,
                guidance: 3.0,
                strength: 1.0,
                motion_tail_frames: 25,
                output_format: OutputFormat::Mp4,
            },
            stages: vec![ChainStage {
                prompt: "a cat walks into the autumn forest".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            }],
        }
    }

    #[test]
    fn write_emits_schema_header_first() {
        let toml_out = write_script(&sample_script()).unwrap();
        assert!(
            toml_out.starts_with("schema = \"mold.chain.v1\""),
            "got:\n{toml_out}"
        );
    }

    #[test]
    fn write_uses_array_of_tables_for_stages() {
        let toml_out = write_script(&sample_script()).unwrap();
        assert!(toml_out.contains("[[stage]]"), "got:\n{toml_out}");
    }

    #[test]
    fn write_omits_empty_reserved_fields() {
        let toml_out = write_script(&sample_script()).unwrap();
        assert!(!toml_out.contains("loras"), "got:\n{toml_out}");
        assert!(!toml_out.contains("references"), "got:\n{toml_out}");
        assert!(!toml_out.contains("model =\n"), "got:\n{toml_out}");
    }

    #[test]
    fn read_accepts_missing_schema_header() {
        let toml_src = r#"
            [chain]
            model = "ltx-2-19b-distilled:fp8"
            width = 1216
            height = 704
            fps = 24
            steps = 8
            guidance = 3.0
            strength = 1.0
            motion_tail_frames = 25
            output_format = "mp4"

            [[stage]]
            prompt = "hello"
            frames = 97
        "#;
        let script = read_script(toml_src).unwrap();
        assert_eq!(script.schema, "mold.chain.v1");
        assert_eq!(script.stages.len(), 1);
    }

    #[test]
    fn read_rejects_unknown_schema_version() {
        let toml_src = r#"
            schema = "mold.chain.v99"
            [chain]
            model = "x"
            width = 1
            height = 1
            fps = 1
            steps = 1
            guidance = 1.0
            strength = 1.0
            motion_tail_frames = 0
            output_format = "mp4"
        "#;
        let err = read_script(toml_src).unwrap_err().to_string();
        assert!(err.contains("mold.chain.v99"), "got: {err}");
        assert!(err.contains("not supported"), "got: {err}");
    }

    #[test]
    fn read_resolves_source_image_path_relative_to_script_dir() {
        let dir = tempfile::tempdir().unwrap();
        let img_bytes: [u8; 6] = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a];
        std::fs::write(dir.path().join("hero.png"), img_bytes).unwrap();

        let toml_src = r#"
            [chain]
            model = "ltx-2-19b-distilled:fp8"
            width = 1216
            height = 704
            fps = 24
            steps = 8
            guidance = 3.0
            strength = 1.0
            motion_tail_frames = 25
            output_format = "mp4"

            [[stage]]
            prompt = "opening shot"
            frames = 97
            source_image_path = "hero.png"
        "#;

        let script = read_script_resolving_paths(toml_src, dir.path()).unwrap();
        assert_eq!(script.stages.len(), 1);
        assert_eq!(
            script.stages[0].source_image.as_deref(),
            Some(&img_bytes[..]),
        );
    }

    #[test]
    fn read_resolves_per_stage_source_image_path() {
        let dir = tempfile::tempdir().unwrap();
        let bytes_a: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];
        let bytes_b: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
        std::fs::write(dir.path().join("a.png"), bytes_a).unwrap();
        std::fs::write(dir.path().join("b.png"), bytes_b).unwrap();

        let toml_src = r#"
            [chain]
            model = "ltx-2-19b-distilled:fp8"
            width = 1216
            height = 704
            fps = 24
            steps = 8
            guidance = 3.0
            strength = 1.0
            motion_tail_frames = 25
            output_format = "mp4"

            [[stage]]
            prompt = "open"
            frames = 97
            source_image_path = "a.png"

            [[stage]]
            prompt = "close"
            frames = 97
            transition = "cut"
            source_image_path = "b.png"
        "#;

        let script = read_script_resolving_paths(toml_src, dir.path()).unwrap();
        assert_eq!(script.stages.len(), 2);
        assert_eq!(script.stages[0].source_image.as_deref(), Some(&bytes_a[..]));
        assert_eq!(script.stages[1].source_image.as_deref(), Some(&bytes_b[..]));
    }

    #[test]
    fn read_accepts_source_image_b64() {
        let bytes: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        let toml_src = format!(
            r#"
            [chain]
            model = "ltx-2-19b-distilled:fp8"
            width = 1216
            height = 704
            fps = 24
            steps = 8
            guidance = 3.0
            strength = 1.0
            motion_tail_frames = 25
            output_format = "mp4"

            [[stage]]
            prompt = "x"
            frames = 97
            source_image_b64 = "{encoded}"
        "#
        );

        let script = read_script_resolving_paths(&toml_src, Path::new(".")).unwrap();
        assert_eq!(script.stages[0].source_image.as_deref(), Some(&bytes[..]));
    }

    #[test]
    fn read_rejects_conflicting_source_image_fields() {
        let toml_src = r#"
            [chain]
            model = "ltx-2-19b-distilled:fp8"
            width = 1216
            height = 704
            fps = 24
            steps = 8
            guidance = 3.0
            strength = 1.0
            motion_tail_frames = 25
            output_format = "mp4"

            [[stage]]
            prompt = "x"
            frames = 97
            source_image_path = "a.png"
            source_image_b64 = "AAAA"
        "#;
        let err = read_script_resolving_paths(toml_src, Path::new("."))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("at most one of"),
            "error must name the conflict, got: {err}",
        );
    }

    #[test]
    fn read_reports_missing_source_image_file() {
        let dir = tempfile::tempdir().unwrap();
        let toml_src = r#"
            [chain]
            model = "ltx-2-19b-distilled:fp8"
            width = 1216
            height = 704
            fps = 24
            steps = 8
            guidance = 3.0
            strength = 1.0
            motion_tail_frames = 25
            output_format = "mp4"

            [[stage]]
            prompt = "x"
            frames = 97
            source_image_path = "missing.png"
        "#;
        let err = read_script_resolving_paths(toml_src, dir.path())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("missing.png"),
            "error must name the missing file, got: {err}",
        );
    }
}
