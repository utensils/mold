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

use crate::chain::{ChainScript, ChainScriptChain, ChainStage};
use crate::error::{MoldError, Result};

// TOML integers are signed i64; JSON generation requests keep their existing
// u64 contract. These private wrappers change only the two seed fields.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum TomlSeed {
    Integer(u64),
    Decimal(String),
}

impl From<u64> for TomlSeed {
    fn from(value: u64) -> Self {
        if value <= i64::MAX as u64 {
            Self::Integer(value)
        } else {
            Self::Decimal(value.to_string())
        }
    }
}

impl TomlSeed {
    fn into_u64(self, field: &str) -> Result<u64> {
        match self {
            Self::Integer(value) => Ok(value),
            Self::Decimal(value)
                if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) =>
            {
                value.parse().map_err(|_| {
                    MoldError::Validation(format!(
                        "{field} must be a decimal u64 in range 0..={}",
                        u64::MAX
                    ))
                })
            }
            Self::Decimal(_) => Err(MoldError::Validation(format!(
                "{field} must be an unsigned integer or decimal string"
            ))),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TomlChain {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<TomlSeed>,
    #[serde(flatten)]
    fields: ChainScriptChain,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TomlStage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed_offset: Option<TomlSeed>,
    #[serde(flatten)]
    fields: ChainStage,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TomlScript {
    schema: String,
    chain: TomlChain,
    #[serde(rename = "stage")]
    stages: Vec<TomlStage>,
}

impl From<&ChainScript> for TomlScript {
    fn from(script: &ChainScript) -> Self {
        let mut fields = script.chain.clone();
        let seed = fields.seed.take().map(TomlSeed::from);
        Self {
            schema: script.schema.clone(),
            chain: TomlChain { seed, fields },
            stages: script
                .stages
                .iter()
                .map(|stage| {
                    let mut fields = stage.clone();
                    let seed_offset = fields.seed_offset.take().map(TomlSeed::from);
                    TomlStage {
                        seed_offset,
                        fields,
                    }
                })
                .collect(),
        }
    }
}

impl TomlScript {
    fn into_script(self) -> Result<ChainScript> {
        let mut chain = self.chain.fields;
        chain.seed = self
            .chain
            .seed
            .map(|seed| seed.into_u64("chain.seed"))
            .transpose()?;
        let stages = self
            .stages
            .into_iter()
            .enumerate()
            .map(|(index, stage)| {
                let mut fields = stage.fields;
                fields.seed_offset = stage
                    .seed_offset
                    .map(|seed| seed.into_u64(&format!("stage {index} seed_offset")))
                    .transpose()?;
                Ok(fields)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ChainScript {
            schema: self.schema,
            chain,
            stages,
        })
    }
}

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
    toml::from_str::<TomlScript>(parse_target)
        .map_err(|e| MoldError::Validation(format!("chain TOML parse failed: {e}")))?
        .into_script()
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
            resolve_stage_boundary_frames(stage, idx, script_dir)?;
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

/// Resolve FL2VA boundary sugar without accepting ambiguous endpoint authority.
fn resolve_stage_boundary_frames(stage: &mut toml::Value, idx: usize, dir: &Path) -> Result<()> {
    let Some(table) = stage.as_table_mut() else {
        return Ok(());
    };
    let first = table.remove("first_frame_path");
    let last = table.remove("last_frame_path");
    if first.is_none() && last.is_none() {
        return Ok(());
    }
    if [
        "keyframes",
        "source_image",
        "source_image_path",
        "source_image_b64",
    ]
    .iter()
    .any(|key| table.contains_key(*key))
    {
        return Err(MoldError::Validation(format!("stage {idx}: cannot combine first_frame_path/last_frame_path with keyframes or source_image")));
    }
    let frames = table
        .get("frames")
        .and_then(toml::Value::as_integer)
        .filter(|n| *n > 0)
        .ok_or_else(|| {
            MoldError::Validation(format!(
                "stage {idx}: boundary paths require positive frames"
            ))
        })?;
    let mut keyframes = Vec::new();
    for (role, frame, path) in [("first", 0, first), ("last", frames - 1, last)] {
        let Some(path) = path else {
            continue;
        };
        let rel = path.as_str().ok_or_else(|| {
            MoldError::Validation(format!("stage {idx}: {role}_frame_path must be a string"))
        })?;
        let path = dir.join(rel);
        let bytes = std::fs::read(&path).map_err(|error| {
            MoldError::Validation(format!(
                "stage {idx}: failed to read {role}_frame_path '{}': {error}",
                path.display()
            ))
        })?;
        let mut endpoint = toml::map::Map::new();
        endpoint.insert("frame".into(), toml::Value::Integer(frame));
        endpoint.insert(
            "image".into(),
            toml::Value::String(base64::engine::general_purpose::STANDARD.encode(bytes)),
        );
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            endpoint.insert("name".into(), toml::Value::String(name.into()));
        }
        keyframes.push(toml::Value::Table(endpoint));
    }
    table.insert("keyframes".into(), toml::Value::Array(keyframes));
    Ok(())
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
    let body = toml::to_string_pretty(&TomlScript::from(script))
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

    #[test]
    fn canonical_toml_round_trips_linked_part_authority() {
        let mut script = sample_script();
        script.chain.continue_from = Some(crate::chain::ChainContinuation {
            job_id: "previous-part".into(),
            finalize_take: 3,
        });
        script.chain.seed = Some(u64::MAX);
        let text = write_script(&script).unwrap();
        let restored = read_script(&text).unwrap();
        assert_eq!(restored.chain.continue_from, script.chain.continue_from);
        assert_eq!(restored.chain.seed, Some(u64::MAX));
        let with_paths = read_script_resolving_paths(&text, Path::new(".")).unwrap();
        assert_eq!(with_paths.chain.continue_from, script.chain.continue_from);
    }

    #[test]
    fn canonical_toml_round_trips_full_range_seeds_and_stage_offsets() {
        for seed in [
            0,
            i64::MAX as u64,
            i64::MAX as u64 + 1,
            17_309_142_824_083_333_797,
            u64::MAX,
        ] {
            let mut script = sample_script();
            script.chain.seed = Some(seed);
            script.stages[0].seed_offset = Some(seed);
            let text =
                write_script(&script).expect("every materialized u64 seed must export to TOML");
            let restored = read_script(&text).unwrap();
            assert_eq!(restored.chain.seed, Some(seed));
            assert_eq!(restored.stages[0].seed_offset, Some(seed));
            let with_paths = read_script_resolving_paths(&text, Path::new(".")).unwrap();
            assert_eq!(with_paths.chain.seed, Some(seed));
            assert_eq!(with_paths.stages[0].seed_offset, Some(seed));
            if seed > i64::MAX as u64 {
                assert!(text.contains(&format!("seed = \"{seed}\"")));
                assert!(text.contains(&format!("seed_offset = \"{seed}\"")));
            }
            // The JSON/API shape keeps the existing numeric seed contract.
            let json = serde_json::to_value(&restored).unwrap();
            assert_eq!(json["chain"]["seed"].as_u64(), Some(seed));
            assert_eq!(json["stage"][0]["seed_offset"].as_u64(), Some(seed));
        }
    }

    #[test]
    fn toml_seed_strings_accept_decimal_only_and_reject_overflow() {
        let text = write_script(&sample_script()).unwrap();
        let quoted = text.replace("seed = 42", "seed = \"42\"");
        assert_eq!(read_script(&quoted).unwrap().chain.seed, Some(42));
        for invalid in ["-1", "+1", " 42", "1.5", "18446744073709551616", ""] {
            let invalid = text.replace("seed = 42", &format!("seed = \"{invalid}\""));
            assert!(
                read_script(&invalid).is_err(),
                "accepted invalid seed: {invalid}"
            );
        }
    }

    #[test]
    fn resolves_h3_boundary_paths_and_preserves_typed_stage_media() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("first.png"), b"first").unwrap();
        std::fs::write(dir.path().join("last.png"), b"last").unwrap();
        let script = read_script_resolving_paths(
            r#"
            [chain]
            model = "minimax-h3-fl2va:comfy-pruned-int8"
            width = 256
            height = 256
            fps = 24
            steps = 21
            guidance = 0.0
            strength = 1.0
            motion_tail_frames = 1
            output_format = "mp4"
            [[stage]]
            prompt = "walk"
            frames = 107
            first_frame_path = "first.png"
            last_frame_path = "last.png"
        "#,
            dir.path(),
        )
        .unwrap();
        let wire = serde_json::to_value(&script.stages[0]).unwrap();
        assert_eq!(wire["keyframes"][0]["frame"], 0);
        assert_eq!(wire["keyframes"][1]["frame"], 106);
        assert_eq!(
            serde_json::to_value(read_script(&write_script(&script).unwrap()).unwrap()).unwrap(),
            serde_json::to_value(&script).unwrap()
        );
    }

    #[test]
    fn boundary_paths_cannot_silently_replace_existing_source() {
        let err = read_script_resolving_paths(
            r#"
            [chain]
            model = "minimax-h3-fl2va:comfy-pruned-int8"
            width = 256
            height = 256
            fps = 24
            steps = 21
            guidance = 0.0
            strength = 1.0
            motion_tail_frames = 1
            output_format = "mp4"
            [[stage]]
            prompt = "walk"
            frames = 107
            source_image = "YQ=="
            first_frame_path = "missing.png"
        "#,
            Path::new("."),
        )
        .unwrap_err();
        assert!(err.to_string().contains("cannot combine"), "{err}");
    }

    fn sample_script() -> ChainScript {
        ChainScript {
            schema: "mold.chain.v1".into(),
            chain: ChainScriptChain {
                source_fit: None,
                continue_from: None,
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
                enable_audio: None,
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
                generation_references: None,
                keyframes: None,
                authored_media_sha256: None,
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
