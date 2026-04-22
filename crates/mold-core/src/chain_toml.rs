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
//! ```
//!
//! Round-trip invariant: `read(write(script)) == script` for every script
//! that `ChainRequest::normalise` accepts.

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
}
