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
}
