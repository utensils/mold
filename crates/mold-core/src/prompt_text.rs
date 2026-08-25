use serde::{Deserialize, Deserializer};
use std::borrow::Cow;

/// Canonicalize prompt line endings and decode newline escapes left literal
/// by shell arguments or over-encoded API clients.
///
/// A doubled backslash preserves a literal escape (`\\\\n` -> `\\n`), so a
/// prompt can still describe source text or paths containing those characters.
pub fn normalize_prompt_newlines(input: &str) -> Cow<'_, str> {
    if !input.contains(['\\', '\r']) {
        return Cow::Borrowed(input);
    }

    let mut chars = input.chars().peekable();
    let mut output = String::with_capacity(input.len());
    while let Some(ch) = chars.next() {
        match ch {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                output.push('\n');
            }
            '\\' => match chars.peek().copied() {
                Some('\\') => {
                    chars.next();
                    output.push('\\');
                }
                Some('n') => {
                    chars.next();
                    output.push('\n');
                }
                Some('r') => {
                    chars.next();
                    if chars.peek() == Some(&'\\') {
                        let mut lookahead = chars.clone();
                        lookahead.next();
                        if lookahead.peek() == Some(&'n') {
                            chars.next();
                            chars.next();
                        }
                    }
                    output.push('\n');
                }
                _ => output.push('\\'),
            },
            _ => output.push(ch),
        }
    }
    Cow::Owned(output)
}

pub(crate) fn deserialize_prompt<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    Ok(normalize_prompt_newlines(&value).into_owned())
}

pub(crate) fn deserialize_optional_prompt<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)
        .map(|value| value.map(|prompt| normalize_prompt_newlines(&prompt).into_owned()))
}

#[cfg(test)]
mod tests {
    use super::normalize_prompt_newlines;
    use crate::{
        ChainRequest, ExpandRequest, GenerateRequest, OutputMetadata, RemixRequest, Scheduler,
    };

    #[test]
    fn decodes_literal_and_platform_newlines() {
        assert_eq!(
            normalize_prompt_newlines(r"first\n\nsecond\r\nthird"),
            "first\n\nsecond\nthird"
        );
        assert_eq!(
            normalize_prompt_newlines("first\r\nsecond\rthird"),
            "first\nsecond\nthird"
        );
    }

    #[test]
    fn doubled_backslash_preserves_literal_escape_text() {
        assert_eq!(
            normalize_prompt_newlines(r"show \\n literally"),
            r"show \n literally"
        );
        assert_eq!(
            normalize_prompt_newlines(r"C:\\new\\render"),
            r"C:\new\render"
        );
    }

    #[test]
    fn leaves_plain_prompts_borrowed_and_unchanged() {
        let prompt = "a portrait with soft window light";
        let normalized = normalize_prompt_newlines(prompt);
        assert!(matches!(normalized, std::borrow::Cow::Borrowed(_)));
        assert_eq!(normalized, prompt);
    }

    #[test]
    fn direct_api_generation_normalizes_all_prompt_provenance() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": r"first\n\nsecond",
            "negative_prompt": r"blur\r\nwatermark",
            "original_prompt": r"first\n\nsecond",
            "prompt_transform": {
                "operation": "expand",
                "root_prompt": r"root\n\nidea",
                "source_prompt": r"source\r\nprompt",
                "source_kind": "direct",
                "task": "text-to-image"
            },
            "model": "flux-schnell",
            "width": 512,
            "height": 512,
            "steps": 4,
            "batch_size": 1
        }))
        .unwrap();

        assert_eq!(request.prompt, "first\n\nsecond");
        assert_eq!(request.negative_prompt.as_deref(), Some("blur\nwatermark"));
        assert_eq!(request.original_prompt.as_deref(), Some("first\n\nsecond"));
        let transform = request.prompt_transform.as_ref().unwrap();
        assert_eq!(transform.root_prompt.as_deref(), Some("root\n\nidea"));
        assert_eq!(transform.source_prompt, "source\nprompt");

        // Legacy embedded metadata is deserialized through the same contract,
        // so Library and Reuse settings improve without regenerating a print.
        let metadata =
            OutputMetadata::from_generate_request(&request, 42, Some(Scheduler::Ddim), "test");
        let mut wire = serde_json::to_value(metadata).unwrap();
        wire["prompt"] = serde_json::Value::String(r"legacy\n\nprompt".to_string());
        wire["negative_prompt"] = serde_json::Value::String(r"legacy\r\nnegative".to_string());
        let restored: OutputMetadata = serde_json::from_value(wire).unwrap();
        assert_eq!(restored.prompt, "legacy\n\nprompt");
        assert_eq!(
            restored.negative_prompt.as_deref(),
            Some("legacy\nnegative")
        );
    }

    #[test]
    fn sequence_and_prompt_transform_api_shapes_share_normalization() {
        let chain: ChainRequest = serde_json::from_value(serde_json::json!({
            "model": "ltx-2-19b-distilled:fp8",
            "stages": [{
                "prompt": r"opening\n\nshot",
                "negative_prompt": r"jitter\r\nflicker",
                "frames": 97
            }],
            "width": 768,
            "height": 512,
            "steps": 8,
            "guidance": 3.0,
            "original_prompt": r"source\n\nidea",
            "prompt": r"automatic\r\nsequence"
        }))
        .unwrap();
        assert_eq!(chain.stages[0].prompt, "opening\n\nshot");
        assert_eq!(
            chain.stages[0].negative_prompt.as_deref(),
            Some("jitter\nflicker")
        );
        assert_eq!(chain.original_prompt.as_deref(), Some("source\n\nidea"));
        assert_eq!(chain.prompt.as_deref(), Some("automatic\nsequence"));

        let expand: ExpandRequest = serde_json::from_value(serde_json::json!({
            "prompt": r"expand\n\nthis"
        }))
        .unwrap();
        assert_eq!(expand.prompt, "expand\n\nthis");

        let remix: RemixRequest = serde_json::from_value(serde_json::json!({
            "source_prompt": r"source\r\nprompt",
            "root_prompt": r"root\n\nprompt"
        }))
        .unwrap();
        assert_eq!(remix.source_prompt, "source\nprompt");
        assert_eq!(remix.root_prompt.as_deref(), Some("root\n\nprompt"));
    }
}
