use serde::{Deserialize, Deserializer};
use std::borrow::Cow;

/// Canonicalize prompt line endings and decode newline escapes left literal
/// by shell arguments or over-encoded API clients.
///
/// A doubled backslash preserves a literal escape (`\\\\n` -> `\\n`), so a
/// prompt can still describe source text or paths containing those characters.
/// This transformation is deliberately single-pass and must be applied at
/// exactly one input boundary.
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

fn protect_prompt_for_wire(prompt: &mut String) {
    if prompt.contains('\\') {
        *prompt = prompt.replace('\\', "\\\\");
    }
}

pub(crate) fn protect_generate_request_for_wire(
    request: &crate::GenerateRequest,
) -> crate::GenerateRequest {
    let mut wire = request.clone();
    protect_prompt_for_wire(&mut wire.prompt);
    if let Some(prompt) = wire.negative_prompt.as_mut() {
        protect_prompt_for_wire(prompt);
    }
    if let Some(prompt) = wire.original_prompt.as_mut() {
        protect_prompt_for_wire(prompt);
    }
    if let Some(transform) = wire.prompt_transform.as_mut() {
        if let Some(prompt) = transform.root_prompt.as_mut() {
            protect_prompt_for_wire(prompt);
        }
        protect_prompt_for_wire(&mut transform.source_prompt);
    }
    wire
}

pub(crate) fn protect_chain_request_for_wire(request: &crate::ChainRequest) -> crate::ChainRequest {
    let mut wire = request.clone();
    for stage in &mut wire.stages {
        protect_prompt_for_wire(&mut stage.prompt);
        if let Some(prompt) = stage.negative_prompt.as_mut() {
            protect_prompt_for_wire(prompt);
        }
    }
    if let Some(prompt) = wire.original_prompt.as_mut() {
        protect_prompt_for_wire(prompt);
    }
    if let Some(prompt) = wire.prompt.as_mut() {
        protect_prompt_for_wire(prompt);
    }
    if let Some(transform) = wire.prompt_transform.as_mut() {
        if let Some(prompt) = transform.root_prompt.as_mut() {
            protect_prompt_for_wire(prompt);
        }
        protect_prompt_for_wire(&mut transform.source_prompt);
    }
    wire
}

pub(crate) fn protect_expand_request_for_wire(
    request: &crate::ExpandRequest,
) -> crate::ExpandRequest {
    let mut wire = request.clone();
    protect_prompt_for_wire(&mut wire.prompt);
    wire
}

pub(crate) fn protect_remix_request_for_wire(request: &crate::RemixRequest) -> crate::RemixRequest {
    let mut wire = request.clone();
    protect_prompt_for_wire(&mut wire.source_prompt);
    if let Some(prompt) = wire.root_prompt.as_mut() {
        protect_prompt_for_wire(prompt);
    }
    wire
}

pub(crate) fn protect_retake_request_for_wire(
    request: &crate::chain_job::RetakeRequest,
) -> crate::chain_job::RetakeRequest {
    let mut wire = request.clone();
    if let Some(prompt) = wire.prompt.as_mut() {
        protect_prompt_for_wire(prompt);
    }
    wire
}

#[cfg(test)]
mod tests {
    use super::{
        normalize_prompt_newlines, protect_chain_request_for_wire, protect_expand_request_for_wire,
        protect_generate_request_for_wire, protect_remix_request_for_wire,
        protect_retake_request_for_wire,
    };
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
        let mut chain: ChainRequest = serde_json::from_value(serde_json::json!({
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
        chain.normalize_prompt_newlines();
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

    #[test]
    fn shared_rust_client_wire_hop_preserves_canonical_prompt_text() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "placeholder",
            "model": "flux-schnell",
            "width": 512,
            "height": 512,
            "steps": 4,
            "batch_size": 1
        }))
        .unwrap();
        let mut request = request;
        request.prompt = r"canonical C:\new\render and \n token".to_string();
        request.negative_prompt = Some("blur\nwatermark\\n literal".to_string());
        request.original_prompt = Some("source\n\nidea\\n literal".to_string());
        let admitted: GenerateRequest = serde_json::from_slice(
            &serde_json::to_vec(&protect_generate_request_for_wire(&request)).unwrap(),
        )
        .unwrap();
        assert_eq!(admitted.prompt, request.prompt);
        assert_eq!(admitted.negative_prompt, request.negative_prompt);
        assert_eq!(admitted.original_prompt, request.original_prompt);

        let expand = ExpandRequest {
            prompt: r"expand C:\new and \n token".to_string(),
            model_family: "flux".to_string(),
            variations: 1,
            style: None,
            task: None,
        };
        let admitted: ExpandRequest = serde_json::from_slice(
            &serde_json::to_vec(&protect_expand_request_for_wire(&expand)).unwrap(),
        )
        .unwrap();
        assert_eq!(admitted.prompt, expand.prompt);

        let remix = RemixRequest {
            source_prompt: r"remix C:\new and \n token".to_string(),
            root_prompt: Some("root\nidea\\n literal".to_string()),
            ..serde_json::from_value(serde_json::json!({
                "source_prompt": "placeholder"
            }))
            .unwrap()
        };
        let admitted: RemixRequest = serde_json::from_slice(
            &serde_json::to_vec(&protect_remix_request_for_wire(&remix)).unwrap(),
        )
        .unwrap();
        assert_eq!(admitted.source_prompt, remix.source_prompt);
        assert_eq!(admitted.root_prompt, remix.root_prompt);

        let mut chain: ChainRequest = serde_json::from_value(serde_json::json!({
            "model": "ltx-2-19b-distilled:fp8",
            "stages": [{
                "prompt": "placeholder",
                "frames": 97
            }],
            "width": 768,
            "height": 512,
            "steps": 8,
            "guidance": 3.0
        }))
        .unwrap();
        chain.stages[0].prompt = r"sequence C:\new and \n token".to_string();
        chain.stages[0].negative_prompt = Some("jitter\nflicker\\n literal".to_string());
        chain.original_prompt = Some("source\nidea\\n literal".to_string());
        let mut admitted: ChainRequest = serde_json::from_slice(
            &serde_json::to_vec(&protect_chain_request_for_wire(&chain)).unwrap(),
        )
        .unwrap();
        admitted.normalize_prompt_newlines();
        assert_eq!(admitted.stages[0].prompt, chain.stages[0].prompt);
        assert_eq!(
            admitted.stages[0].negative_prompt,
            chain.stages[0].negative_prompt
        );
        assert_eq!(admitted.original_prompt, chain.original_prompt);

        let retake = crate::chain_job::RetakeRequest {
            stage_idx: 0,
            mode: crate::chain_job::RetakeMode::Cascade,
            seed_offset: None,
            prompt: Some("retake\nidea\\n literal".to_string()),
        };
        let mut admitted: crate::chain_job::RetakeRequest = serde_json::from_slice(
            &serde_json::to_vec(&protect_retake_request_for_wire(&retake)).unwrap(),
        )
        .unwrap();
        admitted.normalize_prompt_newlines();
        assert_eq!(admitted.prompt, retake.prompt);
    }
}
