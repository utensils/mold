//! Model-family-aware system prompt templates for LLM prompt expansion.
//!
//! Different diffusion models have different prompt styles and token limits.
//! This module provides tailored system prompts for each model family.

/// Return the word limit and prompt style notes for a given model family.
fn family_config(family: &str) -> (u32, &'static str) {
    match family.to_lowercase().as_str() {
        "sd15" | "sd1.5" | "stable-diffusion-1.5" => (
            50,
            "SD 1.5 uses CLIP-L (77 tokens). Use comma-separated keyword phrases. \
             Include quality tokens like 'masterpiece, best quality, detailed'. Keep under 50 words.",
        ),
        "sdxl" => (
            60,
            "SDXL uses dual CLIP (CLIP-L + CLIP-G, 77 tokens). Mix natural language with \
             style/quality keywords. Include art style and quality descriptors. Keep under 60 words.",
        ),
        "wuerstchen" | "wuerstchen-v2" => (
            50,
            "Wuerstchen uses CLIP-G (77 tokens). Use short descriptive keyword phrases. \
             Keep under 50 words.",
        ),
        // All flow-matching models with T5/Qwen3 encoders support longer prompts
        _ => (
            150,
            "This model uses a large text encoder (T5-XXL or Qwen3) that understands natural language well. \
             Write descriptive, vivid natural language. Include composition, lighting, color palette, \
             textures, atmosphere, and camera angle. Up to 150 words.",
        ),
    }
}

const SINGLE_SYSTEM_TEMPLATE: &str = "\
You are an image generation prompt writer. Expand the user's brief description into a detailed, vivid image prompt.

Rules:
1. PRESERVE the user's core subject and intent exactly.
2. ADD: composition, lighting, color palette, textures, atmosphere, camera angle.
3. Use comma-separated descriptive phrases.
4. Keep under {WORD_LIMIT} words.
5. Output ONLY the expanded prompt, nothing else. No preamble, no explanation.

{MODEL_NOTES}";

const BATCH_SYSTEM_TEMPLATE: &str = "\
You are an image generation prompt writer. Generate {N} distinct image prompts based on the user's concept.

Each prompt must:
- Keep the core concept but explore a DIFFERENT angle
- Vary at least 2 of: time of day, weather, camera angle, color palette, mood, artistic style
- Be self-contained and under {WORD_LIMIT} words
- Use comma-separated descriptive phrases

Output as a JSON array of {N} strings, nothing else. No preamble, no explanation.
Example format: [\"prompt one\", \"prompt two\"]

{MODEL_NOTES}";

/// Build chat messages for a single prompt expansion.
///
/// Returns `Vec<(role, content)>` tuples.
pub fn build_single_messages(prompt: &str, family: &str) -> Vec<(String, String)> {
    let (word_limit, model_notes) = family_config(family);
    let system = SINGLE_SYSTEM_TEMPLATE
        .replace("{WORD_LIMIT}", &word_limit.to_string())
        .replace("{MODEL_NOTES}", model_notes);

    vec![
        ("system".to_string(), system),
        ("user".to_string(), prompt.to_string()),
    ]
}

/// Build chat messages for batch variation generation.
///
/// Returns `Vec<(role, content)>` tuples.
pub fn build_batch_messages(
    prompt: &str,
    family: &str,
    variations: usize,
) -> Vec<(String, String)> {
    let (word_limit, model_notes) = family_config(family);
    let system = BATCH_SYSTEM_TEMPLATE
        .replace("{N}", &variations.to_string())
        .replace("{WORD_LIMIT}", &word_limit.to_string())
        .replace("{MODEL_NOTES}", model_notes);

    vec![
        ("system".to_string(), system),
        ("user".to_string(), prompt.to_string()),
    ]
}

/// Format a ChatML prompt string for local Qwen3 inference.
///
/// When `thinking` is false, appends `<think>\n\n</think>\n\n` after the
/// assistant prefix to disable thinking mode (same pattern as Flux.2 Klein).
pub fn format_chatml(messages: &[(String, String)], thinking: bool) -> String {
    let mut result = String::new();
    for (role, content) in messages {
        result.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    result.push_str("<|im_start|>assistant\n");
    if !thinking {
        result.push_str("<think>\n\n</think>\n\n");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_messages_flux() {
        let msgs = build_single_messages("a cat", "flux");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].0, "system");
        assert!(msgs[0].1.contains("150 words"));
        assert_eq!(msgs[1].0, "user");
        assert_eq!(msgs[1].1, "a cat");
    }

    #[test]
    fn single_messages_sd15() {
        let msgs = build_single_messages("a cat", "sd15");
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].1.contains("50 words"));
        assert!(msgs[0].1.contains("keyword"));
    }

    #[test]
    fn batch_messages_sdxl() {
        let msgs = build_batch_messages("sunset", "sdxl", 3);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].1.contains("3 distinct"));
        assert!(msgs[0].1.contains("JSON array"));
        assert!(msgs[0].1.contains("60 words"));
    }

    #[test]
    fn chatml_without_thinking() {
        let msgs = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "hello".to_string()),
        ];
        let result = format_chatml(&msgs, false);
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn chatml_with_thinking() {
        let msgs = vec![("user".to_string(), "hello".to_string())];
        let result = format_chatml(&msgs, true);
        assert!(result.contains("<|im_start|>assistant\n"));
        assert!(!result.contains("<think>"));
    }
}
