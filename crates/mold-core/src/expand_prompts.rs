//! Model-family-aware system prompt templates for LLM prompt expansion.
//!
//! Different diffusion models have different prompt styles and token limits.
//! This module provides tailored system prompts for each model family.

use crate::expand::FamilyOverride;

/// Return the word limit and prompt style notes for a given model family.
pub(crate) fn family_config(family: &str) -> (u32, &'static str) {
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

/// Resolve the effective word limit and model notes for a family, applying
/// any user-provided overrides on top of the built-in defaults.
fn resolve_family_config(family: &str, overrides: Option<&FamilyOverride>) -> (u32, String) {
    let (default_limit, default_notes) = family_config(family);
    match overrides {
        Some(ov) => (
            ov.word_limit.unwrap_or(default_limit),
            ov.style_notes
                .clone()
                .unwrap_or_else(|| default_notes.to_string()),
        ),
        None => (default_limit, default_notes.to_string()),
    }
}

/// Build chat messages for a single prompt expansion.
///
/// Accepts optional custom template and per-family overrides.
/// Returns `Vec<(role, content)>` tuples.
pub fn build_single_messages(
    prompt: &str,
    family: &str,
    custom_template: Option<&str>,
    family_override: Option<&FamilyOverride>,
) -> Vec<(String, String)> {
    let (word_limit, model_notes) = resolve_family_config(family, family_override);
    let template = custom_template.unwrap_or(SINGLE_SYSTEM_TEMPLATE);
    let system = template
        .replace("{WORD_LIMIT}", &word_limit.to_string())
        .replace("{MODEL_NOTES}", &model_notes);

    vec![
        ("system".to_string(), system),
        ("user".to_string(), prompt.to_string()),
    ]
}

/// Build chat messages for batch variation generation.
///
/// Accepts optional custom template and per-family overrides.
/// Returns `Vec<(role, content)>` tuples.
pub fn build_batch_messages(
    prompt: &str,
    family: &str,
    variations: usize,
    custom_template: Option<&str>,
    family_override: Option<&FamilyOverride>,
) -> Vec<(String, String)> {
    let (word_limit, model_notes) = resolve_family_config(family, family_override);
    let template = custom_template.unwrap_or(BATCH_SYSTEM_TEMPLATE);
    let system = template
        .replace("{N}", &variations.to_string())
        .replace("{WORD_LIMIT}", &word_limit.to_string())
        .replace("{MODEL_NOTES}", &model_notes);

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

    // ── family_config ────────────────────────────────────────────────────

    #[test]
    fn family_config_sd15_variants() {
        // All SD 1.5 aliases should resolve to the same config
        for family in &["sd15", "sd1.5", "stable-diffusion-1.5"] {
            let (limit, notes) = family_config(family);
            assert_eq!(limit, 50);
            assert!(notes.contains("keyword"));
        }
    }

    #[test]
    fn family_config_sdxl() {
        let (limit, notes) = family_config("sdxl");
        assert_eq!(limit, 60);
        assert!(notes.contains("CLIP-L + CLIP-G"));
    }

    #[test]
    fn family_config_wuerstchen_variants() {
        for family in &["wuerstchen", "wuerstchen-v2"] {
            let (limit, _) = family_config(family);
            assert_eq!(limit, 50);
        }
    }

    #[test]
    fn family_config_flux_uses_long_default() {
        let (limit, notes) = family_config("flux");
        assert_eq!(limit, 150);
        assert!(notes.contains("natural language"));
    }

    #[test]
    fn family_config_sd3_uses_long_default() {
        let (limit, _) = family_config("sd3");
        assert_eq!(limit, 150);
    }

    #[test]
    fn family_config_zimage_uses_long_default() {
        let (limit, _) = family_config("z-image");
        assert_eq!(limit, 150);
    }

    #[test]
    fn family_config_unknown_uses_long_default() {
        let (limit, _) = family_config("some-future-model");
        assert_eq!(limit, 150);
    }

    #[test]
    fn family_config_case_insensitive() {
        let (limit, _) = family_config("SD15");
        assert_eq!(limit, 50);
        let (limit, _) = family_config("SDXL");
        assert_eq!(limit, 60);
    }

    // ── build_single_messages ────────────────────────────────────────────

    #[test]
    fn single_messages_flux() {
        let msgs = build_single_messages("a cat", "flux", None, None);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].0, "system");
        assert!(msgs[0].1.contains("150 words"));
        assert_eq!(msgs[1].0, "user");
        assert_eq!(msgs[1].1, "a cat");
    }

    #[test]
    fn single_messages_sd15() {
        let msgs = build_single_messages("a cat", "sd15", None, None);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].1.contains("50 words"));
        assert!(msgs[0].1.contains("keyword"));
    }

    #[test]
    fn single_messages_preserves_user_prompt() {
        let prompt = "a cyberpunk city at night with neon reflections";
        let msgs = build_single_messages(prompt, "flux", None, None);
        assert_eq!(msgs[1].1, prompt);
    }

    // ── build_batch_messages ─────────────────────────────────────────────

    #[test]
    fn batch_messages_sdxl() {
        let msgs = build_batch_messages("sunset", "sdxl", 3, None, None);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].1.contains("3 distinct"));
        assert!(msgs[0].1.contains("JSON array"));
        assert!(msgs[0].1.contains("60 words"));
    }

    #[test]
    fn batch_messages_count_substitution() {
        for n in [2, 5, 10] {
            let msgs = build_batch_messages("test", "flux", n, None, None);
            assert!(
                msgs[0].1.contains(&format!("{n} distinct")),
                "should contain '{n} distinct' for variations={n}"
            );
        }
    }

    #[test]
    fn batch_messages_preserves_user_prompt() {
        let prompt = "a dragon in a crystal cave";
        let msgs = build_batch_messages(prompt, "sdxl", 4, None, None);
        assert_eq!(msgs[1].1, prompt);
    }

    // ── custom templates and family overrides ────────────────────────────

    #[test]
    fn single_messages_custom_template() {
        let custom = "Custom system: limit {WORD_LIMIT}. Notes: {MODEL_NOTES}";
        let msgs = build_single_messages("a cat", "flux", Some(custom), None);
        assert!(msgs[0].1.contains("Custom system: limit 150"));
        assert!(msgs[0].1.contains("natural language"));
    }

    #[test]
    fn batch_messages_custom_template() {
        let custom = "Generate {N} prompts, max {WORD_LIMIT} words. {MODEL_NOTES}";
        let msgs = build_batch_messages("a cat", "flux", 3, Some(custom), None);
        assert!(msgs[0].1.contains("Generate 3 prompts"));
        assert!(msgs[0].1.contains("max 150 words"));
    }

    #[test]
    fn single_messages_family_override_word_limit() {
        let ov = FamilyOverride {
            word_limit: Some(200),
            style_notes: None,
        };
        let msgs = build_single_messages("a cat", "flux", None, Some(&ov));
        assert!(msgs[0].1.contains("200 words"));
        // Should still use built-in style notes
        assert!(msgs[0].1.contains("natural language"));
    }

    #[test]
    fn single_messages_family_override_style_notes() {
        let ov = FamilyOverride {
            word_limit: None,
            style_notes: Some("Use haiku style.".to_string()),
        };
        let msgs = build_single_messages("a cat", "sd15", None, Some(&ov));
        // Word limit should still be the SD1.5 default (50)
        assert!(msgs[0].1.contains("50 words"));
        // But style notes should be overridden
        assert!(msgs[0].1.contains("Use haiku style."));
        assert!(!msgs[0].1.contains("keyword"));
    }

    #[test]
    fn batch_messages_family_override_both() {
        let ov = FamilyOverride {
            word_limit: Some(75),
            style_notes: Some("Cinematic descriptions only.".to_string()),
        };
        let msgs = build_batch_messages("a cat", "sdxl", 4, None, Some(&ov));
        assert!(msgs[0].1.contains("75 words"));
        assert!(msgs[0].1.contains("Cinematic descriptions only."));
    }

    #[test]
    fn custom_template_with_family_override() {
        let custom = "Limit: {WORD_LIMIT}. Style: {MODEL_NOTES}";
        let ov = FamilyOverride {
            word_limit: Some(300),
            style_notes: Some("Go wild.".to_string()),
        };
        let msgs = build_single_messages("test", "flux", Some(custom), Some(&ov));
        assert_eq!(msgs[0].1, "Limit: 300. Style: Go wild.");
    }

    #[test]
    fn resolve_family_config_defaults_preserved() {
        let (limit, notes) = resolve_family_config("sd15", None);
        assert_eq!(limit, 50);
        assert!(notes.contains("keyword"));
    }

    #[test]
    fn resolve_family_config_partial_override() {
        let ov = FamilyOverride {
            word_limit: Some(100),
            style_notes: None,
        };
        let (limit, notes) = resolve_family_config("sd15", Some(&ov));
        assert_eq!(limit, 100);
        // Notes should fall back to default
        assert!(notes.contains("keyword"));
    }

    // ── format_chatml ────────────────────────────────────────────────────

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

    #[test]
    fn chatml_ends_with_assistant_prefix() {
        let msgs = vec![("user".to_string(), "test".to_string())];
        for thinking in [true, false] {
            let result = format_chatml(&msgs, thinking);
            assert!(
                result.contains("<|im_start|>assistant\n"),
                "should end with assistant prefix"
            );
        }
    }

    #[test]
    fn chatml_empty_messages() {
        let msgs: Vec<(String, String)> = vec![];
        let result = format_chatml(&msgs, false);
        // Should still have assistant prefix + thinking disable
        assert!(result.starts_with("<|im_start|>assistant\n"));
    }
}
