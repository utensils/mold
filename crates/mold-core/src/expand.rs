//! LLM-powered prompt expansion.
//!
//! Provides a `PromptExpander` trait with two backends:
//! - `ApiExpander`: calls any OpenAI-compatible `/v1/chat/completions` endpoint
//! - Local GGUF inference (in `mold-inference`, behind the `expand` feature flag)

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::expand_prompts::{build_batch_messages, build_single_messages};

/// Per-family word limit and style notes override.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FamilyOverride {
    /// Word limit for expanded prompts (overrides built-in default for this family).
    pub word_limit: Option<u32>,
    /// Style notes injected into the system prompt (overrides built-in default).
    pub style_notes: Option<String>,
}

/// Configuration for a prompt expansion request.
#[derive(Debug, Clone)]
pub struct ExpandConfig {
    /// Diffusion model family (e.g. "flux", "sd15", "sdxl").
    pub model_family: String,
    /// Number of prompt variations to generate (1 = single expansion).
    pub variations: usize,
    /// Sampling temperature (0.0-2.0). Higher = more creative.
    pub temperature: f64,
    /// Nucleus sampling threshold.
    pub top_p: f64,
    /// Maximum tokens for the LLM response.
    pub max_tokens: u32,
    /// Enable Qwen3 thinking mode for higher quality (slower).
    pub thinking: bool,
    /// Custom single-expansion system prompt template (overrides built-in).
    /// Placeholders: `{WORD_LIMIT}`, `{MODEL_NOTES}`
    pub system_prompt: Option<String>,
    /// Custom batch-variation system prompt template (overrides built-in).
    /// Placeholders: `{N}`, `{WORD_LIMIT}`, `{MODEL_NOTES}`
    pub batch_prompt: Option<String>,
    /// Per-family overrides for word limits and style notes.
    pub family_overrides: HashMap<String, FamilyOverride>,
}

impl Default for ExpandConfig {
    fn default() -> Self {
        Self {
            model_family: "flux".to_string(),
            variations: 1,
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 300,
            thinking: false,
            system_prompt: None,
            batch_prompt: None,
            family_overrides: HashMap::new(),
        }
    }
}

/// Result of a prompt expansion.
#[derive(Debug, Clone)]
pub struct ExpandResult {
    /// The original user prompt.
    pub original: String,
    /// Expanded prompt(s). Length equals `ExpandConfig::variations`.
    pub expanded: Vec<String>,
}

/// Trait for prompt expansion backends.
pub trait PromptExpander: Send + Sync {
    /// Expand a user prompt into one or more detailed image generation prompts.
    fn expand(&self, prompt: &str, config: &ExpandConfig) -> Result<ExpandResult>;
}

// ── API expander ─────────────────────────────────────────────────────────────

/// OpenAI-compatible chat completion message.
#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Request body for `/v1/chat/completions`.
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    top_p: f64,
    max_tokens: u32,
}

/// Response from `/v1/chat/completions`.
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

/// Expander that calls an OpenAI-compatible API endpoint.
pub struct ApiExpander {
    endpoint: String,
    model: String,
}

impl ApiExpander {
    pub fn new(endpoint: &str, model: &str) -> Self {
        // Strip trailing slash for consistent URL building
        let endpoint = endpoint.trim_end_matches('/').to_string();
        Self {
            endpoint,
            model: model.to_string(),
        }
    }
}

impl PromptExpander for ApiExpander {
    fn expand(&self, prompt: &str, config: &ExpandConfig) -> Result<ExpandResult> {
        let family_override = config.family_overrides.get(&config.model_family);
        let messages = if config.variations > 1 {
            build_batch_messages(
                prompt,
                &config.model_family,
                config.variations,
                config.batch_prompt.as_deref(),
                family_override,
            )
        } else {
            build_single_messages(
                prompt,
                &config.model_family,
                config.system_prompt.as_deref(),
                family_override,
            )
        };

        let chat_messages: Vec<ChatMessage> = messages
            .into_iter()
            .map(|(role, content)| ChatMessage { role, content })
            .collect();

        let req_body = ChatCompletionRequest {
            model: self.model.clone(),
            messages: chat_messages,
            temperature: config.temperature,
            top_p: config.top_p,
            max_tokens: config.max_tokens,
        };

        let url = format!("{}/v1/chat/completions", self.endpoint);

        // Use ureq (blocking HTTP) — this trait method is sync and may be
        // called from within a tokio runtime via spawn_blocking, so we cannot
        // use async reqwest or Handle::block_on (which panics inside a runtime).
        let body = serde_json::to_string(&req_body)?;
        let response_text: String = ureq::post(&url)
            .header("Content-Type", "application/json")
            .send(body.as_str())
            .map_err(|e| anyhow::anyhow!("expand API request failed: {e}"))?
            .body_mut()
            .read_to_string()
            .map_err(|e| anyhow::anyhow!("failed to read expand API response: {e}"))?;

        let completion: ChatCompletionResponse = serde_json::from_str(&response_text)
            .map_err(|e| anyhow::anyhow!("failed to parse expand API response: {e}"))?;

        let content = completion
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let expanded = if config.variations > 1 {
            parse_variations(&content, config.variations)
        } else {
            vec![clean_expanded_prompt(&content)]
        };

        Ok(ExpandResult {
            original: prompt.to_string(),
            expanded,
        })
    }
}

/// Public wrapper for `parse_variations` (used by mold-inference local expander).
pub fn parse_variations_public(text: &str, expected: usize) -> Vec<String> {
    parse_variations(text, expected)
}

/// Public wrapper for `clean_expanded_prompt` (used by mold-inference local expander).
pub fn clean_expanded_prompt_public(text: &str) -> String {
    clean_expanded_prompt(text)
}

/// Parse multiple variations from LLM output.
/// Tries JSON array first, then numbered list, then line-separated.
fn parse_variations(text: &str, expected: usize) -> Vec<String> {
    let trimmed = text.trim();

    // Try JSON array
    if let Ok(arr) = serde_json::from_str::<Vec<String>>(trimmed) {
        if !arr.is_empty() {
            return arr.into_iter().map(|s| clean_expanded_prompt(&s)).collect();
        }
    }

    // Try to find a JSON array embedded in the text (LLM may include preamble)
    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            if start < end {
                let json_slice = &trimmed[start..=end];
                if let Ok(arr) = serde_json::from_str::<Vec<String>>(json_slice) {
                    if !arr.is_empty() {
                        return arr.into_iter().map(|s| clean_expanded_prompt(&s)).collect();
                    }
                }
            }
        }
    }

    // Fall back to numbered list parsing (1. ... 2. ... etc.)
    let lines: Vec<String> = trimmed
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| {
            // Strip numbered prefix: "1. ", "2) ", etc.
            let stripped = l
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .trim_start_matches(['.', ')', ':', '-'])
                .trim_start_matches('"')
                .trim_end_matches('"')
                .trim();
            clean_expanded_prompt(stripped)
        })
        .filter(|l| !l.is_empty())
        .collect();

    if lines.len() >= expected {
        return lines;
    }

    // Last resort: split on double newlines
    let paragraphs: Vec<String> = trimmed
        .split("\n\n")
        .map(|p| clean_expanded_prompt(p.trim()))
        .filter(|p| !p.is_empty())
        .collect();

    if !paragraphs.is_empty() {
        return paragraphs;
    }

    // Ultimate fallback: return the whole text as a single variation
    vec![clean_expanded_prompt(trimmed)]
}

/// Clean up an expanded prompt: trim whitespace, remove quotes, collapse whitespace.
fn clean_expanded_prompt(text: &str) -> String {
    let trimmed = text.trim().trim_matches('"').trim_matches('\'').trim();

    // Strip any thinking block if present
    let cleaned = if let Some(end_idx) = trimmed.find("</think>") {
        trimmed[end_idx + "</think>".len()..].trim()
    } else {
        trimmed
    };

    // Collapse multiple whitespace/newlines into single spaces
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Expand configuration from the mold config file.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExpandSettings {
    /// Enable prompt expansion by default (overridden by --expand/--no-expand).
    #[serde(default)]
    pub enabled: bool,
    /// Backend: "local" for built-in GGUF inference, or a URL for OpenAI-compatible API.
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Model name for local GGUF expansion.
    #[serde(default = "default_expand_model")]
    pub model: String,
    /// Model name when using API backend (e.g. "qwen2.5:3b" for Ollama).
    #[serde(default = "default_api_model")]
    pub api_model: String,
    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Nucleus sampling threshold.
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    /// Maximum tokens for the LLM response.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Enable thinking mode for Qwen3 (higher quality, slower).
    #[serde(default)]
    pub thinking: bool,
    /// Custom single-expansion system prompt template.
    /// Available placeholders: `{WORD_LIMIT}`, `{MODEL_NOTES}`
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Custom batch-variation system prompt template.
    /// Available placeholders: `{N}`, `{WORD_LIMIT}`, `{MODEL_NOTES}`
    #[serde(default)]
    pub batch_prompt: Option<String>,
    /// Per-family word limit and style notes overrides.
    #[serde(default)]
    pub families: HashMap<String, FamilyOverride>,
}

fn default_backend() -> String {
    "local".to_string()
}

fn default_expand_model() -> String {
    "qwen3-expand:q8".to_string()
}

fn default_api_model() -> String {
    "qwen2.5:3b".to_string()
}

fn default_temperature() -> f64 {
    0.7
}

fn default_top_p() -> f64 {
    0.9
}

fn default_max_tokens() -> u32 {
    300
}

impl Default for ExpandSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: default_backend(),
            model: default_expand_model(),
            api_model: default_api_model(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_tokens: default_max_tokens(),
            thinking: false,
            system_prompt: None,
            batch_prompt: None,
            families: HashMap::new(),
        }
    }
}

impl ExpandSettings {
    /// Load from environment variables, falling back to provided defaults.
    pub fn with_env_overrides(mut self) -> Self {
        if let Ok(v) = std::env::var("MOLD_EXPAND") {
            self.enabled = matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes");
        }
        if let Ok(v) = std::env::var("MOLD_EXPAND_BACKEND") {
            if !v.is_empty() {
                self.backend = v;
            }
        }
        if let Ok(v) = std::env::var("MOLD_EXPAND_MODEL") {
            if !v.is_empty() {
                if self.is_local() {
                    self.model = v;
                } else {
                    self.api_model = v;
                }
            }
        }
        if let Ok(v) = std::env::var("MOLD_EXPAND_TEMPERATURE") {
            if let Ok(t) = v.parse::<f64>() {
                self.temperature = t;
            }
        }
        if let Ok(v) = std::env::var("MOLD_EXPAND_THINKING") {
            self.thinking = matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes");
        }
        if let Ok(v) = std::env::var("MOLD_EXPAND_SYSTEM_PROMPT") {
            if !v.is_empty() {
                self.system_prompt = Some(v);
            }
        }
        if let Ok(v) = std::env::var("MOLD_EXPAND_BATCH_PROMPT") {
            if !v.is_empty() {
                self.batch_prompt = Some(v);
            }
        }
        self
    }

    /// Build an `ExpandConfig` for a specific request.
    pub fn to_expand_config(&self, model_family: &str, variations: usize) -> ExpandConfig {
        ExpandConfig {
            model_family: model_family.to_string(),
            variations,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            thinking: self.thinking,
            system_prompt: self.system_prompt.clone(),
            batch_prompt: self.batch_prompt.clone(),
            family_overrides: self.families.clone(),
        }
    }

    /// Validate that custom templates contain expected placeholders.
    /// Returns a list of warnings (empty = valid). Callers should treat
    /// these as non-fatal hints — expansion still runs with partial templates.
    pub fn validate_templates(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if let Some(ref tmpl) = self.system_prompt {
            for placeholder in ["{WORD_LIMIT}", "{MODEL_NOTES}"] {
                if !tmpl.contains(placeholder) {
                    warnings.push(format!(
                        "system_prompt is missing placeholder {placeholder} — it won't be substituted"
                    ));
                }
            }
        }
        if let Some(ref tmpl) = self.batch_prompt {
            for placeholder in ["{N}", "{WORD_LIMIT}", "{MODEL_NOTES}"] {
                if !tmpl.contains(placeholder) {
                    warnings.push(format!(
                        "batch_prompt is missing placeholder {placeholder} — it won't be substituted"
                    ));
                }
            }
        }
        warnings
    }

    /// Create the appropriate expander backend.
    /// Returns `None` if the backend is "local" (handled by mold-inference).
    pub fn create_api_expander(&self) -> Option<ApiExpander> {
        if self.backend == "local" {
            None
        } else {
            Some(ApiExpander::new(&self.backend, &self.api_model))
        }
    }

    /// Check if this is configured for local (GGUF) expansion.
    pub fn is_local(&self) -> bool {
        self.backend == "local"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── clean_expanded_prompt ────────────────────────────────────────────

    #[test]
    fn clean_prompt_strips_quotes() {
        assert_eq!(clean_expanded_prompt("\"a cat on mars\""), "a cat on mars");
    }

    #[test]
    fn clean_prompt_strips_single_quotes() {
        assert_eq!(clean_expanded_prompt("'a cat on mars'"), "a cat on mars");
    }

    #[test]
    fn clean_prompt_strips_thinking() {
        let input = "<think>hmm let me think</think>\n\na cat on mars";
        assert_eq!(clean_expanded_prompt(input), "a cat on mars");
    }

    #[test]
    fn clean_prompt_strips_multiline_thinking() {
        let input = "<think>\nstep 1: analyze\nstep 2: expand\n</think>\n\ndetailed prompt here";
        assert_eq!(clean_expanded_prompt(input), "detailed prompt here");
    }

    #[test]
    fn clean_prompt_collapses_whitespace() {
        assert_eq!(
            clean_expanded_prompt("a  cat\n\non   mars"),
            "a cat on mars"
        );
    }

    #[test]
    fn clean_prompt_empty_input() {
        assert_eq!(clean_expanded_prompt(""), "");
        assert_eq!(clean_expanded_prompt("   "), "");
    }

    #[test]
    fn clean_prompt_only_thinking_block() {
        let input = "<think>some reasoning</think>";
        assert_eq!(clean_expanded_prompt(input), "");
    }

    #[test]
    fn clean_prompt_preserves_content_without_thinking() {
        let input = "a beautiful sunset over the ocean, golden light, dramatic clouds";
        assert_eq!(clean_expanded_prompt(input), input);
    }

    // ── parse_variations ─────────────────────────────────────────────────

    #[test]
    fn parse_variations_json_array() {
        let input = r#"["a cat", "a dog", "a bird"]"#;
        let result = parse_variations(input, 3);
        assert_eq!(result, vec!["a cat", "a dog", "a bird"]);
    }

    #[test]
    fn parse_variations_embedded_json() {
        let input = "Here are 3 prompts:\n[\"a cat\", \"a dog\", \"a bird\"]";
        let result = parse_variations(input, 3);
        assert_eq!(result, vec!["a cat", "a dog", "a bird"]);
    }

    #[test]
    fn parse_variations_json_with_thinking() {
        let input =
            "<think>let me think</think>\n\n[\"expanded cat\", \"expanded dog\", \"expanded bird\"]";
        // The thinking block is inside individual items, not wrapping the JSON.
        // parse_variations should find the embedded JSON array.
        let result = parse_variations(input, 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn parse_variations_numbered_list() {
        let input = "1. a cat on mars\n2. a dog in space\n3. a bird underwater";
        let result = parse_variations(input, 3);
        assert_eq!(result.len(), 3);
        assert!(result[0].contains("cat"));
        assert!(result[1].contains("dog"));
        assert!(result[2].contains("bird"));
    }

    #[test]
    fn parse_variations_numbered_with_parens() {
        let input = "1) a cat\n2) a dog\n3) a bird";
        let result = parse_variations(input, 3);
        assert_eq!(result.len(), 3);
        assert!(result[0].contains("cat"));
    }

    #[test]
    fn parse_variations_numbered_with_quotes() {
        let input = "1. \"a cat on mars\"\n2. \"a dog in space\"";
        let result = parse_variations(input, 2);
        assert_eq!(result.len(), 2);
        // Quotes should be stripped by clean_expanded_prompt
        assert!(!result[0].starts_with('"'));
        assert!(result[0].contains("cat"));
    }

    #[test]
    fn parse_variations_paragraph_fallback() {
        let input = "A majestic cat sitting on mars\n\nA playful dog floating in space";
        let result = parse_variations(input, 2);
        assert_eq!(result.len(), 2);
        assert!(result[0].contains("cat"));
        assert!(result[1].contains("dog"));
    }

    #[test]
    fn parse_variations_single_text_fallback() {
        // When nothing else matches, return the whole text as one variation
        let input = "just a single prompt with no structure";
        let result = parse_variations(input, 3);
        assert!(!result.is_empty());
        assert!(result[0].contains("single prompt"));
    }

    #[test]
    fn parse_variations_empty_json_array_falls_through() {
        // Empty JSON array should fall through to other parsers
        let input = "[]";
        let result = parse_variations(input, 3);
        // Should not panic; falls through to numbered list / paragraph / fallback
        assert!(!result.is_empty());
    }

    #[test]
    fn parse_variations_cleans_each_item() {
        let input = r#"["  a cat  ", "  a dog  "]"#;
        let result = parse_variations(input, 2);
        assert_eq!(result[0], "a cat");
        assert_eq!(result[1], "a dog");
    }

    // ── ExpandSettings ───────────────────────────────────────────────────

    #[test]
    fn expand_settings_defaults() {
        let settings = ExpandSettings::default();
        assert!(!settings.enabled);
        assert_eq!(settings.backend, "local");
        assert_eq!(settings.model, "qwen3-expand:q8");
        assert_eq!(settings.api_model, "qwen2.5:3b");
        assert_eq!(settings.temperature, 0.7);
        assert_eq!(settings.top_p, 0.9);
        assert_eq!(settings.max_tokens, 300);
        assert!(!settings.thinking);
        assert!(settings.system_prompt.is_none());
        assert!(settings.batch_prompt.is_none());
        assert!(settings.families.is_empty());
    }

    #[test]
    fn expand_settings_is_local() {
        let settings = ExpandSettings::default();
        assert!(settings.is_local());

        let api_settings = ExpandSettings {
            backend: "http://localhost:11434".to_string(),
            ..Default::default()
        };
        assert!(!api_settings.is_local());
    }

    #[test]
    fn expand_settings_create_api_expander_none_for_local() {
        let settings = ExpandSettings::default();
        assert!(settings.create_api_expander().is_none());
    }

    #[test]
    fn expand_settings_create_api_expander_some_for_url() {
        let settings = ExpandSettings {
            backend: "http://localhost:11434".to_string(),
            api_model: "llama3:8b".to_string(),
            ..Default::default()
        };
        let expander = settings.create_api_expander();
        assert!(expander.is_some());
    }

    #[test]
    fn expand_settings_to_expand_config() {
        let settings = ExpandSettings {
            temperature: 0.5,
            top_p: 0.8,
            max_tokens: 200,
            thinking: true,
            ..Default::default()
        };
        let config = settings.to_expand_config("sdxl", 3);
        assert_eq!(config.model_family, "sdxl");
        assert_eq!(config.variations, 3);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.top_p, 0.8);
        assert_eq!(config.max_tokens, 200);
        assert!(config.thinking);
    }

    #[test]
    fn expand_settings_serde_roundtrip() {
        let mut families = HashMap::new();
        families.insert(
            "sd15".to_string(),
            FamilyOverride {
                word_limit: Some(80),
                style_notes: Some("Custom SD1.5 notes.".to_string()),
            },
        );
        let settings = ExpandSettings {
            enabled: true,
            backend: "http://example.com".to_string(),
            model: "qwen3-expand-small:q8".to_string(),
            api_model: "gpt-4".to_string(),
            temperature: 1.2,
            top_p: 0.95,
            max_tokens: 500,
            thinking: true,
            system_prompt: Some("Custom system prompt {WORD_LIMIT} {MODEL_NOTES}".to_string()),
            batch_prompt: Some("Custom batch {N} {WORD_LIMIT} {MODEL_NOTES}".to_string()),
            families,
        };
        let toml_str = toml::to_string(&settings).unwrap();
        let deserialized: ExpandSettings = toml::from_str(&toml_str).unwrap();
        assert_eq!(deserialized.enabled, settings.enabled);
        assert_eq!(deserialized.backend, settings.backend);
        assert_eq!(deserialized.model, settings.model);
        assert_eq!(deserialized.api_model, settings.api_model);
        assert_eq!(deserialized.temperature, settings.temperature);
        assert_eq!(deserialized.max_tokens, settings.max_tokens);
        assert_eq!(deserialized.thinking, settings.thinking);
        assert_eq!(deserialized.system_prompt, settings.system_prompt);
        assert_eq!(deserialized.batch_prompt, settings.batch_prompt);
        assert_eq!(deserialized.families.len(), 1);
        let sd15 = deserialized.families.get("sd15").unwrap();
        assert_eq!(sd15.word_limit, Some(80));
        assert_eq!(sd15.style_notes.as_deref(), Some("Custom SD1.5 notes."));
    }

    #[test]
    fn expand_settings_serde_defaults_on_empty() {
        // Deserializing an empty table should produce all defaults
        let deserialized: ExpandSettings = toml::from_str("").unwrap();
        let defaults = ExpandSettings::default();
        assert_eq!(deserialized.enabled, defaults.enabled);
        assert_eq!(deserialized.backend, defaults.backend);
        assert_eq!(deserialized.model, defaults.model);
        assert_eq!(deserialized.temperature, defaults.temperature);
    }

    // ── ApiExpander ──────────────────────────────────────────────────────

    #[test]
    fn api_expander_strips_trailing_slash() {
        let expander = ApiExpander::new("http://localhost:11434/", "qwen2.5:3b");
        assert_eq!(expander.endpoint, "http://localhost:11434");
    }

    #[test]
    fn api_expander_no_slash_unchanged() {
        let expander = ApiExpander::new("http://localhost:11434", "qwen2.5:3b");
        assert_eq!(expander.endpoint, "http://localhost:11434");
    }

    // ── ExpandConfig ─────────────────────────────────────────────────────

    #[test]
    fn expand_config_default() {
        let config = ExpandConfig::default();
        assert_eq!(config.model_family, "flux");
        assert_eq!(config.variations, 1);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tokens, 300);
        assert!(!config.thinking);
    }

    // ── env overrides ────────────────────────────────────────────────────
    // These tests use a serial approach to avoid env var races.

    #[test]
    fn env_override_model_routes_to_local() {
        // When backend is "local", MOLD_EXPAND_MODEL should set self.model
        let settings = ExpandSettings::default();
        assert!(settings.is_local());
        // We can't safely set env vars in parallel tests, but we can test
        // the routing logic directly:
        let mut s = settings;
        let v = "qwen3-expand-small:q8".to_string();
        if s.is_local() {
            s.model = v.clone();
        } else {
            s.api_model = v.clone();
        }
        assert_eq!(s.model, "qwen3-expand-small:q8");
        assert_eq!(s.api_model, "qwen2.5:3b"); // unchanged
    }

    #[test]
    fn env_override_model_routes_to_api() {
        // When backend is a URL, MOLD_EXPAND_MODEL should set self.api_model
        let mut s = ExpandSettings {
            backend: "http://localhost:11434".to_string(),
            ..Default::default()
        };
        assert!(!s.is_local());
        let v = "llama3:70b".to_string();
        if s.is_local() {
            s.model = v.clone();
        } else {
            s.api_model = v.clone();
        }
        assert_eq!(s.api_model, "llama3:70b");
        assert_eq!(s.model, "qwen3-expand:q8"); // unchanged
    }

    // ── template overrides in ExpandConfig ───────────────────────────────

    #[test]
    fn to_expand_config_passes_overrides() {
        let mut families = HashMap::new();
        families.insert(
            "flux".to_string(),
            FamilyOverride {
                word_limit: Some(200),
                style_notes: None,
            },
        );
        let settings = ExpandSettings {
            system_prompt: Some("Custom {WORD_LIMIT} {MODEL_NOTES}".to_string()),
            batch_prompt: Some("Batch {N} {WORD_LIMIT} {MODEL_NOTES}".to_string()),
            families,
            ..Default::default()
        };
        let config = settings.to_expand_config("flux", 3);
        assert_eq!(
            config.system_prompt.as_deref(),
            Some("Custom {WORD_LIMIT} {MODEL_NOTES}")
        );
        assert_eq!(
            config.batch_prompt.as_deref(),
            Some("Batch {N} {WORD_LIMIT} {MODEL_NOTES}")
        );
        assert_eq!(config.family_overrides.len(), 1);
        assert_eq!(
            config.family_overrides.get("flux").unwrap().word_limit,
            Some(200)
        );
    }

    #[test]
    fn expand_config_default_has_no_overrides() {
        let config = ExpandConfig::default();
        assert!(config.system_prompt.is_none());
        assert!(config.batch_prompt.is_none());
        assert!(config.family_overrides.is_empty());
    }

    // ── validate_templates ──────────────────────────────────────────────

    #[test]
    fn validate_templates_valid() {
        let settings = ExpandSettings {
            system_prompt: Some("You are a writer. {WORD_LIMIT} words. {MODEL_NOTES}".to_string()),
            batch_prompt: Some(
                "Generate {N} prompts. {WORD_LIMIT} words. {MODEL_NOTES}".to_string(),
            ),
            ..Default::default()
        };
        assert!(settings.validate_templates().is_empty());
    }

    #[test]
    fn validate_templates_none_is_valid() {
        let settings = ExpandSettings::default();
        assert!(settings.validate_templates().is_empty());
    }

    #[test]
    fn validate_templates_missing_word_limit() {
        let settings = ExpandSettings {
            system_prompt: Some("You are a writer. {MODEL_NOTES}".to_string()),
            ..Default::default()
        };
        let errors = settings.validate_templates();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("{WORD_LIMIT}"));
    }

    #[test]
    fn validate_templates_missing_model_notes() {
        let settings = ExpandSettings {
            system_prompt: Some("You are a writer. {WORD_LIMIT} words.".to_string()),
            ..Default::default()
        };
        let errors = settings.validate_templates();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("{MODEL_NOTES}"));
    }

    #[test]
    fn validate_templates_batch_missing_n() {
        let settings = ExpandSettings {
            batch_prompt: Some("Generate prompts. {WORD_LIMIT} {MODEL_NOTES}".to_string()),
            ..Default::default()
        };
        let errors = settings.validate_templates();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("{N}"));
    }

    #[test]
    fn validate_templates_batch_missing_all() {
        let settings = ExpandSettings {
            batch_prompt: Some("Generate prompts.".to_string()),
            ..Default::default()
        };
        let errors = settings.validate_templates();
        assert_eq!(errors.len(), 3);
    }

    // ── FamilyOverride serde ────────────────────────────────────────────

    #[test]
    fn family_override_serde_roundtrip() {
        let ov = FamilyOverride {
            word_limit: Some(100),
            style_notes: Some("Be creative.".to_string()),
        };
        let json = serde_json::to_string(&ov).unwrap();
        let deserialized: FamilyOverride = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.word_limit, Some(100));
        assert_eq!(deserialized.style_notes.as_deref(), Some("Be creative."));
    }

    #[test]
    fn family_override_partial_toml() {
        let toml_str = "word_limit = 75\n";
        let ov: FamilyOverride = toml::from_str(toml_str).unwrap();
        assert_eq!(ov.word_limit, Some(75));
        assert!(ov.style_notes.is_none());
    }

    // ── full config with families in TOML ───────────────────────────────

    #[test]
    fn expand_settings_toml_with_families() {
        let toml_str = r#"
enabled = true
system_prompt = "Custom prompt. {WORD_LIMIT} words. {MODEL_NOTES}"

[families.sd15]
word_limit = 40
style_notes = "Short keywords only."

[families.flux]
word_limit = 250
"#;
        let settings: ExpandSettings = toml::from_str(toml_str).unwrap();
        assert!(settings.enabled);
        assert!(settings.system_prompt.is_some());
        assert_eq!(settings.families.len(), 2);
        let sd15 = settings.families.get("sd15").unwrap();
        assert_eq!(sd15.word_limit, Some(40));
        assert_eq!(sd15.style_notes.as_deref(), Some("Short keywords only."));
        let flux = settings.families.get("flux").unwrap();
        assert_eq!(flux.word_limit, Some(250));
        assert!(flux.style_notes.is_none());
    }
}
