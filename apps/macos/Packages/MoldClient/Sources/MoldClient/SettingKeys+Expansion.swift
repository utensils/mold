import Foundation

// Settings ▸ Expansion (design "The pane map"): the eight `expand.*` keys,
// `config_keys.rs:134-176`. Bounds are the engine's -- `expand.max_tokens`
// tops out at 65535 (`:170`), not the studio's narrower client cap.
public extension SettingKeys {
    static let expansion: [SettingKey] = [
        SettingKey(
            key: "expand.enabled", label: "Expand prompts",
            help: "Offer a rewritten prompt before a render starts.",
            editor: .toggle),
        SettingKey(
            key: "expand.backend", label: "Backend",
            // config_keys.rs:677: parse_string, no validate_enum -- offering
            // a fixed choice where the engine accepts any string (a local
            // model name or an API base URL) would invent a constraint.
            help: "\"local\" for the machine's own model, or an API base URL.",
            editor: .text),
        SettingKey(
            key: "expand.model", label: "Local model",
            help: "The local model name expansion loads when the backend is local.",
            editor: .text),
        SettingKey(
            key: "expand.api_model", label: "API model",
            help: "The model name sent to an API backend.",
            editor: .text),
        SettingKey(
            key: "expand.temperature", label: "Temperature",
            // config_keys.rs:680: parse_f64(raw, 0.0, 2.0, key)
            help: "Higher values vary the rewrite more.",
            editor: .number(min: 0.0, max: 2.0, step: 0.1)),
        SettingKey(
            key: "expand.top_p", label: "Top P",
            // config_keys.rs:681: parse_f64(raw, 0.0, 1.0, key)
            help: "Narrows which words the rewrite can choose from.",
            editor: .number(min: 0.0, max: 1.0, step: 0.05)),
        SettingKey(
            key: "expand.max_tokens", label: "Max tokens",
            // config_keys.rs:682: parse_u32(raw, 1, 65535, key)
            help: "The longest rewrite expansion will return.",
            editor: .number(min: 1, max: 65535, step: 1)),
        SettingKey(
            key: "expand.thinking", label: "Thinking",
            help: "Let the expansion model reason before it answers, where it supports that.",
            editor: .toggle),
    ]
}
