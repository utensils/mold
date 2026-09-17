import Foundation

// Settings ▸ Generation (design "The pane map"): render defaults every model
// starts from unless "Use as default for this model" (M3) overrides them,
// plus what happens to a saved picture. Bounds are the ENGINE's
// (`crates/mold-core/src/config_keys.rs`), not the studio's narrower client
// caps -- a value set from the CLI inside the engine's range must not read
// as out of range here. `models_dir` and `output_dir` get no row (design
// decision 18: one is refused outright by a live server, the other doesn't
// take effect until restart); `umt5_variant` gets no row either -- the
// engine registers it but never lists it (design fact 2, pinned by
// `aKeyTheEngineNeverListsIsNeverCurated`).
public extension SettingKeys {
    static let generationRendering: [SettingKey] = [
        SettingKey(
            key: "default_model", label: "Model to start with",
            help: "The model a new render opens with, until you pick another.",
            editor: .text),
        SettingKey(
            key: "default_width", label: "Width",
            // config_keys.rs:652: parse_u32(raw, 64, 8192, key)
            help: "Default width for a new render, in pixels.",
            editor: .number(min: 64, max: 8192, step: 64)),
        SettingKey(
            key: "default_height", label: "Height",
            // config_keys.rs:653: parse_u32(raw, 64, 8192, key)
            help: "Default height for a new render, in pixels.",
            editor: .number(min: 64, max: 8192, step: 64)),
        SettingKey(
            key: "default_steps", label: "Steps",
            // config_keys.rs:654: parse_u32(raw, 1, 1000, key)
            help: "Default number of steps for a new render.",
            editor: .number(min: 1, max: 1000, step: 1)),
        SettingKey(
            key: "default_negative_prompt", label: "Negative prompt",
            help: "Applied to a new render unless you type your own.",
            editor: .text),
        SettingKey(
            key: "t5_variant", label: "T5 precision",
            // config_keys.rs:659: validate_enum's own list
            help: "Which T5 text-encoder weights FLUX and SD3 models load.",
            editor: .choice(["auto", "fp16", "q8", "q6", "q5", "q4", "q3"])),
        SettingKey(
            key: "qwen3_variant", label: "Qwen3 precision",
            // config_keys.rs:668: validate_enum's own list
            help: "Which Qwen3 text-encoder weights Qwen-Image models load.",
            editor: .choice(["auto", "bf16", "q8", "q6", "iq4", "q3"])),
    ]

    /// "When a picture is saved" -- `embed_metadata` (`config_keys.rs:103`)
    /// and `generate.auto_tag_title` (`GENERATE_AUTO_TAG_TITLE_KEY`,
    /// `config_keys.rs:49`), a CLIENT-side default (it shapes what a request
    /// carries, not how the server behaves) rather than a server toggle.
    static let generationSaving: [SettingKey] = [
        SettingKey(
            key: "embed_metadata", label: "Embed metadata in saved files",
            help: "Writes the prompt and settings into every saved image or video.",
            editor: .toggle),
        SettingKey(
            key: "generate.auto_tag_title", label: "Tag a titled print with its title",
            help: "When you title a print at create time, also add its title as a tag.",
            editor: .toggle),
    ]

    static var generation: [SettingKey] { generationRendering + generationSaving }
}
