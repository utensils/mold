import Foundation

/// What one machine has been told a model's controls should start at.
///
/// These are the eight `models.<name>.<field>` keys, and they are the whole
/// of what a host can persist per model over HTTP -- `model_prefs`, the DB
/// table whose module doc describes exactly this feature, has no route on any
/// mold and no production writer. Batch size, output format, the upscaler and
/// Save to library have no key here, which is why the menu item names the
/// fields it saved instead of saying "these settings".
public struct ModelDefaults: Hashable, Sendable {
    public var steps: Int?
    public var guidance: Double?
    public var width: Int?
    public var height: Int?
    public var scheduler: String?
    public var negativePrompt: String?
    public var lora: String?
    public var loraScale: Double?

    public init(
        steps: Int? = nil, guidance: Double? = nil, width: Int? = nil, height: Int? = nil,
        scheduler: String? = nil, negativePrompt: String? = nil, lora: String? = nil,
        loraScale: Double? = nil
    ) {
        self.steps = steps
        self.guidance = guidance
        self.width = width
        self.height = height
        self.scheduler = scheduler
        self.negativePrompt = negativePrompt
        self.lora = lora
        self.loraScale = loraScale
    }

    public var isEmpty: Bool {
        steps == nil && guidance == nil && width == nil && height == nil
            && scheduler == nil && negativePrompt == nil && lora == nil && loraScale == nil
    }

    /// The order `MODEL_FIELDS` declares them in
    /// (`crates/mold-core/src/config_keys.rs:359-368`).
    private static let fields = [
        "default_steps", "default_guidance", "default_width", "default_height",
        "scheduler", "negative_prompt", "lora", "lora_scale",
    ]

    /// The eight keys for one model, in registry order.
    public static func keys(for model: String) -> [String] {
        fields.map { "models.\(model).\($0)" }
    }

    /// Reads a whole listing. An absent key means this machine has never been
    /// told anything about this model; a PRESENT key with a null value means
    /// the same thing and is what a configured-but-unset model looks like --
    /// all sixteen `models.*` rows on workstation are exactly that.
    public init(from listing: ConfigListing, model: String) {
        let prefix = "models.\(model)."
        var byField: [String: ConfigScalar] = [:]
        for entry in listing.entries where entry.key.hasPrefix(prefix) {
            byField[String(entry.key.dropFirst(prefix.count))] = entry.value
        }
        steps = byField["default_steps"]?.int
        guidance = byField["default_guidance"]?.double
        width = byField["default_width"]?.int
        height = byField["default_height"]?.int
        scheduler = byField["scheduler"]?.text
        negativePrompt = byField["negative_prompt"]?.text
        lora = byField["lora"]?.text
        loraScale = byField["lora_scale"]?.double
    }

    /// What to PUT for a draft, as `(key, ConfigScalar)` pairs.
    ///
    /// Only the four numbers and the negative prompt: the app has no
    /// scheduler control and no LoRA control until M4, and writing a key the
    /// user cannot see would silently clear a default set from the CLI. An
    /// empty negative prompt writes `null` -- an explicit clear, not the
    /// string `""`.
    public func writes(for draft: RenderDraft, model: String) -> [(String, ConfigScalar)] {
        let prefix = "models.\(model)."
        return [
            (prefix + "default_steps", .number(Double(draft.steps))),
            (prefix + "default_guidance", .number(draft.guidance)),
            (prefix + "default_width", .number(Double(draft.width))),
            (prefix + "default_height", .number(Double(draft.height))),
            (prefix + "negative_prompt",
             draft.negativePrompt.isEmpty ? .null : .string(draft.negativePrompt)),
        ]
    }
}
