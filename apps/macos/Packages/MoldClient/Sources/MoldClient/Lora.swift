import Foundation

/// An installed adapter as `GET /api/loras?model=<name>` reports it.
///
/// The server filters this list to the chosen model's family already
/// (`catalog_api.rs:1098-1115`), so the app never matches families itself --
/// see `RecipeCapabilities.loraStack`, which gates whether to ask at all.
/// `types.rs:2707-2733`.
public struct LoraInfo: Codable, Hashable, Sendable, Identifiable {
    /// Catalog id, e.g. `"cv:2486534"`.
    public let id: String
    public let name: String
    public let family: String
    public let author: String?
    /// The absolute server-side path that goes in `loras[].path` verbatim.
    /// It may contain spaces; it is a JSON string value, never a URL
    /// component.
    public let path: String
    public let trainedWords: [String]
    public let sizeBytes: Int?
    public let thumbnailUrl: String?
    public let addedAt: Int
}

/// One adapter in a draft's stack.
///
/// `types.rs:2648-2667`'s `LoraWeight` carries only `path`, `scale` and an
/// optional `expert` -- no name. `name` here is DISPLAY ONLY and is never
/// encoded onto the wire (`GenerateRequest+Encoding.swift` builds its own
/// path/scale-only representation).
public struct LoraChoice: Codable, Hashable, Sendable, Identifiable {
    public var path: String
    /// `validation.rs:1582-1588`: `[0.0, 2.0]`.
    public var scale: Double
    public var name: String
    public var id: String { path }

    public init(path: String, scale: Double = Lora.defaultScale, name: String) {
        self.path = path
        self.scale = scale
        self.name = name
    }
}

/// LoRA adapter constants shared by the reading extensions and the draft.
public enum Lora {
    /// `validation.rs:1582-1588`.
    public static let scaleRange: ClosedRange<Double> = 0 ... 2
    public static let defaultScale = 1.0
    /// `generation_profile.rs:2316` -- every recipe but LTX-2's `ic-lora`
    /// pipeline gets four slots; `ic-lora` itself advertises its own
    /// narrower `lora.max_count` via the block, so this is only the
    /// no-block fallback (`RecipeCapabilities.loraStack`).
    public static let defaultMaxStack = 4
}
