import Foundation

/// A model as a host reports it on `GET /api/models`.
///
/// Note the two flattened groups on the Rust side (`ModelInfo` and
/// `ModelDefaults`): their fields arrive at the TOP level of each object, not
/// nested, which is why they are plain properties here.
public struct Model: Codable, Hashable, Sendable, Identifiable {
    /// The request id -- `flux-dev:q4`. This is what goes in a request, and it
    /// is the identity; `displayName` is only a label.
    public let name: String
    public let family: String
    /// Written by the manifest as "Title — plain-English trade-off".
    public let description: String
    public let sizeGb: Double?
    public let isLoaded: Bool?
    public let downloaded: Bool?
    public let hfRepo: String?
    public let displayName: String?
    /// Non-zero means partially installed and needing repair -- which is a
    /// different state from "not installed", and says so in the UI.
    public let remainingDownloadBytes: Int?
    public let generationProfile: GenerationProfileSet?

    public var id: String { name }
}

public extension Model {
    /// Families that are not standalone picture-makers.
    ///
    /// These mirror `UTILITY_FAMILIES`, `UPSCALER_FAMILIES` and
    /// `AUXILIARY_FAMILIES` in `crates/mold-core/src/manifest.rs`, and
    /// `ModelFamilyContractTests` fails if they drift. A prompt-expansion LLM
    /// or a ControlNet in a model picker is a bug, not a listing.
    static let utilityFamilies: Set<String> = ["qwen3-expand", "companion"]
    static let upscalerFamilies: Set<String> = ["upscaler"]
    static let auxiliaryFamilies: Set<String> = [
        "controlnet", "ltx2-control", "ltx2-camera-control",
        "pulid", "ip-adapter", "hunyuan3d-paint",
    ]

    var isUtility: Bool { Self.utilityFamilies.contains(family) }
    var isUpscaler: Bool { Self.upscalerFamilies.contains(family) }
    var isAuxiliary: Bool { Self.auxiliaryFamilies.contains(family) }

    /// True when a person picking "what should make this picture" should see it.
    var isGenerator: Bool { !isUtility && !isUpscaler && !isAuxiliary }

    /// The part before the em-dash: "FLUX.1 Dev Q4".
    ///
    /// The manifest already writes every description this way, so the app
    /// splits rather than inventing copy of its own.
    var headline: String {
        guard let range = description.range(of: " — ") else {
            return displayName ?? description
        }
        return String(description[..<range.lowerBound])
    }

    /// The part after the em-dash: "smaller/faster, good quality". This is the
    /// sentence that tells someone what the model is FOR.
    var tradeOff: String? {
        guard let range = description.range(of: " — ") else { return nil }
        return String(description[range.upperBound...])
    }

    /// Bytes still to fetch before this model can run, if any.
    var repairBytes: Int? {
        guard let remaining = remainingDownloadBytes, remaining > 0 else { return nil }
        return remaining
    }

    /// Installed and complete.
    var isReady: Bool { (downloaded ?? false) && repairBytes == nil }

    /// The recipe a plain request against this model will run.
    var defaultRecipe: GenerationRecipe? { generationProfile?.defaultRecipe }
}
