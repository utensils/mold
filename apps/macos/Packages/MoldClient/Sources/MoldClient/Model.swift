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
    /// Bytes this model occupies on the machine, present only when
    /// `downloaded` (`catalog.rs:220-225`). NEVER SUM THIS COLUMN: a shared
    /// VAE or encoder is counted once per model that references it -- the
    /// machine's own figure is `/api/status.models_disk`.
    public let diskUsageBytes: Int?
    /// Catalog classification for a catalog-installed model. Absent for
    /// manifest rows, whose `family` is sufficient, and for older servers.
    public let kind: String?
    /// Catalog modality (`image` / `video`). Same absence rule as `kind`.
    public let modality: String?
    /// Mature-content classification. `nil` means UNKNOWN, never safe.
    public let nsfw: Bool?
    /// Whether this build can execute this model. `nil` on servers that
    /// predate the field: read as "runnable" (`runtimeAvailable != false`).
    public let runtimeAvailable: Bool?
    /// One sentence naming why `runtimeAvailable` is false.
    public let runtimeUnavailableReason: String?

    public var id: String { name }
}

public extension Model {
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

    /// Prefixes that make the colon a NAMESPACE rather than a variant tag.
    ///
    /// `flux-dev:q4` is one model at one quantization, but `cv:252914` is a
    /// Civitai id and `hf:owner/repo` a Hugging Face one. Splitting those the
    /// same way gives every catalog model the base name "cv", which collapses
    /// unrelated checkpoints into one group -- Juggernaut XL filed under
    /// DreamShaper.
    static let catalogPrefixes: Set<String> = ["cv", "hf"]

    private var colonSplit: (head: String, tail: String)? {
        let parts = name.split(separator: ":", maxSplits: 1)
        guard parts.count == 2 else { return nil }
        return (String(parts[0]), String(parts[1]))
    }

    /// `flux-dev` out of `flux-dev:q4`. Variants of one model share this.
    var baseName: String {
        guard let split = colonSplit, !Self.catalogPrefixes.contains(split.head) else {
            return name
        }
        return split.head
    }

    /// `q4` out of `flux-dev:q4`. nil for an untagged model, and nil for a
    /// catalog id, whose suffix identifies the model rather than a variant.
    var tag: String? {
        guard let split = colonSplit, !Self.catalogPrefixes.contains(split.head) else {
            return nil
        }
        return split.tail
    }

    /// The headline with trailing variant words dropped, so a group heading
    /// names the model and each row underneath names the variant.
    ///
    /// "FLUX.1 Dev Q4" becomes "FLUX.1 Dev", and "FLUX.2 [dev] Q4 GGUF"
    /// becomes "FLUX.2 [dev]" -- more than one word can be variant noise, so
    /// they come off until a real word is reached.
    var baseTitle: String {
        var words = headline.split(separator: " ").map(String.init)
        while words.count > 1, Self.isVariantWord(words[words.count - 1]) {
            words.removeLast()
        }
        return words.joined(separator: " ")
    }

    static func isVariantWord(_ word: String) -> Bool {
        let lower = word.lowercased()
        // A container format says nothing about which model this is.
        if lower == "gguf" || lower == "safetensors" { return true }
        // Quantizations carry a digit: Q4, Q8, BF16, FP8. A real trailing
        // word like "Dev", "Turbo" or "Schnell" does not.
        return lower.range(of: "^(q|fp|bf|int|nvfp)[0-9]", options: .regularExpression) != nil
    }
}
