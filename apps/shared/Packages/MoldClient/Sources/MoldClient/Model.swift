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
    /// The manifest's own sentence, or a catalog install's provider title.
    /// Never parsed for a name: `presentation` is the naming authority.
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
    /// A checkpoint-level veto for audio generation. `false` means this row
    /// lacks the audio assets even when its family recipe supports sound;
    /// absent on older hosts and models that inherit the recipe answer.
    public let supportsAudio: Bool?
    public var supportsDurationPrediction: Bool? = nil
    public var runtimeReady: Bool? = nil
    public var offersDurationPrediction: Bool { supportsDurationPrediction == true && runtimeReady != false }
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
    /// `mold_core::model_presentation`'s four fields, as `/api/models` sends
    /// them (`title`, `variant`, `summary`, `family_label`). Absent on catalog
    /// installs and older servers. Read them through `presentation`.
    public var presentedTitle: String? = nil
    public var presentedVariant: String? = nil
    public var presentedSummary: String? = nil
    public var presentedFamilyLabel: String? = nil

    public var id: String { name }

    // The wire names arrive camel-cased by `MoldJSON`'s snake-case strategy,
    // so `family_label` matches `familyLabel` here.
    enum CodingKeys: String, CodingKey {
        case name, family, description, sizeGb, isLoaded, downloaded, hfRepo, displayName
        case remainingDownloadBytes, generationProfile, supportsAudio
        case supportsDurationPrediction, runtimeReady, diskUsageBytes, kind, modality, nsfw
        case runtimeAvailable, runtimeUnavailableReason
        case presentedTitle = "title"
        case presentedVariant = "variant"
        case presentedSummary = "summary"
        case presentedFamilyLabel = "familyLabel"
    }
}

public extension Model {
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
}
