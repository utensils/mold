import Foundation

// The two sentences a one-shot auto-chain earns when nothing can cross its
// seam, and the single-clip ceiling that follows from the first of them.
//
// Byte-identical mirrors of `mold_core::chain::text_only_auto_chain_refusal`
// (`chain.rs:765-800`), which is the single authority: the CLI router calls
// it and the server renders it at `POST /api/chain-jobs` for the same
// ephemeral job this app submits, so a person meets ONE sentence whichever
// door they came through. `tests/fixtures/wan/surface-parity-v1.json` pins the
// template and every surface reads it.
public extension ChainRouting {
    static func textOnlyRefusal(
        family: String?, model: String, sourceImage: SourceImageCapability?,
        totalFrames: Int, clipFrames: Int
    ) -> String? {
        guard totalFrames > clipFrames else { return nil }
        switch canonical(family) {
        case "wan" where sourceImage == .unsupported:
            return "'\(model)' is text-to-video and cannot continue motion across a clip "
                + "boundary, so rendering \(totalFrames) frames would repeat the same "
                + "~\(clipFrames)-frame clip rather than extend it. Reduce the frame count "
                + "to \(clipFrames) or fewer for one continuous clip, or use an "
                + "image-to-video tier (wan22-i2v-a14b, wan22-ti2v-5b:turbo), which "
                + "seeds each continuation with the previous clip's final frame."
        case "ltx-video":
            return "'\(model)' is legacy LTX-Video and cannot continue motion across a clip "
                + "boundary, so rendering \(totalFrames) frames would repeat the same "
                + "~\(clipFrames)-frame clip rather than extend it. Reduce the frame count "
                + "to \(clipFrames) or fewer for one continuous clip, or use LTX-2.3 or "
                + "LTX-2.5, which carries context into each continuation."
        default:
            return nil
        }
    }
}

/// A request field the automatic split cannot carry without silently changing
/// what was asked for. Canonical authored sequences support a wider per-clip
/// schema; `mold run --script` is where that lives
/// (`chainRouting.ts:72-101`).
public enum AutoChainField: String, Hashable, Sendable, CaseIterable {
    case negativePrompt = "negative prompt"
    case loras = "LoRAs or camera motion"
    case audioFile = "conditioning audio"
    case sourceVideo = "source video"
    case extendVideo = "video continuation"
    case keyframes
    case pipeline = "pipeline selection"
    case retakeRange = "retake range"
    case guidanceOverrides = "guidance overrides"

    /// The label this reads as in a sentence. It IS the raw value for all but
    /// one, which keeps the table honest -- a case added without a label
    /// would read as its own name rather than silently as something else.
    public var label: String { rawValue }
}

public extension AutoChainField {
    /// "a, b, and c" -- port of `autoChainFieldList` (`chainRouting.ts:137-144`).
    static func list(_ fields: [AutoChainField]) -> String {
        let labels = fields.map(\.label)
        if labels.count <= 1 { return labels.first ?? "selected options" }
        if labels.count == 2 { return "\(labels[0]) and \(labels[1])" }
        return labels.dropLast().joined(separator: ", ") + ", and " + (labels.last ?? "")
    }

    /// What THIS request carries that an automatic split would drop. Read off
    /// the built request rather than the draft, so a field parked by the
    /// recipe cannot count against a chain it is not even in.
    static func unsupported(in request: GenerateRequest) -> [AutoChainField] {
        var fields: [AutoChainField] = []
        if request.negativePrompt?.trimmingCharacters(in: .whitespaces).isEmpty == false {
            fields.append(.negativePrompt)
        }
        if request.loras?.isEmpty == false { fields.append(.loras) }
        if request.audioFile != nil { fields.append(.audioFile) }
        if request.sourceVideo != nil { fields.append(.sourceVideo) }
        if request.extendVideo != nil || request.extendOverlapFrames != nil {
            fields.append(.extendVideo)
        }
        if request.keyframes?.isEmpty == false { fields.append(.keyframes) }
        if request.pipeline != nil { fields.append(.pipeline) }
        if request.guidanceOverrides != nil { fields.append(.guidanceOverrides) }
        return fields
    }
}
