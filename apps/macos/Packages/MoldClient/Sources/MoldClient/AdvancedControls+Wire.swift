import Foundation

// What the sampler controls put on the request, and what they refuse to.
// Split from the declaration purely for size.
//
// The two rules are studio's, verbatim: an untouched control must not appear
// on the wire, and a touched control must reach the server OR SAY WHY IT
// CANNOT (`guidanceOverrides.ts:9-15`). So a value the wire cannot carry --
// an unparsable block list, a fractional skip stride -- contributes nothing
// HERE, and `refusal` is what the pane shows before anything is submitted.
public extension AdvancedControls {
    /// Ceilings mirrored from `mold-core`'s validation. The server re-checks
    /// the block indices against the resolved checkpoint's real depth
    /// (`guidanceOverrides.ts:38-47`, `validation.rs:1728-1795`).
    static let maxStgBlockIndex = 64
    static let maxStgBlocks = 8
    static let maxGuidanceScale: Double = 10
    static let maxGuidanceSkipStep = 8
    /// `(0, 4]` -- `wanRecipe.ts:35-37`.
    static let maxWanDistillStrength: Double = 4

    /// The block list, or `nil` when the text is empty or unusable. Port of
    /// `parseStgBlocks` (`guidanceOverrides.ts:154-162`).
    var parsedStgBlocks: [Int]? {
        guard stgBlocksRefusal == nil else { return nil }
        let blocks = stgBlocks.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .compactMap(Int.init)
        return blocks.isEmpty ? nil : blocks
    }

    /// Port of `stgBlocksError` (`guidanceOverrides.ts:82-106`). The sentences
    /// are studio's, because one mold says one thing.
    var stgBlocksRefusal: String? {
        let trimmed = stgBlocks.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return nil }
        let entries = trimmed.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        if entries.isEmpty { return "List at least one block index." }
        if entries.count > Self.maxStgBlocks { return "At most \(Self.maxStgBlocks) blocks." }
        var seen: Set<Int> = []
        for entry in entries {
            guard entry.allSatisfy(\.isNumber), let block = Int(entry) else {
                return "\"\(entry)\" is not a block index."
            }
            if block >= Self.maxStgBlockIndex {
                return "Block \(block) is deeper than any LTX-2 checkpoint."
            }
            if !seen.insert(block).inserted { return "Block \(block) is listed twice." }
        }
        return nil
    }

    /// One sentence covering every sampler control, for the pane's submit
    /// gate. Port of `guidanceOverridesError` (`:139-152`) and `wanRecipeError`
    /// (`wanRecipe.ts:90-99`) -- the bands mirror `mold-core`'s validation so
    /// a request that would come back 422 is caught before the round trip,
    /// and the server stays the authority.
    var refusal: String? {
        if let blocks = stgBlocksRefusal { return "STG blocks: \(blocks)" }
        if let skip = Self.skipStepRefusal(skipStep) { return "Guidance skip stride: \(skip)" }
        if let stg = Self.scaleRefusal(stgScale, "STG scale", Self.maxGuidanceScale) { return stg }
        if let rescale = Self.scaleRefusal(rescaleScale, "CFG rescale", 1) { return rescale }
        if let modality = Self.scaleRefusal(modalityScale, "Modality scale", Self.maxGuidanceScale) {
            return modality
        }
        if let shift = Self.sampleShiftRefusal(sampleShift) { return shift }
        if let high = Self.distillRefusal(distillStrengthHigh, "High-noise") { return high }
        return Self.distillRefusal(distillStrengthLow, "Low-noise")
    }

    /// `skip_step` is a `u32`: a fractional value fails JSON deserialization
    /// outright, so the request would come back as an opaque body-parse
    /// failure instead of a field-named 422 (`guidanceOverrides.ts:108-121`).
    static func skipStepRefusal(_ value: Int?) -> String? {
        guard let value else { return nil }
        guard (0 ... maxGuidanceSkipStep).contains(value) else {
            return "Enter a stride between 0 and \(maxGuidanceSkipStep)."
        }
        return nil
    }

    static func scaleRefusal(_ value: Double?, _ label: String, _ max: Double) -> String? {
        guard let value else { return nil }
        guard value.isFinite else { return "\(label) must be a number." }
        guard value >= 0, value <= max else {
            return "\(label) must be between 0 and \(Int(max))."
        }
        return nil
    }

    static func sampleShiftRefusal(_ value: Double?) -> String? {
        guard let value else { return nil }
        guard value.isFinite, value > 0 else { return "Flow shift must be a positive number." }
        return nil
    }

    static func distillRefusal(_ value: Double?, _ label: String) -> String? {
        guard let value else { return nil }
        guard value.isFinite, value > 0 else {
            return "\(label) distill strength must be greater than 0."
        }
        guard value <= maxWanDistillStrength else {
            return "\(label) distill strength must be at most \(Int(maxWanDistillStrength))."
        }
        return nil
    }
}
