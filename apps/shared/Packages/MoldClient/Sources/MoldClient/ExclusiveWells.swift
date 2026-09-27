import Foundation

/// One of the two wells a `singleOrReferences` recipe draws.
public enum ExclusiveWell: String, Hashable, Sendable, Codable {
    case source
    case references
}

/// Which well ships and which one waits, for the EXCLUSIVE relation.
///
/// Port of `studio/lib/sourceMediaPlan.ts:163-197` (`resolveExclusiveWells`):
/// LAST WRITE WINS, and the parked media is KEPT rather than discarded.
/// Attaching to either well parks the other -- it does not refuse the drop
/// and it does not clear the earlier picture, so removing the active media
/// restores the parked well exactly as it was. Generate stays enabled
/// throughout: only the active well's media reaches the wire, which is what
/// makes parking safe.
public struct ExclusiveWells: Hashable, Sendable {
    /// The well whose media ships; `nil` while neither holds any.
    public let active: ExclusiveWell?
    /// The well that is parked, or `nil` while nothing is active.
    public let parked: ExclusiveWell?

    /// The inline note the parked well renders. Studio's own sentence,
    /// verbatim (`EXCLUSIVE_WELLS_NOTE`), so one checkpoint reads the same
    /// on every surface.
    public static let note =
        "This model renders from a source image OR reference images, not both "
        + "— remove one to use the other."

    public static func resolve(
        hasSource: Bool, referenceCount: Int, lastWrite: ExclusiveWell?
    ) -> ExclusiveWells {
        let hasReferences = referenceCount > 0
        guard hasSource || hasReferences else {
            return ExclusiveWells(active: nil, parked: nil)
        }
        let active: ExclusiveWell
        if hasSource, hasReferences {
            // Both hold media: the last write decides, and an unmarked
            // restore reads as the source well.
            active = lastWrite == .references ? .references : .source
        } else {
            active = hasSource ? .source : .references
        }
        return ExclusiveWells(active: active, parked: active == .source ? .references : .source)
    }
}

/// WHICH conditioning a request built in a given mode carries -- the one
/// decision behind the request builder, the strength control and the mask row.
///
/// Port of `studio/lib/sourceMediaPlan.ts:217-241` (`conditioningForRequest`).
/// It exists because the exclusive relation breaks the old shorthand
/// ("references if there are any, else the source"): Klein's request is one or
/// the other depending on what was attached, and a builder that emitted both
/// is refused at admission. `both` is the ADDITIVE answer, and the reason this
/// is a union rather than one well -- an IP-Adapter render carries
/// `source_image` (with its strength and mask) AND `edit_images` in the same
/// request.
public enum RequestConditioning: String, Hashable, Sendable {
    case source
    case references
    case both
    case none

    /// Whether the request carries `source_image`, and so its strength and mask.
    public var carriesSource: Bool { self == .source || self == .both }
    /// Whether the request carries `edit_images`.
    public var carriesReferences: Bool { self == .references || self == .both }

    public static func resolve(
        mode: SourceImageMode, hasSource: Bool, referenceCount: Int,
        lastWrite: ExclusiveWell?
    ) -> RequestConditioning {
        let hasReferences = referenceCount > 0
        switch mode {
        case .single:
            return hasSource ? .source : .none
        case .references, .qwenEdit:
            return hasReferences ? .references : .none
        case .singleOrReferences:
            let wells = ExclusiveWells.resolve(
                hasSource: hasSource, referenceCount: referenceCount, lastWrite: lastWrite)
            return wells.active.map { $0 == .source ? .source : .references } ?? .none
        case .singleAndReferences:
            // Nothing parks, so the answer is simply what each well holds.
            if hasSource, hasReferences { return .both }
            if hasSource { return .source }
            return hasReferences ? .references : .none
        }
    }
}
