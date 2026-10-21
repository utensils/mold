import Foundation

/// Scalar source identity survives together with the ordinary draft. It cannot
/// authorize a different host just because that host has the same filename.
public struct VideoLibrarySourceOwner: Codable, Hashable, Sendable {
    public let hostId: UUID
    public let origin: String
    public let instanceId: String
    public let connectionMode: HostConnectionMode
    public let title: String?
    public init(owner: VideoJobOwner, title: String?) {
        hostId = owner.host.id; origin = owner.host.baseURL.absoluteString
        instanceId = owner.instanceId; connectionMode = owner.host.connectionMode
        self.title = title
    }
    public func matches(_ owner: VideoJobOwner) -> Bool {
        hostId == owner.host.id && origin == owner.host.baseURL.absoluteString
            && instanceId == owner.instanceId && connectionMode == owner.host.connectionMode
    }
}

/// Kept in DraftDescriptor's existing private atomic file, not preferences.
/// No media bytes, upload grants or credentials belong in this authoring state.
public struct VideoDurationAuthoring: Codable, Hashable, Sendable {
    public var draft: VideoDurationDraft
    public var sourceOwner: VideoLibrarySourceOwner?
    public var reuseNotice: String?
    public init(draft: VideoDurationDraft, sourceOwner: VideoLibrarySourceOwner? = nil, reuseNotice: String? = nil) {
        self.draft = draft; self.sourceOwner = sourceOwner; self.reuseNotice = reuseNotice
    }
    public func request(generation: GenerateRequest, owner: VideoJobOwner) throws -> VideoJobRequest {
        if draft.source != nil, sourceOwner?.matches(owner) != true {
            throw SequenceDraftError.invalid("Reconnect to this video's original machine, or choose Start new video.")
        }
        return try draft.request(generation: generation)
    }
    /// Only an affirmative current plan can hide an unused opening control.
    public func usesSourceEnding(plan: VideoJobPlan?) -> Bool {
        draft.source != nil && plan?.joinMode == .continuous
    }
    public func openingConflict(plan: VideoJobPlan?, generation: GenerateRequest) -> String? {
        guard usesSourceEnding(plan: plan), generation.sourceImage != nil
                || generation.keyframes?.contains(where: { $0.frame == 0 }) == true else { return nil }
        return "This video continues the original ending. Remove the attached first frame, or choose Start new video."
    }
    public mutating func startFresh() { draft.source = nil; sourceOwner = nil; reuseNotice = nil }
}
