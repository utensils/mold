import Foundation

/// What a host can do, as it reports it on `GET /api/capabilities`.
///
/// This is the authority the app reads instead of probing routes for 404s or
/// matching on model names. Absence has a DIFFERENT meaning per field and is
/// never flattened to "no" -- each optional below says which it is.
public struct Capabilities: Codable, Hashable, Sendable {
    public let generationProfileV1: Bool?
    public let gallery: GalleryCapabilities?
    public let queue: QueueCapabilities?
    public let events: EventsCapabilities?
    public let licenses: Bool?

    /// Absence means an older host that predates the field, so the app falls
    /// back to unconditional listing rather than assuming it is unsupported.
    public var supportsConditionalGallery: Bool { gallery?.conditionalGet ?? false }

    /// Row-level gallery events let the app update one tile instead of
    /// re-listing 1,500.
    public var supportsGalleryRowEvents: Bool { gallery?.rowEvents ?? false }

    /// The presence of this number is how a client knows the host generates at
    /// all -- not a separate boolean.
    public var generates: Bool { queue?.heterogeneousBatchMaxOutputs != nil }

    public var maxBatchOutputs: Int { queue?.heterogeneousBatchMaxOutputs ?? 1 }

    /// Durable work survives a dropped connection. Where this is true, a job
    /// whose stream died is still running and must NOT be dead-lettered.
    public var hasDurableQueue: Bool { queue?.durableQueue ?? false }
}

public struct GalleryCapabilities: Codable, Hashable, Sendable {
    public let canDelete: Bool?
    public let organize: Bool?
    public let bulkMutations: Bool?
    public let mediaVersion: Bool?
    public let conditionalGet: Bool?
    public let rowEvents: Bool?
    public let persistsOutputs: Bool?
    public let trash: TrashCapabilities?
}

public struct TrashCapabilities: Codable, Hashable, Sendable {
    public let enabled: Bool
    /// 0 means keep forever.
    public let retentionDays: Int?
}

public struct QueueCapabilities: Codable, Hashable, Sendable {
    public let canPause: Bool?
    public let canCancelAll: Bool?
    public let canReorder: Bool?
    public let durableQueue: Bool?
    public let heterogeneousBatchMaxOutputs: Int?
}

public struct EventsCapabilities: Codable, Hashable, Sendable {
    public let available: Bool?
}
