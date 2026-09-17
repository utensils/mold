import Foundation

/// What a host can do, as it reports it on `GET /api/capabilities`.
///
/// This is the authority the app reads instead of probing routes for 404s or
/// matching on model names. Absence has a DIFFERENT meaning per field and is
/// never flattened to "no" -- `Capabilities+Reading` is where each one says
/// which, and that file is the only thing the app should ask.
public struct Capabilities: Codable, Hashable, Sendable {
    public let generationProfileV1: Bool?
    public let gallery: GalleryCapabilities?
    public let queue: QueueCapabilities?
    public let events: EventsCapabilities?
    public let licenses: Bool?
    public let catalog: CatalogCapabilities?
    public let discovery: DiscoveryCapabilities?
    public let devices: DeviceCapabilities?
    public let dispatch: DispatchCapabilities?
    public let expand: ExpandCapabilities?
    /// Advertised ONLY when the identity runtime is available, which is what
    /// makes its absence a definitive no rather than an older host.
    public let identity: IdentityCapabilities?
    public let videoUpscale: VideoUpscaleCapabilities?
    public let durableMedia: DurableMediaCapabilities?
    public let referenceUploads: ReferenceUploadCapabilities?
    /// 3-D. Absent on a host with no mesh family at all, which is why it is
    /// read through `meshExports` rather than directly -- see
    /// `MeshCapabilities`.
    public let mesh: MeshCapabilities?
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
    /// 0 means keep forever, which is NOT the same as "purged in 0 days".
    /// Read it through `Capabilities.trashRetentionDays`, which returns nil.
    public let retentionDays: Int?
}

public struct QueueCapabilities: Codable, Hashable, Sendable {
    public let canPause: Bool?
    public let canPauseJob: Bool?
    public let canCancelAll: Bool?
    public let canReorder: Bool?
    public let stableDevicePins: Bool?
    public let cooperativeCancellation: Bool?
    public let durableQueue: Bool?
    public let heterogeneousBatchMaxOutputs: Int?
}

public struct EventsCapabilities: Codable, Hashable, Sendable {
    public let available: Bool?
}
