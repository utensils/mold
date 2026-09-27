import Foundation

// The blocks that describe a MACHINE and what it will condition a render
// on, split from `Capabilities.swift` for size. Every one of these is read
// through `Capabilities+Reading`, never directly -- see that file for what
// each absence means.

public struct CatalogCapabilities: Codable, Hashable, Sendable {
    public let available: Bool
    /// Which families this host will look for in the catalogs. An empty list
    /// is a host that browses nothing, not a host that browses everything.
    public let families: [String]?
    public let sort: [String]?
}

public struct DiscoveryCapabilities: Codable, Hashable, Sendable {
    public let canBrowse: Bool?
}

public struct DeviceCapabilities: Codable, Hashable, Sendable {
    /// `GET /api/devices` answers.
    public let available: Bool
    /// A live enable/disable is authoritative. True only while scheduler V2
    /// owns dispatch -- legacy, observe and maintenance runtimes report false,
    /// and persisting a change they cannot enforce would be a lie.
    public let lifecycle: Bool
    /// A disabled device can still be enabled for the NEXT restart even where
    /// a live change is not authoritative. A different power, not a weaker one.
    public let restartEnable: Bool?
    public let stablePins: Bool?
    public let plannedLanes: Bool?
    public let learnedEta: Bool?
}

public struct DispatchCapabilities: Codable, Hashable, Sendable {
    public let modes: [String]?
    public let activeMode: String?
    public let v2Authoritative: Bool?
    public let requestPlacementPreview: Bool?
}

public enum ExpandBackend: String, OpenWireEnum {
    case local
    case api
    case unknown
}

public struct ExpandCapabilities: Codable, Hashable, Sendable {
    /// An API backend is configured, or local expansion is compiled in.
    public let configured: Bool
    /// Whether the configured LOCAL model is installed. Meaningless for an API
    /// backend, which is why it is optional rather than false there.
    public let modelPresent: Bool?
    public let backend: ExpandBackend?
    /// Subject-preserving Remix is its own endpoint and its own flag: a host
    /// that expands may not remix.
    public let remix: Bool?
    /// The manifest model local expansion resolves. Present so nobody
    /// hard-codes `qwen3-expand` to offer the pull. Absent for API backends
    /// and for hosts that predate the field.
    public let model: String?
}

public struct IdentityCapabilities: Codable, Hashable, Sendable {
    public let multiPhoto: Bool?
    public let maxPhotos: Int?
    public let trueCfg: Bool?
}

public struct VideoUpscaleCapabilities: Codable, Hashable, Sendable {
    public let available: Bool
    public let galleryImage: Bool?
    public let sourceLibrary: Bool?
    public let sourceUpload: Bool?
    public let inputContainers: [String]?
    public let outputContainer: String?
    /// The host's own sentence about what framewise upscaling does to a clip.
    /// Shown verbatim -- it is a caveat, and paraphrasing a caveat weakens it.
    public let disclosure: String?
}

public struct DurableMediaCapabilities: Codable, Hashable, Sendable {
    public let protocolVersion: Int?
    public let encryptedAtRest: Bool?
    public let generateRequestMedia: Bool?
    public let identity: Bool?
}

public struct ReferenceUploadCapabilities: Codable, Hashable, Sendable {
    /// Advertised-but-off is real: the protocol needs API-key auth, so a
    /// keyless host reports the block with `available: false`.
    public let available: Bool
    public let protocolVersion: Int?
    public let requiresApiKey: Bool?
    public let sessionPath: String?
    public let uploadPath: String?
    public let sessionHandleHeader: String?
    public let uploadHandleHeader: String?
    public let maxFileBytes: Int?
    public let maxSessionBytes: Int?
}
