import Foundation

/// A mold server the app can talk to.
///
/// Every call carries its host. mold's own rule is that a print, a job and a
/// workflow each live on exactly ONE machine, and an identity must never be
/// inferred from anything but the copy that carries it -- so there is no
/// ambient "current host" inside the client, only an explicit one here.
public struct MoldHost: Identifiable, Hashable, Codable, Sendable {
    public let id: UUID
    public var name: String
    public var baseURL: URL
    /// `nil` on a keyless host, which is a first-class state and NOT an error:
    /// a mold server with no `MOLD_API_KEY` leaves every route open.
    public var apiKey: String?

    public init(id: UUID = UUID(), name: String, baseURL: URL, apiKey: String? = nil) {
        self.id = id
        self.name = name
        self.baseURL = baseURL
        self.apiKey = apiKey
    }
}

/// What a host reports about itself. Mirrors `GET /api/status`.
public struct ServerStatus: Hashable, Codable, Sendable {
    public let version: String
    public let hostname: String
    public let busy: Bool
    public let queueDepth: Int?
    public let memoryStatus: String?
    public let gpus: [GPU]?

    public struct GPU: Hashable, Codable, Sendable {
        public let ordinal: Int
        public let name: String
        public let vramTotalBytes: UInt64?
        public let vramUsedBytes: UInt64?
        public let state: String?

        private enum CodingKeys: String, CodingKey {
            case ordinal, name, state
            case vramTotalBytes = "vram_total_bytes"
            case vramUsedBytes = "vram_used_bytes"
        }
    }

    private enum CodingKeys: String, CodingKey {
        case version, hostname, busy, gpus
        case queueDepth = "queue_depth"
        case memoryStatus = "memory_status"
    }
}
