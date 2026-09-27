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
    /// The commit the server was built from. `version` alone is the
    /// workspace's release number, which every build between two releases
    /// shares -- a week-stale engine and tonight's nightly both said
    /// "0.31.0". `nil` on a build without git metadata.
    public let gitSha: String?
    public let buildDate: String?
    /// `nil` when the host cannot resolve its own hostname. `#[serde(skip_serializing_if)]`
    /// on the server omits the key entirely rather than sending an empty
    /// string, and a non-optional `String` here threw on that host and made
    /// `check(_:)` report it as DOWN.
    public let hostname: String?
    public let busy: Bool
    public let queueDepth: Int?
    public let memoryStatus: String?
    public let gpus: [GPU]?
    /// Identifies this run of the server. A retry must name it, so work is
    /// never aimed at a host that has restarted since.
    public let instanceId: String?
    public let uptimeSecs: UInt64
    /// The MACHINE's own figure for what its models occupy, never a sum of
    /// installed rows' `diskUsageBytes` -- a shared VAE or encoder is counted
    /// once per model that references it, so the column never adds up to
    /// this (`routes.rs:5931-5948`, design fact 3, M5). `nil` on a host that
    /// predates the field, which is a real absence, not a zero.
    public let modelsDisk: ModelsDisk?
    /// Whether this machine is dispatching new work at all
    /// (`POST /api/queue/pause`). `nil` on a machine that predates the field,
    /// which reads as not paused -- it has no gate to be behind.
    public let queuePaused: Bool?

    public struct GPU: Hashable, Codable, Sendable {
        public let ordinal: Int
        public let name: String
        public let vramTotalBytes: UInt64?
        public let vramUsedBytes: UInt64?
        public let state: String?

    }

    public struct ModelsDisk: Hashable, Codable, Sendable {
        public let totalBytes: UInt64
        public let freeBytes: UInt64
    }

}

extension ServerStatus {
    /// `0.31.0 (9c81f69)` -- the release number and, where the machine knows
    /// it, the commit it was built from.
    public var versionLabel: String {
        guard let gitSha, !gitSha.isEmpty else { return version }
        return "\(version) (\(gitSha.prefix(7)))"
    }
}
