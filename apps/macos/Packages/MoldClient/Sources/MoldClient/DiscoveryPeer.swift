import Foundation

/// One DNS-SD result from `GET /api/discovery/peers`. The app connects to
/// `url` itself; the serving host is discovery-only and proxies nothing.
public struct DiscoveryPeer: Codable, Hashable, Sendable, Identifiable {
    public let name: String
    public let url: String
    public let authRequired: Bool
    /// The fleet identity, when the peer is new enough to announce one. It is
    /// what catches the same machine found at a second address.
    public let instanceId: String?
    public let isThisMachine: Bool

    public var id: String { url }
}
