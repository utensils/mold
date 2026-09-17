import Foundation

/// `POST /api/pairing/sessions`'s answer. `routes.rs:9499-9506`
/// (`PairingSessionResponse`). The durable key never rides this response --
/// `token` redeems it exactly once, against this exact server, within
/// `expiresAt`.
public struct PairingSession: Codable, Hashable, Sendable {
    /// `nil` on a keyless host -- there is no key to hand over
    /// (`routes.rs:9557-9598`).
    public let token: String?
    public let expiresAt: UInt64?
    public let authRequired: Bool
    public let instanceId: String
    public let hostname: String?

    public init(
        token: String?, expiresAt: UInt64?, authRequired: Bool, instanceId: String, hostname: String?
    ) {
        self.token = token
        self.expiresAt = expiresAt
        self.authRequired = authRequired
        self.instanceId = instanceId
        self.hostname = hostname
    }
}

/// One row of `GET /api/pairing/clients`. `routes.rs:9523-9529`
/// (`PairedClientResponse`).
public struct PairedClient: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let clientKind: String
    public let createdAtMs: Int64
    public let lastUsedAtMs: Int64?

    public init(
        id: String, name: String, clientKind: String, createdAtMs: Int64, lastUsedAtMs: Int64?
    ) {
        self.id = id
        self.name = name
        self.clientKind = clientKind
        self.createdAtMs = createdAtMs
        self.lastUsedAtMs = lastUsedAtMs
    }
}

/// `GET /api/pairing/clients`. `routes.rs:9532-9535` (`PairedClientsResponse`).
public struct PairedClients: Codable, Hashable, Sendable {
    public let authRequired: Bool
    public let pairingAvailable: Bool
    public let clients: [PairedClient]

    public init(authRequired: Bool, pairingAvailable: Bool, clients: [PairedClient]) {
        self.authRequired = authRequired
        self.pairingAvailable = pairingAvailable
        self.clients = clients
    }

    /// Whether this machine has anything to pair. `pairingAvailable` is
    /// `true` even on a keyless host (`routes.rs:9678-9685`), so it is NOT
    /// the gate -- a host with no key has no key to hand over and can never
    /// hold a client. `authRequired` is the only honest gate.
    public var canPair: Bool { authRequired && pairingAvailable }
}
