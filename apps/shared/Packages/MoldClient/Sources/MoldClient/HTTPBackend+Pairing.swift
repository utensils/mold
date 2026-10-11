import Foundation

/// What a redeemed pairing code returns (`PairingClaimResponse`): the durable
/// `mold_pair_…` credential -- `nil` from a keyless machine -- and which
/// machine answered.
public struct PairingClaim: Codable, Hashable, Sendable {
    public let apiKey: String?
    public let instanceId: String
    public let hostname: String?
    public var endpoints: [ConnectionEndpoint]?
    public var resolvedBaseURL: URL?
    public var grant: SecureDeviceGrant?
    public var hostKey: String?
    public var hostNoiseKey: String?
    public var hostBindingSignature: String?
    public var namespace: String?
    public var secure: SecureHost?
}

public enum PairingClaimError: Error, Equatable, LocalizedError {
    /// The one-use code was already redeemed, or its two minutes ran out.
    case expiredOrUsed
    /// A different machine answered at the code's address than the one that
    /// made the code. Its key is refused, not stored.
    case wrongMachine

    public var errorDescription: String? {
        switch self {
        case .expiredOrUsed:
            "This pairing code has expired. Make a new one from Pair a Phone… on your Mac."
        case .wrongMachine:
            "This code belongs to a different machine than the one that answered at that address."
        }
    }
}

public extension HTTPBackend {
    /// Redeem only after the pinned host completed Noise authentication. The invitation
    /// seed stays here; the host receives a signature bound to this exact handshake.
    static func claimPairing(
        _ payload: MobilePairingPayload, clientName: String, clientKind: String,
        now: Date = .now, session: URLSession = APISession.api
    ) async throws -> PairingClaim {
        if payload.isExpired(at: now) { throw PairingClaimError.expiredOrUsed }
        guard let invitation = payload.secureInvitation, let base = URL(string: payload.baseURL)
        else { throw SecureConnectionError.pairingRequired }
        let identity = try await SecureIdentityStore.shared.identity()
        let (connection, origin) = try await pairingConnection(invitation, identity: identity, payload: payload, base: base)
        return try await claimPairing(payload, invitation: invitation, identity: identity, connection: connection,
                                      origin: origin, clientName: clientName, clientKind: clientKind)
    }
}

extension HTTPBackend {
    static func claimPairing(_ payload: MobilePairingPayload, invitation: SecureInvitation, identity: SecureIdentity,
                            connection: SecureWebSocketConnection, origin: URL, clientName: String, clientKind: String) async throws -> PairingClaim {
        let transport = SecureHTTPTransport(connection: connection)
        do {
            let proof = try invitation.claimProof(handshakeHash: await connection.handshakeHash(), device: identity.binding())
            let backend = HTTPBackend(host: MoldHost(name: payload.name, baseURL: origin), transport: transport)
            struct Claim: Encodable { let proof: String; let clientName: String; let clientKind: String }
            var claim: PairingClaim = try await backend.post("/api/pairing/claim", body: Claim(proof: proof, clientName: clientName, clientKind: clientKind))
            guard claim.instanceId == payload.instanceId, claim.hostKey == invitation.binding.signingKey,
                  claim.hostNoiseKey == invitation.binding.noiseKey, claim.hostBindingSignature == invitation.binding.signature,
                  claim.namespace == invitation.namespace, let grant = claim.grant, claim.apiKey?.isEmpty == false
            else { throw PairingClaimError.wrongMachine }
            let secure = SecureHost(binding: invitation.binding, namespace: invitation.namespace, websocketURL: invitation.websocketURL, grant: grant)
            try SecureAdmission.validate(host: secure, device: identity.binding())
            claim.secure = secure; claim.resolvedBaseURL = origin
            await transport.close()
            return claim
        } catch { await transport.close(); throw error }
    }

    private static func pairingConnection(_ invitation: SecureInvitation, identity: SecureIdentity,
                                          payload: MobilePairingPayload, base: URL) async throws -> (SecureWebSocketConnection, URL) {
        // A one-use invitation is not raced across cloud joins. Direct candidates get
        // bounded attempts first; only the selected channel ever submits the claim.
        for endpoint in ConnectionRoutes.sanitized(payload.endpoints ?? []) where endpoint.kind != .relay {
            try Task.checkCancellation()
            guard let origin = URL(string: endpoint.url), var parts = URLComponents(url: origin, resolvingAgainstBaseURL: false) else { continue }
            parts.scheme = parts.scheme == "https" ? "wss" : "ws"; parts.path = "/_mold/secure"
            guard let url = parts.url else { continue }
            if let socket = try? await SecureWebSocketConnection.connectInvitation(invitation, identity: identity, endpoint: url, timeout: 2) {
                return (socket, origin)
            }
        }
        try Task.checkCancellation()
        return (try await SecureWebSocketConnection.connectInvitation(invitation, identity: identity), base)
    }
}
