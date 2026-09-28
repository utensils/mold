import Foundation

/// What a redeemed pairing code returns (`PairingClaimResponse`): the durable
/// `mold_pair_…` credential -- `nil` from a keyless machine -- and which
/// machine answered.
public struct PairingClaim: Codable, Hashable, Sendable {
    public let apiKey: String?
    public let instanceId: String
    public let hostname: String?
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

private struct PairingClaimBody: Encodable {
    let token: String?
    let clientName: String
    let clientKind: String
}

public extension HTTPBackend {
    /// Redeems a pairing code at the address it names (`POST
    /// /api/pairing/claim`). The request carries NO key -- the machine has
    /// none for us yet -- and goes through the same `RedirectGuard` as every
    /// other call. `clientKind` is `iphone` or `ipad`; the server files
    /// anything else as `mobile`.
    static func claimPairing(
        _ payload: MobilePairingPayload, clientName: String, clientKind: String,
        now: Date = .now, session: URLSession = APISession.api
    ) async throws -> PairingClaim {
        if payload.isExpired(at: now) { throw PairingClaimError.expiredOrUsed }
        guard let base = URL(string: payload.baseURL) else { throw MobilePairingPayload.ParseError.unsupported }
        let backend = HTTPBackend(host: MoldHost(name: payload.name, baseURL: base), session: session)
        let claim: PairingClaim
        do {
            claim = try await backend.post(
                "/api/pairing/claim",
                body: PairingClaimBody(token: payload.token, clientName: clientName, clientKind: clientKind))
        } catch MoldClientError.unauthorized {
            throw PairingClaimError.expiredOrUsed
        }
        guard claim.instanceId == payload.instanceId else { throw PairingClaimError.wrongMachine }
        return claim
    }
}
