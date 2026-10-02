import Foundation

/// The QR payload, ported from `studio/api/pairing.ts:36-56`. The Mac only
/// PRODUCES a code (`url`); the iPhone companion reads one (`parse`, in
/// `MobilePairingPayload+Parse.swift`) and redeems it (`HTTPBackend.claimPairing`).
public struct MobilePairingPayload: Hashable, Sendable {
    static let version = 1

    public let baseURL: String
    public let token: String?
    public let expiresAt: UInt64?
    public let instanceId: String
    public let name: String
    public let endpoints: [ConnectionEndpoint]?

    init(baseURL: String, token: String?, expiresAt: UInt64?, instanceId: String, name: String, endpoints: [ConnectionEndpoint]? = nil) {
        self.baseURL = baseURL
        self.token = token
        self.expiresAt = expiresAt
        self.instanceId = instanceId
        self.name = name
        self.endpoints = endpoints
    }

    /// A keyless machine's code carries its address and identity alone --
    /// no token, no expiry -- and the phone's claim is answered with no key
    /// to store. `nil` only for a keyed machine that sent no token: that code
    /// would redeem nothing, so there is nothing honest to show.
    public init?(session: PairingSession, baseURL: URL, name: String) {
        guard session.token != nil || !session.authRequired else { return nil }
        self.baseURL = baseURL.absoluteString
        self.token = session.token
        expiresAt = session.expiresAt
        instanceId = session.instanceId
        self.name = name
        endpoints = session.endpoints
    }

    /// Where a pairing code points (`MOBILE_PAIRING_LINK` in `pairing.ts`): a
    /// universal link the Companion claims, and on a phone without it a page
    /// that says what to install. The payload rides in the fragment, which a
    /// browser never sends to utensils.io.
    public static let link = URL(string: "https://utensils.io/mold/pair")!

    /// `https://utensils.io/mold/pair#version=1&base_url=…&token=…&expires_at=…&instance_id=…&name=…`,
    /// byte-identical to `mobilePairingUrl` (`studio/api/pairing.ts`; both
    /// held to `studio/api/pairing.fixtures.json`): the same field order,
    /// `token` and `expires_at` present only when non-nil, and no `type` at
    /// all -- the parser on the other end synthesises it. `nil` only if
    /// `FormURLEncoded`'s output somehow failed to parse as a URL, which does
    /// not happen for its own escaping.
    private static func endpointData(_ endpoints: [ConnectionEndpoint]) -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try? encoder.encode(ConnectionRoutes.sanitized(endpoints))
    }

    public var url: URL? {
        var pairs: [(String, String)] = [("version", String(Self.version)), ("base_url", baseURL)]
        if let token { pairs.append(("token", token)) }
        if let expiresAt { pairs.append(("expires_at", String(expiresAt))) }
        pairs.append(("instance_id", instanceId))
        pairs.append(("name", name))
        if let endpoints, !endpoints.isEmpty,
           let data = Self.endpointData(endpoints),
           let json = String(data: data, encoding: .utf8) { pairs.append(("endpoints", json)) }
        return URL(string: "\(Self.link.absoluteString)#\(FormURLEncoded.queryString(pairs))")
    }
}
