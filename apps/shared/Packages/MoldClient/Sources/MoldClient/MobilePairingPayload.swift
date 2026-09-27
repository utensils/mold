import Foundation

/// The QR payload, ported from `studio/api/pairing.ts:36-56`. This app is
/// never a pairing CLAIMANT (decision 14) -- it has manual host-and-key entry
/// and only ever PRODUCES a code, so there is no parser here, only `url`.
public struct MobilePairingPayload: Hashable, Sendable {
    static let version = 1

    public let baseURL: String
    public let token: String?
    public let expiresAt: UInt64?
    public let instanceId: String
    public let name: String

    /// `nil` when `session.token == nil` -- a code with no token redeems
    /// nothing, so there is nothing honest to show.
    public init?(session: PairingSession, baseURL: URL, name: String) {
        guard let token = session.token else { return nil }
        self.baseURL = baseURL.absoluteString
        self.token = token
        expiresAt = session.expiresAt
        instanceId = session.instanceId
        self.name = name
    }

    /// `mold://pair?version=1&base_url=…&token=…&expires_at=…&instance_id=…&name=…`,
    /// byte-identical to `mobilePairingUrl` (`pairing.ts:47-56`): the same
    /// field order, `token` and `expires_at` present only when non-nil, and
    /// no `type` at all -- the parser on the other end synthesises it
    /// (`pairing.ts:133`). `nil` only if `FormURLEncoded`'s output somehow
    /// failed to parse as a URL, which does not happen for its own escaping.
    public var url: URL? {
        var pairs: [(String, String)] = [("version", String(Self.version)), ("base_url", baseURL)]
        if let token { pairs.append(("token", token)) }
        if let expiresAt { pairs.append(("expires_at", String(expiresAt))) }
        pairs.append(("instance_id", instanceId))
        pairs.append(("name", name))
        return URL(string: "mold://pair?\(FormURLEncoded.queryString(pairs))")
    }
}
