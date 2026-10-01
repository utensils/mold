import Foundation

/// A short-lived ticket for fetching media where headers cannot be set.
///
/// `AVPlayer` builds its own requests, so it cannot carry `X-Api-Key`. mold
/// mints a token instead, valid for GET/HEAD on one media path for 15 minutes,
/// and dead after a server restart.
///
/// Thumbnails are NOT ticketable -- the route only covers full media -- which
/// is why the thumbnail cache sends the header instead.
public struct MediaToken: Codable, Hashable, Sendable {
    public let token: String?
    public let expiresAt: UInt64?
    /// False on a keyless host, where the plain URL already works.
    public let authRequired: Bool
    public let url: String?
    public let relay: RelayMediaTransfer?
}

public struct RelayMediaTransfer: Codable, Hashable, Sendable {
    public let id: String
    public let state: String
}

private struct RelayMediaStatus: Decodable {
    let state: String
    let url: String?
    let expiresAt: UInt64?
}

public extension HTTPBackend {
    /// Internal: a ticket exists for `playableURL`, its only caller.
    internal func mediaToken(forPath path: String) async throws -> MediaToken {
        struct Request: Encodable { let path: String }
        return try await post("/api/gallery/media-token", body: Request(path: path))
    }

    /// A URL a player can open directly.
    ///
    /// On a keyless host this is just the plain URL; on a keyed one it carries
    /// the ticket. Either way the caller does not have to know which -- but
    /// on a keyed host a ticket that fails to mint is a `throw`, not a plain
    /// URL the player would send with no credential and get a 401 from. The
    /// keyless answer is the HOST's (`auth_required`), never this Mac's own
    /// configuration.
    func playableURL(for filename: String) async throws -> URL {
        let urls = MediaURL(baseURL: host.baseURL)
        let plain = urls.media(filename)
        guard host.apiKey?.isEmpty == false else { return plain }

        // Signed over the same encoded path `plain` carries, not a hand-built
        // string -- a filename with a space used to sign the raw form while
        // the server compares against the request's (encoded) path, so the
        // ticket never matched.
        let ticket = try await mediaToken(forPath: urls.mediaPath(filename))
        if let resolved = try await relayMediaURL(ticket) { return resolved }
        // A host that answers `auth_required: false` is keyless and the
        // direct URL IS the right request there. That is the case this Mac
        // lands in holding a key the host no longer wants, and the server
        // answers it deliberately (`routes.rs:9800-9803`).
        guard ticket.authRequired else { return plain }

        // Past here the machine has said a ticket is required. Handing
        // `AVPlayer` the plain URL would send a request with no credential --
        // it builds its own requests and cannot set `X-Api-Key` -- and a 401
        // it has no way to report becomes a silent playback failure. So this
        // fails as the auth failure it is.
        guard let token = ticket.token,
              var components = URLComponents(url: plain, resolvingAgainstBaseURL: false)
        else { throw MoldClientError.unauthorized }

        var query = components.queryItems ?? []
        query.append(URLQueryItem(name: "media_token", value: token))
        if let expires = ticket.expiresAt {
            query.append(URLQueryItem(name: "expires", value: String(expires)))
        }
        components.queryItems = query
        guard let url = components.url else { throw MoldClientError.unauthorized }
        return url
    }

    internal func relayMediaURL(_ ticket: MediaToken) async throws -> URL? {
        if let url = ticket.url { return try await relayObjectURL(url) }
        guard let transfer = ticket.relay else { return nil }
        guard !transfer.id.isEmpty, transfer.state == "pending" else { throw MoldClientError.malformedResponse }
        let deadline = ContinuousClock.now.advanced(by: .seconds(840))
        while ContinuousClock.now < deadline {
            try Task.checkCancellation()
            let state: RelayMediaStatus = try await get("/_mold/relay/media/\(escaped(transfer.id))")
            if state.state == "ready", let url = state.url {
                if let expires = state.expiresAt, expires <= UInt64(Date().timeIntervalSince1970) { throw MoldClientError.unauthorized }
                return try await relayObjectURL(url)
            }
            guard state.state == "pending" else { throw MoldClientError.malformedResponse }
            try await Task.sleep(for: .seconds(1))
        }
        throw MoldClientError.unreachable("Media staging timed out. Try again.")
    }
}
