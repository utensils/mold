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
}

public extension HTTPBackend {
    func mediaToken(forPath path: String) async throws -> MediaToken {
        struct Request: Encodable { let path: String }
        return try await post("/api/gallery/media-token", body: Request(path: path))
    }

    /// A URL a player can open directly.
    ///
    /// On a keyless host this is just the plain URL; on a keyed one it carries
    /// the ticket. Either way the caller does not have to know which.
    func playableURL(for filename: String) async -> URL {
        let urls = MediaURL(baseURL: host.baseURL)
        let plain = urls.media(filename)
        guard host.apiKey?.isEmpty == false else { return plain }

        guard let ticket = try? await mediaToken(forPath: "/api/gallery/image/\(filename)"),
              ticket.authRequired, let token = ticket.token,
              var components = URLComponents(url: plain, resolvingAgainstBaseURL: false)
        else { return plain }

        var query = components.queryItems ?? []
        query.append(URLQueryItem(name: "media_token", value: token))
        if let expires = ticket.expiresAt {
            query.append(URLQueryItem(name: "expires", value: String(expires)))
        }
        components.queryItems = query
        return components.url ?? plain
    }
}
