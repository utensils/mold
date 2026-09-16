import Foundation

/// Builds the media URLs for one host.
///
/// Thumbnails are NOT ticketable and must be fetched with `X-Api-Key`; full
/// media can carry a short-lived `media_token` for players that cannot set
/// headers. Keeping both in one place is what stops those two rules being
/// applied to the wrong route.
public struct MediaURL: Sendable {
    public let baseURL: URL

    public init(baseURL: URL) { self.baseURL = baseURL }

    /// `size` is honoured only by hosts new enough to report the rendition
    /// header; older ones ignore it and return their own default.
    public func thumbnail(_ filename: String, size: Int = 256, trashed: Bool = false) -> URL {
        var components = base("/api/gallery/thumbnail/\(filename)")
        var query = [URLQueryItem(name: "size", value: String(size))]
        if trashed { query.append(URLQueryItem(name: "view", value: "trash")) }
        components.queryItems = query
        return components.url!
    }

    public func media(_ filename: String, trashed: Bool = false) -> URL {
        var components = base("/api/gallery/image/\(filename)")
        if trashed { components.queryItems = [URLQueryItem(name: "view", value: "trash")] }
        return components.url!
    }

    /// `appending(path:)` percent-encodes the segment itself. Pre-encoding
    /// here too would escape the escapes and ask for a file called `a%20b`.
    private func base(_ path: String) -> URLComponents {
        URLComponents(url: baseURL.appending(path: path), resolvingAgainstBaseURL: false)!
    }

}
