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

    /// The renditions a host will render, in the server's own order
    /// (`thumbnails.rs` `SIZES`). Anything else is a 422, not a rounding --
    /// Quick Look on a mesh asked for 1024 and silently showed nothing.
    public static let thumbnailSizes = [256, 512]
    public static var largestThumbnail: Int { thumbnailSizes.max()! }

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

    /// The percent-encoded path `media(_:trashed:)` serves at, with no host
    /// and no query -- what the server actually compares a ticket against
    /// (`request.uri().path()`). Deriving it from the same URL, rather than
    /// hand-building `"/api/gallery/image/\(filename)"`, is what keeps a
    /// ticket signed over a space or any other character the raw string
    /// would not have encoded valid against what gets requested.
    public func mediaPath(_ filename: String, trashed: Bool = false) -> String {
        media(filename, trashed: trashed).path(percentEncoded: true)
    }

    /// `appending(path:)` percent-encodes the segment itself. Pre-encoding
    /// here too would escape the escapes and ask for a file called `a%20b`.
    private func base(_ path: String) -> URLComponents {
        URLComponents(url: baseURL.appending(path: path), resolvingAgainstBaseURL: false)!
    }

}
