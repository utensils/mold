import AppKit
import MoldClient
import SwiftUI

/// Thumbnails, fetched once and shared.
///
/// mold's server renders thumbnails itself (`?size=256`) and gives each one an
/// ETag keyed on the print's `media_version`, so this app needs no on-disk
/// baking tier of its own -- a URLCache plus a small memory cache does it, and
/// a re-rendered poster invalidates correctly for free.
@MainActor
@Observable
final class ThumbnailCache {
    private let images = NSCache<NSString, NSImage>()
    /// Two cells asking for the same picture share one fetch. Without this a
    /// fast scroll issues the same request several times over.
    private var inFlight: [String: Task<NSImage?, Never>] = [:]
    private let session: URLSession
    /// Kept so `purge` can empty it. A `URLSession`'s own `urlCache` is the
    /// same object, but reading it back is not guaranteed to be.
    private let responses: URLCache

    /// `stubbing` prepends protocol classes for a test and changes NOTHING
    /// else: the session a test drives is built by the same lines the shipped
    /// one is, so the configuration -- the cache, the policy -- is exercised
    /// rather than bypassed. Injecting a whole session left the shipped
    /// construction as the one branch nothing covered.
    init(stubbing protocolClasses: [AnyClass]? = nil) {
        images.totalCostLimit = 96 * 1024 * 1024
        // Deliberately modest, and emptied with everything else: this is the
        // SECOND on-disk copy of somebody's library on this Mac, and the
        // README's promise -- capped, and emptied when Mold quits -- was true
        // of the media cache and false of this one, which sat at 512 MB and
        // was swept by nothing. A thumbnail is cheap to fetch again.
        responses = URLCache(memoryCapacity: 32 * 1024 * 1024,
                             diskCapacity: 64 * 1024 * 1024)
        let configuration = URLSessionConfiguration.default
        configuration.urlCache = responses
        // Let the server's ETag decide freshness rather than a local guess.
        configuration.requestCachePolicy = .useProtocolCachePolicy
        if let protocolClasses {
            configuration.protocolClasses = protocolClasses + (configuration.protocolClasses ?? [])
        }
        session = URLSession(configuration: configuration)
    }

    /// Empties both tiers. Called when Mold quits and by Settings ▸ Empty Now,
    /// the same two doors the media cache answers.
    func purge() {
        images.removeAllObjects()
        responses.removeAllCachedResponses()
    }

    func cached(_ key: String) -> NSImage? { images.object(forKey: key as NSString) }

    func image(for item: LibraryEntry, host: MoldHost, size: Int) async -> NSImage? {
        let key = cacheKey(item, size: size)
        if let hit = cached(key) { return hit }
        if let running = inFlight[key] { return await running.value }

        let task = Task<NSImage?, Never> { [session] in
            var request = URLRequest(
                url: MediaURL(baseURL: host.baseURL).thumbnail(item.print.filename, size: size)
            )
            // Thumbnails are NOT ticketable -- the media-token route covers
            // only full media. An authenticated host needs the header here.
            if let apiKey = host.apiKey, !apiKey.isEmpty {
                request.setValue(apiKey, forHTTPHeaderField: "X-Api-Key")
            }
            // Streamed and BOUNDED, not `data(for:)`: a thumbnail is tens of
            // kilobytes and the answer is decoded into an `NSImage`, so an
            // unbounded body from a broken or hostile host is an allocation
            // this app never recovers from. The declared length is refused
            // before a byte is read, and the count is kept as it arrives
            // because a host that lies about the length is exactly the one
            // this guards against.
            guard let (stream, response) = try? await session.bytes(for: request),
                  let http = response as? HTTPURLResponse,
                  (200..<300).contains(http.statusCode),
                  http.expectedContentLength <= Int64(ResponseCeiling.thumbnail),
                  let data = try? await stream.collected(upTo: ResponseCeiling.thumbnail)
            else { return nil }
            return NSImage(data: data)
        }
        inFlight[key] = task
        let image = await task.value
        inFlight[key] = nil
        if let image {
            images.setObject(image, forKey: key as NSString, cost: cost(of: image))
        }
        return image
    }

    /// Keyed on `media_version` so re-rendered bytes are a different entry
    /// rather than a stale hit under the same name.
    private func cacheKey(_ item: LibraryEntry, size: Int) -> String {
        "\(item.hostID)|\(item.print.filename)|\(item.print.mediaVersion ?? "")|\(size)"
    }

    private func cost(of image: NSImage) -> Int {
        Int(image.size.width * image.size.height * 4)
    }
}
