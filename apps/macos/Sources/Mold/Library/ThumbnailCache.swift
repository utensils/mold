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

    init() {
        images.totalCostLimit = 96 * 1024 * 1024
        let configuration = URLSessionConfiguration.default
        configuration.urlCache = URLCache(
            memoryCapacity: 32 * 1024 * 1024,
            diskCapacity: 512 * 1024 * 1024
        )
        // Let the server's ETag decide freshness rather than a local guess.
        configuration.requestCachePolicy = .useProtocolCachePolicy
        session = URLSession(configuration: configuration)
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
            guard let (data, response) = try? await session.data(for: request),
                  let http = response as? HTTPURLResponse,
                  (200..<300).contains(http.statusCode)
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
