import ImageIO
import MoldClient
import SwiftUI
import UIKit

/// Thumbnails for the grid, the viewer's first frame, and the widget.
///
/// Memory first, then the bounded disk cache (`DiskThumbnailStore`: 4,000 /
/// 256 MiB / 2 MiB), then the machine. Two tiles asking for the same picture
/// share one request. A print with no `media_version` never reaches disk --
/// nothing could tell a re-render from the old file.
@Observable
final class ThumbnailLoader {
    @ObservationIgnored let hosts: HostStore
    @ObservationIgnored let disk: DiskThumbnailStore
    @ObservationIgnored private let memory = NSCache<NSString, UIImage>()
    @ObservationIgnored private var inflight: [String: Task<UIImage?, Never>] = [:]

    init(hosts: HostStore, directory: URL = URL.cachesDirectory.appending(path: "thumbnails")) {
        self.hosts = hosts
        self.disk = DiskThumbnailStore(directory: directory)
        memory.totalCostLimit = 96 << 20
    }

    /// `pixels` is the longer edge the tile draws, rounded up to a bucket so a
    /// pinch between tile sizes reuses what is already here.
    func image(for entry: LibraryEntry, pixels: Int, trashed: Bool = false) async -> UIImage? {
        let size = Self.bucket(pixels)
        let key = memoryKey(entry, size)
        if let cached = memory.object(forKey: key as NSString) { return cached }
        if let running = inflight[key] { return await running.value }
        let task = Task { await self.load(entry, size: size, trashed: trashed) }
        inflight[key] = task
        let image = await task.value
        inflight[key] = nil
        if let image {
            memory.setObject(image, forKey: key as NSString, cost: Int(image.size.width * image.size.height * 4))
        }
        return image
    }

    /// A machine removed: its thumbnails go with it.
    func forget(host: MoldHost.ID) async {
        memory.removeAllObjects()
        await disk.evict(host: host.uuidString)
    }

    func emptyCaches() async {
        memory.removeAllObjects()
        await disk.purge()
    }

    private func load(_ entry: LibraryEntry, size: Int, trashed: Bool) async -> UIImage? {
        let diskKey = entry.print.mediaVersion.map {
            DiskThumbnailStore.Key(host: entry.hostID.uuidString, filename: entry.print.filename,
                                   mediaVersion: $0, size: size)
        }
        if let diskKey, let data = await disk.data(for: diskKey), let image = Self.decode(data, pixels: size) {
            return image
        }
        guard let client = hosts.backend(for: entry.hostID),
              let data = try? await client.thumbnail(entry.print.filename, size: size, trashed: trashed)
        else { return nil }
        if let diskKey { await disk.store(data, for: diskKey) }
        return Self.decode(data, pixels: size)
    }

    private func memoryKey(_ entry: LibraryEntry, _ size: Int) -> String {
        "\(entry.hostID.uuidString)|\(entry.print.filename)|\(entry.print.mediaVersion ?? "-")|\(size)"
    }

    /// 256, 512, 1024, 2048: few enough sizes that the caches hit.
    static func bucket(_ pixels: Int) -> Int {
        [256, 512, 1024, 2048].first { $0 >= pixels } ?? 2048
    }

    /// ImageIO's thumbnailer: decodes straight to the size drawn, never the
    /// full print into memory first.
    nonisolated static func decode(_ data: Data, pixels: Int) -> UIImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: pixels,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        return UIImage(cgImage: image)
    }
}

/// A print's picture at the size it is drawn, from the loader; a quiet
/// placeholder until it arrives (never a spinner per tile).
struct PrintThumbnail: View {
    @Environment(ThumbnailLoader.self) private var loader
    @Environment(\.displayScale) private var scale
    let entry: LibraryEntry
    let points: CGFloat
    var trashed = false
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Rectangle().fill(.fill.tertiary)
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            }
        }
        .task(id: "\(entry.id.filename)|\(entry.print.mediaVersion ?? "")|\(Int(points * scale))") {
            image = await loader.image(for: entry, pixels: Int(points * scale), trashed: trashed)
        }
    }
}
