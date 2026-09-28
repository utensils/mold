import ImageIO
import MoldClient
import SwiftUI
import UIKit

/// Pictures for the grid, the viewer and the widget -- and the offline library.
///
/// Memory first, then disk, then the machine; two tiles asking for the same
/// picture share one request, and decoding never happens on the main thread.
/// Two disk stores share Settings' storage limit (`OfflineLimit`): thumbnails
/// (30%) and the full prints opened in the viewer (the rest). Both live in
/// Application Support, excluded from backup, so the system does not clear
/// them the way it may clear Caches -- that is what makes the Library work
/// with no connection.
@Observable
final class ThumbnailLoader {
    /// Background saving for offline (`save(_:)`): how far it has got.
    private(set) var saving: (done: Int, total: Int)?

    @ObservationIgnored let hosts: HostStore
    @ObservationIgnored let disk: DiskThumbnailStore
    @ObservationIgnored let originals: DiskThumbnailStore
    @ObservationIgnored private let memory = NSCache<NSString, UIImage>()
    @ObservationIgnored private var inflight: [String: Task<UIImage?, Never>] = [:]
    @ObservationIgnored private var saveTask: Task<Void, Never>?
    /// Which save run owns `saving`: a cancelled run's last word must not
    /// clear the progress of the run that replaced it.
    @ObservationIgnored private var saveRun = 0
    /// How many callers wait on each in-flight load; when the last one is
    /// cancelled (a viewer page swiped past), the download is too.
    @ObservationIgnored private var waiters: [String: Int] = [:]

    /// The only sizes machines serve (`/api/gallery/thumbnail?size=`): any
    /// other is refused with a 422, which drew every tile empty.
    static let sizes = [256, 512]
    /// The longest edge a viewed print is decoded to for the screen.
    static let viewerPixels = 3_000

    init(hosts: HostStore, directory: URL = ThumbnailLoader.defaultDirectory,
         limit: OfflineLimit = .current()) {
        self.hosts = hosts
        disk = DiskThumbnailStore(directory: directory.appending(path: "thumbnails"),
                                  maxItems: Self.items(limit.thumbnailBytes, each: 20_000),
                                  maxBytes: limit.thumbnailBytes, maxItemBytes: 2 << 20)
        originals = DiskThumbnailStore(directory: directory.appending(path: "originals"),
                                       maxItems: Self.items(limit.originalBytes, each: 1_000_000),
                                       maxBytes: limit.originalBytes, maxItemBytes: 60 << 20)
        memory.totalCostLimit = 128 << 20
        Self.excludeFromBackup(directory)
    }

    nonisolated static var defaultDirectory: URL {
        URL.applicationSupportDirectory.appending(path: "offline", directoryHint: .isDirectory)
    }

    private static func items(_ bytes: Int, each: Int) -> Int { max(200, bytes / each) }

    // MARK: - Reading

    /// `pixels` is the longer edge drawn; it is rounded to a size machines
    /// serve, so a pinch between tile sizes mostly reuses what is here.
    func image(for entry: LibraryEntry, pixels: Int, trashed: Bool = false) async -> UIImage? {
        let size = Self.bucket(pixels)
        let key = memoryKey(entry, size)
        if let cached = memory.object(forKey: key as NSString) { return cached }
        return await shared(key) { await self.loadThumbnail(entry, size: size, trashed: trashed) }
    }

    /// A thumbnail already in memory, for a viewer's first frame.
    func cachedThumbnail(for entry: LibraryEntry) -> UIImage? {
        Self.sizes.reversed().lazy.compactMap { self.memory.object(forKey: self.memoryKey(entry, $0) as NSString) }.first
    }

    /// The print itself, for the viewer: saved on disk after the first view,
    /// so it opens with no connection too. `nil` offline if never opened.
    func original(for entry: LibraryEntry, trashed: Bool = false) async -> UIImage? {
        let key = memoryKey(entry, 0)
        if let cached = memory.object(forKey: key as NSString) { return cached }
        return await shared(key) { await self.loadOriginal(entry, trashed: trashed) }
    }

    private func shared(_ key: String, _ work: @escaping () async -> UIImage?) async -> UIImage? {
        let task: Task<UIImage?, Never>
        if let running = inflight[key] {
            task = running
        } else {
            task = Task { await work() }
            inflight[key] = task
        }
        waiters[key, default: 0] += 1
        let image = await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            Task { @MainActor [weak self] in self?.abandon(key, task) }
        }
        let left = max(0, (waiters[key] ?? 1) - 1)
        waiters[key] = left == 0 ? nil : left
        if left == 0, inflight[key] == task { inflight[key] = nil }
        if let image {
            memory.setObject(image, forKey: key as NSString, cost: Int(image.size.width * image.size.height * 4))
        }
        return image
    }

    /// The last caller waiting on a load gave up: stop downloading.
    private func abandon(_ key: String, _ task: Task<UIImage?, Never>) {
        guard inflight[key] == task, (waiters[key] ?? 0) <= 1 else { return }
        task.cancel()
        inflight[key] = nil
        waiters[key] = nil
    }

    private func loadThumbnail(_ entry: LibraryEntry, size: Int, trashed: Bool) async -> UIImage? {
        // The size asked for, then the other one (a 512 saved for offline
        // draws a 256 tile perfectly well), then the machine.
        for candidate in [size] + Self.sizes.filter({ $0 != size }).sorted(by: { $0 > $1 }) {
            if let key = Self.diskKey(entry, size: candidate), let data = await disk.data(for: key),
               let image = await Self.decoded(data, pixels: size) {
                return image
            }
        }
        guard let client = reachableBackend(for: entry),
              let data = try? await client.thumbnail(entry.print.filename, size: size, trashed: trashed)
        else { return nil }
        if let key = Self.diskKey(entry, size: size) { await disk.store(data, for: key) }
        return await Self.decoded(data, pixels: size)
    }

    /// The machine, only if it is answering: a machine that is down would
    /// have every tile wait out a timeout for nothing.
    private func reachableBackend(for entry: LibraryEntry) -> (any MoldBackend)? {
        guard let host = hosts.host(entry.hostID), hosts.isUp(host) else { return nil }
        return hosts.backend(for: host)
    }

    private func loadOriginal(_ entry: LibraryEntry, trashed: Bool) async -> UIImage? {
        let key = Self.diskKey(entry, size: 0)
        if let key, let data = await originals.data(for: key),
           let image = await Self.decoded(data, pixels: Self.viewerPixels) {
            return image
        }
        guard let client = reachableBackend(for: entry),
              let data = try? await client.media(entry.print.filename, trashed: trashed),
              !Task.isCancelled
        else { return nil }
        if let key { await originals.store(data, for: key) }
        return await Self.decoded(data, pixels: Self.viewerPixels)
    }

    // MARK: - Offline

    /// Saves thumbnails for these prints (newest first) in the background, four
    /// at a time, skipping what is already on disk. A new call replaces the one
    /// running; `cancelSaving()` stops it.
    func save(_ entries: [LibraryEntry], size: Int = 512) {
        saveTask?.cancel()
        saveRun += 1
        let run = saveRun
        let size = Self.bucket(size)
        let work = entries
        saving = (0, work.count)
        saveTask = Task { [weak self] in
            await withTaskGroup(of: Void.self) { group in
                var next = 0
                func enqueue() {
                    guard next < work.count, let self else { return }
                    let entry = work[next]
                    next += 1
                    group.addTask { await self.saveOne(entry, size: size) }
                }
                for _ in 0..<4 { enqueue() }
                while await group.next() != nil {
                    guard !Task.isCancelled else { group.cancelAll(); break }
                    if self?.saveRun == run, let done = self?.saving?.done { self?.saving = (done + 1, work.count) }
                    enqueue()
                }
            }
            if self?.saveRun == run { self?.saving = nil }
        }
    }

    func cancelSaving() {
        saveTask?.cancel()
        saveRun += 1
        saving = nil
    }

    private func saveOne(_ entry: LibraryEntry, size: Int) async {
        guard let key = Self.diskKey(entry, size: size) else { return }
        // Either size on disk is enough to draw the tile offline.
        for candidate in Self.sizes {
            if let saved = Self.diskKey(entry, size: candidate), await disk.contains(saved) { return }
        }
        guard !Task.isCancelled, let client = reachableBackend(for: entry),
              let data = try? await client.thumbnail(entry.print.filename, size: size, trashed: false)
        else { return }
        await disk.store(data, for: key)
    }

    // MARK: - Managing

    /// Bytes on this device, thumbnails and opened prints together.
    func diskBytes() async -> Int64 {
        Int64(await disk.totalBytes) + Int64(await originals.totalBytes)
    }

    func apply(_ limit: OfflineLimit) async {
        await disk.setLimits(maxItems: Self.items(limit.thumbnailBytes, each: 20_000), maxBytes: limit.thumbnailBytes)
        await originals.setLimits(maxItems: Self.items(limit.originalBytes, each: 1_000_000),
                                  maxBytes: limit.originalBytes)
    }

    /// A machine removed: its pictures go with it.
    func forget(host: MoldHost.ID) async {
        memory.removeAllObjects()
        await disk.evict(host: host.uuidString)
        await originals.evict(host: host.uuidString)
    }

    func emptyCaches() async {
        cancelSaving()
        memory.removeAllObjects()
        await disk.purge()
        await originals.purge()
    }

    // MARK: - Keys and decoding

    private func memoryKey(_ entry: LibraryEntry, _ size: Int) -> String {
        "\(entry.hostID.uuidString)|\(entry.print.filename)|\(Self.version(entry.print))|\(size)"
    }

    /// `nil` when nothing could tell a re-rendered print from the old one:
    /// that print is held in memory only, never on disk.
    static func diskKey(_ entry: LibraryEntry, size: Int) -> DiskThumbnailStore.Key? {
        guard let version = diskVersion(entry.print) else { return nil }
        return DiskThumbnailStore.Key(host: entry.hostID.uuidString, filename: entry.print.filename,
                                      mediaVersion: version, size: size)
    }

    /// The machine's `media_version` when it sends one; otherwise the print's
    /// time AND byte size, which together change when a print is re-rendered.
    /// Without a size there is nothing trustworthy: `nil`.
    static func diskVersion(_ print: GalleryPrint) -> String? {
        if let version = print.mediaVersion { return version }
        guard let bytes = print.sizeBytes else { return nil }
        return "t\(print.timestamp)-\(bytes)"
    }

    /// For memory keys and view identity: always something.
    static func version(_ print: GalleryPrint) -> String {
        diskVersion(print) ?? "t\(print.timestamp)"
    }

    /// 256 or 512: the sizes machines serve. Anything larger draws the 512.
    static func bucket(_ pixels: Int) -> Int {
        sizes.first { $0 >= pixels } ?? sizes[sizes.count - 1]
    }

    private static func decoded(_ data: Data, pixels: Int) async -> UIImage? {
        await Task.detached(priority: .userInitiated) { decode(data, pixels: pixels) }.value
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

    private static func excludeFromBackup(_ directory: URL) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var url = directory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
        // The first builds kept thumbnails in Caches; that copy is orphaned.
        try? FileManager.default.removeItem(at: URL.cachesDirectory.appending(path: "thumbnails"))
    }
}
