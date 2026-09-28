import Foundation
import ImageIO
import MoldClient
import UIKit
import UniformTypeIdentifiers
import WidgetKit

/// Writes what the widgets draw into the App Group (DESIGN.md §5.8): the
/// newest prints with a small JPEG each, the machines by name, and the queue
/// in numbers. Timelines reload only when something they show changed.
@Observable
final class WidgetSnapshotWriter {
    @ObservationIgnored let hosts: HostStore
    @ObservationIgnored let library: LibraryStore
    @ObservationIgnored let queue: QueueStore
    @ObservationIgnored let generate: GenerateController
    @ObservationIgnored let thumbnails: ThumbnailLoader
    @ObservationIgnored let directory: URL
    @ObservationIgnored let reload: () -> Void

    static let printLimit = 24
    static let pixels = 360

    init(hosts: HostStore, library: LibraryStore, queue: QueueStore, generate: GenerateController,
         thumbnails: ThumbnailLoader, directory: URL = AppGroup.widget,
         reload: @escaping () -> Void = { WidgetCenter.shared.reloadAllTimelines() }) {
        self.hosts = hosts
        self.library = library
        self.queue = queue
        self.generate = generate
        self.thumbnails = thumbnails
        self.directory = directory
        self.reload = reload
    }

    /// The snapshot as the stores stand, without touching the disk.
    func snapshot(now: Date = .now) -> WidgetSnapshot {
        let recent = library.pool.filter { $0.print.trashedAt == nil }
            .sorted { $0.createdAt > $1.createdAt }
            .prefix(Self.printLimit)
        let listings = queue.listings.values.joined()
        return WidgetSnapshot(
            updated: now,
            prints: recent.map { entry in
                WidgetSnapshot.Print(
                    host: entry.hostID, machine: entry.hostName, filename: entry.print.filename,
                    title: entry.spokenName, image: Self.imageName(entry), favourite: entry.print.isFavorite,
                    kind: entry.print.kind == .mesh ? .mesh : entry.print.kind == .clip ? .clip : .picture,
                    made: entry.createdAt)
            },
            machines: hosts.hosts.map { .init(id: $0.id, name: $0.name) },
            rendering: listings.filter { $0.state == .running }.count,
            held: listings.filter { $0.state == .held }.count,
            waiting: listings.filter { $0.state == .queued || $0.state == .paused }.count,
            // Only while the app is in front is the render's progress live.
            progress: UIApplication.shared.applicationState == .active
                ? generate.run.steps.map { Double($0.done) / Double($0.total) } : nil)
    }

    /// A stable file name per print and version, safe for any filename.
    static func imageName(_ entry: LibraryEntry) -> String {
        let key = "\(entry.hostID.uuidString)|\(entry.print.filename)|\(entry.print.mediaVersion ?? "")"
        let digest = key.utf8.reduce(UInt64(14_695_981_039_346_656_037)) { ($0 ^ UInt64($1)) &* 1_099_511_628_211 }
        return String(digest, radix: 16) + ".jpg"
    }

    func refresh() async {
        let next = snapshot()
        for print in next.prints where !FileManager.default.fileExists(atPath: file(print.image).path) {
            guard let entry = library.pool.first(where: { $0.hostID == print.host && $0.print.filename == print.filename }),
                  let image = await thumbnails.image(for: entry, pixels: Self.pixels)?.cgImage else { continue }
            write(image, to: file(print.image))
        }
        // Thumbnails no print names any more.
        let keep = Set(next.prints.map(\.image) + ["snapshot.json"])
        for name in (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? [] where !keep.contains(name) {
            try? FileManager.default.removeItem(at: file(name))
        }
        let before = WidgetSnapshot.load(from: file("snapshot.json"))
        var compare = next
        compare.updated = before.updated
        guard compare != before else { return }
        try? next.save(to: file("snapshot.json"))
        reload()
    }

    private func file(_ name: String) -> URL { directory.appending(path: name) }

    private func write(_ image: CGImage, to url: URL) {
        guard let out = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(out, image, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        CGImageDestinationFinalize(out)
    }
}
