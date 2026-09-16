import Foundation
import MoldClient

/// A real file on disk for a print, fetched once.
///
/// Quick Look, `ShareLink`, Save to… and the Finder drag all need a file and
/// not bytes, and the bytes live on another machine. One cache serves all
/// four, keyed on `(machine, filename, media_version)` -- the same
/// `media_version` the server's own ETag is built from, so a re-rendered
/// poster or an upscaled print invalidates cleanly rather than serving
/// yesterday's picture forever.
///
/// The cache is emptied when Mold quits. It exists so that previewing a clip,
/// then sharing it, then dragging it out costs one download instead of three
/// -- not to be a second copy of the library on this disk.
@MainActor
@Observable
final class PrintMaterializer {
    /// How much disk the cache may use, in megabytes. Settable, because a
    /// clip is hundreds of megabytes and how much of that is affordable is
    /// not something this app can know.
    static let defaultCapMegabytes = 1_024
    static let capKey = "mediaCacheMegabytes"

    /// Where the cached files live. Internal so `+Budget` can walk it.
    let cacheRoot: URL
    /// Two askers for the same print share one download.
    private var inFlight: [String: Task<URL?, Never>] = [:]

    init(root: URL? = nil) {
        self.cacheRoot = root ?? Self.defaultRoot
        try? FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
    }

    private static var defaultRoot: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let bundle = Bundle.main.bundleIdentifier ?? "io.utensils.mold.native"
        return caches.appending(path: bundle).appending(path: "prints")
    }

    var capBytes: Int {
        let stored = AppStorageSuite.defaults.object(forKey: Self.capKey) as? Int
        return max(0, stored ?? Self.defaultCapMegabytes) * 1_024 * 1_024
    }

    /// A file for this print, downloading it if the cache has not got one.
    func url(for entry: LibraryEntry, fetch: @escaping () async -> Data?) async -> URL? {
        let key = Self.key(for: entry)
        let file = cacheRoot.appending(path: key).appending(path: entry.print.filename)
        if FileManager.default.fileExists(atPath: file.path) {
            touch(file)
            return file
        }
        if let running = inFlight[key] { return await running.value }

        let task = Task<URL?, Never> { [cacheRoot] in
            guard let data = await fetch() else { return nil }
            let folder = cacheRoot.appending(path: key)
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let written = folder.appending(path: entry.print.filename)
            guard (try? data.write(to: written)) != nil else { return nil }
            return written
        }
        inFlight[key] = task
        let url = await task.value
        inFlight[key] = nil
        // After, never before: evicting to make room for a file whose size is
        // still unknown would either be a guess or a second round trip.
        enforceBudget()
        return url
    }

    /// The identity of a print's bytes, as a directory name.
    ///
    /// The filename stays OUT of the key and inside the directory, so what
    /// Quick Look titles and what the Finder receives is the print's own name
    /// rather than a hash. `media_version` is absent on older servers; the
    /// timestamp stands in, which at worst re-downloads once.
    private static func key(for entry: LibraryEntry) -> String {
        let version = entry.print.mediaVersion ?? String(entry.print.timestamp)
        // mold's media versions carry a colon. Legal in a POSIX path component
        // and invisible here, but the Finder renders one as "/" -- so it goes,
        // rather than leaving a cache nobody can read the names of.
        let safe = version.replacingOccurrences(of: ":", with: "-")
        return "\(entry.hostID.uuidString)-\(safe)"
    }

    /// Empties the whole cache. Called when Mold quits.
    func purge() {
        try? FileManager.default.removeItem(at: cacheRoot)
        try? FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
    }
}
