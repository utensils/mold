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
    /// One sentence when a print could not be kept -- read by the Library, so
    /// "too large for the cache" is said out loud rather than presenting as a
    /// preview that never opens. `internal(set)`: `+Budget` writes it.
    internal(set) var note: String?
    /// Two askers for the same print share one download.
    private var inFlight: [String: Task<URL?, Never>] = [:]

    /// Files something is reading right now, which eviction must spare.
    ///
    /// The Quick Look panel by default -- it reads its item's URL lazily, from
    /// its own queues, so deleting the directory under it leaves an empty
    /// panel. A parameter because a test should not have to open one.
    @ObservationIgnored let inUse: @MainActor () -> [URL]

    init(root: URL? = nil, inUse: @escaping @MainActor () -> [URL] = { QuickLook.shared.heldURLs }) {
        self.cacheRoot = root ?? Self.defaultRoot
        self.inUse = inUse
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
        // Both components are resolved through `SafeFilename`, which proves the
        // result is still inside the directory it was built from. The name was
        // already refused at the decode; this is the belt for that brace, and
        // it is what makes "server string, then `write(to:)`" untrue of this
        // function whatever else changes upstream of it.
        guard let folder = SafeFilename.url(key, in: cacheRoot),
              let file = SafeFilename.url(entry.print.filename, in: folder)
        else { return nil }
        if FileManager.default.fileExists(atPath: file.path) {
            touch(file)
            return file
        }
        // The filename rides along: `key` alone is (host, media_version), and
        // two different prints that fall back to the same timestamp version
        // used to coalesce onto one download and hand one of them the
        // other's file.
        let flightKey = "\(key)/\(entry.print.filename)"
        if let running = inFlight[flightKey] { return await running.value }

        let task = Task<URL?, Never> { [file, folder] in
            guard let data = await fetch() else { return nil }
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            guard (try? data.write(to: file)) != nil else { return nil }
            return file
        }
        inFlight[flightKey] = task
        let url = await task.value
        inFlight[flightKey] = nil
        // After, never before: evicting to make room for a file whose size is
        // still unknown would either be a guess or a second round trip. And
        // `keeping:` what was just written, or a print larger than the cap is
        // deleted here and its URL handed back to a caller that then fails
        // silently.
        if let url { noteIfTooLarge(url, named: entry.print.displayName) }
        enforceBudget(keeping: key)
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
        // `media_version` is the machine's string too, and this is the only
        // place the app makes a DIRECTORY out of one -- so it is folded rather
        // than refused (a print with an odd version is still a print) by the
        // same rule the filename is judged against. mold's own versions carry
        // a colon, which is legal in a POSIX component and invisible here but
        // which the Finder renders as "/", so it goes either way.
        let safe = SafeFilename.folded(version, fallback: String(entry.print.timestamp))
        return "\(entry.hostID.uuidString)-\(safe)"
    }

    /// Empties the whole cache. Called when Mold quits.
    func purge() {
        try? FileManager.default.removeItem(at: cacheRoot)
        try? FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
    }
}
