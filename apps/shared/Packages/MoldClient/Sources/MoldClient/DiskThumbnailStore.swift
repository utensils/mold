import CryptoKit
import Foundation

/// Thumbnails on disk, bounded, least-recently-used first out.
///
/// Keyed on (machine, filename, `media_version`, pixel size): the same
/// `media_version` the server builds its ETag from, so a re-rendered print
/// never shows a stale picture -- and a print WITHOUT a media version is never
/// stored here at all (the caller keeps it in memory only). The caps are the
/// phone's (`.claude/rules/mobile.md`): 4,000 items, 256 MiB, 2 MiB each. A
/// thumbnail over the per-item cap is refused rather than stored and evicted.
///
/// Access time is the file's modification date, touched on every read, so the
/// order survives a relaunch without an index file to keep consistent.
public actor DiskThumbnailStore {
    public struct Key: Hashable, Sendable {
        public let host: String
        public let filename: String
        public let mediaVersion: String
        public let size: Int

        public init(host: String, filename: String, mediaVersion: String, size: Int) {
            self.host = host
            self.filename = filename
            self.mediaVersion = mediaVersion
            self.size = size
        }
    }

    public let directory: URL
    public let maxItems: Int
    public let maxBytes: Int
    public let maxItemBytes: Int

    private struct Item { var bytes: Int; var used: Date }
    private var index: [String: Item]?
    private let files = FileManager.default

    public init(directory: URL, maxItems: Int = 4_000, maxBytes: Int = 256 << 20, maxItemBytes: Int = 2 << 20) {
        self.directory = directory
        self.maxItems = maxItems
        self.maxBytes = maxBytes
        self.maxItemBytes = maxItemBytes
    }

    public func data(for key: Key) -> Data? {
        let name = Self.name(for: key)
        guard var item = loaded()[name], let data = try? Data(contentsOf: url(name)) else { return nil }
        item.used = .now
        index?[name] = item
        try? files.setAttributes([.modificationDate: item.used], ofItemAtPath: url(name).path(percentEncoded: false))
        return data
    }

    /// `false` when refused (over the per-item cap) or unwritable.
    @discardableResult
    public func store(_ data: Data, for key: Key) -> Bool {
        guard data.count <= maxItemBytes else { return false }
        _ = loaded()
        let name = Self.name(for: key)
        do {
            try files.createDirectory(at: directory, withIntermediateDirectories: true)
            try data.write(to: url(name), options: .atomic)
        } catch { return false }
        index?[name] = Item(bytes: data.count, used: .now)
        trim()
        return true
    }

    /// Every thumbnail of one machine -- it was removed.
    public func evict(host: String) {
        remove { $0.hasPrefix(Self.hostPrefix(host)) }
    }

    /// Every size of one print on one machine -- it was deleted for good.
    public func evict(host: String, filename: String) {
        remove { $0.hasPrefix(Self.hostPrefix(host) + Self.digest(filename) + "-") }
    }

    public func purge() { remove { _ in true } }

    /// What the cache holds on disk, for Settings.
    public var totalBytes: Int { loaded().values.reduce(0) { $0 + $1.bytes } }

    public var totals: (count: Int, bytes: Int) {
        let all = loaded()
        return (all.count, all.values.reduce(0) { $0 + $1.bytes })
    }

    // MARK: - Internals

    private func loaded() -> [String: Item] {
        if let index { return index }
        var found: [String: Item] = [:]
        let keys: Set<URLResourceKey> = [.fileSizeKey, .contentModificationDateKey]
        for file in (try? files.contentsOfDirectory(at: directory, includingPropertiesForKeys: Array(keys))) ?? [] {
            let values = try? file.resourceValues(forKeys: keys)
            found[file.lastPathComponent] = Item(bytes: values?.fileSize ?? 0,
                                                 used: values?.contentModificationDate ?? .distantPast)
        }
        index = found
        return found
    }

    private func trim() {
        guard var all = index else { return }
        var bytes = all.values.reduce(0) { $0 + $1.bytes }
        let oldestFirst = all.sorted { $0.value.used < $1.value.used }.map(\.key)
        for name in oldestFirst where all.count > maxItems || bytes > maxBytes {
            bytes -= all[name]?.bytes ?? 0
            all[name] = nil
            try? files.removeItem(at: url(name))
        }
        index = all
    }

    private func remove(where matches: (String) -> Bool) {
        guard var all = Optional(loaded()) else { return }
        for name in all.keys where matches(name) {
            all[name] = nil
            try? files.removeItem(at: url(name))
        }
        index = all
    }

    private func url(_ name: String) -> URL { directory.appending(path: name) }

    /// `<host digest>-<file digest>-<version digest>-<size>`: flat, fixed-length
    /// parts, and a prefix per machine and per print for eviction.
    static func name(for key: Key) -> String {
        hostPrefix(key.host) + digest(key.filename) + "-" + digest(key.mediaVersion) + "-\(key.size)"
    }

    static func hostPrefix(_ host: String) -> String { digest(host) + "-" }

    static func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).prefix(10).map { String(format: "%02x", $0) }.joined()
    }
}
