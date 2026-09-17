import Foundation
import MoldClient

// Keeping the cache inside its budget. Split from the fetching half for size.
@MainActor
extension PrintMaterializer {

    /// Everything the cache is holding, as `CacheBudget` wants to see it.
    ///
    /// One directory per print, so the directory is the unit that gets
    /// evicted -- not the file inside it, which would leave an empty folder
    /// behind and an entry that looks present until something tries to read it.
    ///
    /// The whole folder is summed, because a folder is NOT one print: the key
    /// excludes the filename (it is the host and the folded `media_version`,
    /// or the timestamp where a host sends none), so a batch published in one
    /// second shares one. Reporting `files.first`'s size meant the cache
    /// under-reported by the size of the batch, "Using" in Settings said the
    /// same wrong number, and the cap was never reached -- on exactly the
    /// hosts the timestamp fallback exists for.
    ///
    /// A folder with nothing measurable in it -- what a failed write leaves
    /// behind -- is size 0 rather than absent, so eviction can still see it.
    var contents: [CacheBudget.File] {
        let manager = FileManager.default
        let keys: Set<URLResourceKey> = [.contentAccessDateKey, .contentModificationDateKey,
                                         .fileSizeKey]
        let folders = (try? manager.contentsOfDirectory(
            at: cacheRoot, includingPropertiesForKeys: Array(keys))) ?? []
        return folders.map { folder in
            let files = (try? manager.contentsOfDirectory(
                at: folder, includingPropertiesForKeys: Array(keys))) ?? []
            let values = files.compactMap { try? $0.resourceValues(forKeys: keys) }
            // Modification date FIRST, because `touch` is what maintains the
            // recency this is meant to read and modification is all it writes.
            // Preferring the access date meant APFS's own answer -- which the
            // OS updates for its own reasons, and never for ours -- decided
            // the eviction order, and the LRU's hand-kept signal was never
            // read at all. The NEWEST date in the folder speaks for it: one
            // print in it being read is the folder being used.
            let used = values.compactMap { $0.contentModificationDate ?? $0.contentAccessDate }
                .max() ?? .distantPast
            return CacheBudget.File(name: folder.lastPathComponent,
                                    bytes: values.compactMap(\.fileSize).reduce(0, +),
                                    lastUsed: used)
        }
    }

    var usedBytes: Int { contents.reduce(0) { $0 + $1.bytes } }

    /// Evicts down to the cap, sparing what is in use.
    ///
    /// `keeping` is the print just written. This runs between the write and
    /// the return, so a clip bigger than the whole cap used to be downloaded,
    /// written, deleted, and its URL handed back -- after which Quick Look
    /// showed an empty panel, a save wrote nothing through its `try?`, and the
    /// drag reported a generic failure. Nothing said why. It now survives this
    /// pass, so the thing that asked for it gets it, and goes on a later one
    /// when nothing is holding it -- with `note` saying it will not be kept.
    ///
    /// Whatever `inUse` names is spared for the same reason: Quick Look reads
    /// its item's URL lazily. A drag promise needs no entry there -- the
    /// Finder holds an open descriptor, which an unlink does not invalidate.
    /// Spared bytes come OFF the budget rather than out of the reckoning, so
    /// the rest of the cache is still trimmed to what is left of the cap
    /// instead of the whole thing sitting over it.
    func enforceBudget(keeping key: String? = nil) {
        var spared = Set(inUse().map {
            $0.deletingLastPathComponent().lastPathComponent
        })
        // A folder being written into right now is empty for as long as the
        // download takes, and sweeping it would leave the write with nowhere
        // to land.
        spared.formUnion(inFlightKeys)
        if let key { spared.insert(key) }

        let holdings = contents
        let held = holdings.filter { spared.contains($0.name) }.reduce(0) { $0 + $1.bytes }
        let evictable = holdings.filter { !spared.contains($0.name) }
        var doomed = Set(CacheBudget.evictions(from: evictable,
                                               cap: Swift.max(capBytes - held, 0)))
        // Nothing measurable in it is not a budget question: it is rubbish,
        // and it was invisible to both halves of this function before.
        doomed.formUnion(evictable.filter { $0.bytes == 0 }.map(\.name))
        for name in doomed {
            try? FileManager.default.removeItem(at: cacheRoot.appending(path: name))
        }
    }

    /// Says so when a print cannot be kept, instead of letting it disappear.
    func noteIfTooLarge(_ file: URL, named name: String) {
        let bytes = (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        guard bytes > capBytes else { return }
        let size = ByteCountFormatStyle().format(Int64(bytes))
        let cap = ByteCountFormatStyle().format(Int64(capBytes))
        note = "“\(name)” is \(size) and the media cache holds \(cap), "
            + "so Mold cannot keep a copy. Settings ▸ General sets the cap."
    }

    /// Marks a file as used now, so the least-recently-used rule has something
    /// to go on. macOS does not reliably update access times by itself.
    func touch(_ file: URL) {
        try? FileManager.default.setAttributes([.modificationDate: Date()],
                                               ofItemAtPath: file.path)
    }
}
