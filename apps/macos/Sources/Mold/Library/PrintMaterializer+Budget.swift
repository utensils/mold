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
    var contents: [CacheBudget.File] {
        let manager = FileManager.default
        let keys: Set<URLResourceKey> = [.contentAccessDateKey, .contentModificationDateKey,
                                         .fileSizeKey]
        let folders = (try? manager.contentsOfDirectory(
            at: cacheRoot, includingPropertiesForKeys: Array(keys))) ?? []
        return folders.compactMap { folder in
            let files = (try? manager.contentsOfDirectory(
                at: folder, includingPropertiesForKeys: Array(keys))) ?? []
            guard let file = files.first,
                  let values = try? file.resourceValues(forKeys: keys),
                  let bytes = values.fileSize
            else { return nil }
            // Modification date FIRST, because `touch` is what maintains the
            // recency this is meant to read and modification is all it writes.
            // Preferring the access date meant APFS's own answer -- which the
            // OS updates for its own reasons, and never for ours -- decided
            // the eviction order, and the LRU's hand-kept signal was never
            // read at all.
            let used = values.contentModificationDate ?? values.contentAccessDate ?? .distantPast
            return CacheBudget.File(name: folder.lastPathComponent, bytes: bytes, lastUsed: used)
        }
    }

    var usedBytes: Int { contents.reduce(0) { $0 + $1.bytes } }

    /// Evicts down to the cap, sparing what is in use.
    ///
    /// `keeping` is the print just written: this runs between the write and
    /// the return, so a clip bigger than the whole cap used to be downloaded,
    /// written, deleted, and its URL handed back -- after which Quick Look
    /// showed an empty panel, a save wrote nothing through its `try?`, and the
    /// drag reported a generic failure. Nothing was told. A file too big to
    /// keep still goes, but AFTER the thing that asked for it has had it, and
    /// the person is told why it will not be there next time.
    ///
    /// Whatever `inUse` names is spared for the same reason: Quick Look reads
    /// its item's URL lazily. A drag promise needs no entry there -- the
    /// Finder holds an open descriptor, which an unlink does not invalidate.
    func enforceBudget(keeping key: String? = nil) {
        var spared = Set(inUse().map {
            $0.deletingLastPathComponent().lastPathComponent
        })
        if let key { spared.insert(key) }
        for name in CacheBudget.evictions(from: contents, cap: capBytes)
        where !spared.contains(name) {
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
