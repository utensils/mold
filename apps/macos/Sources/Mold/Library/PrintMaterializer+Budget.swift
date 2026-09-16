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
            // Access date where the file system keeps one; modification date
            // otherwise, which `touch` is what keeps current.
            let used = values.contentAccessDate ?? values.contentModificationDate ?? .distantPast
            return CacheBudget.File(name: folder.lastPathComponent, bytes: bytes, lastUsed: used)
        }
    }

    var usedBytes: Int { contents.reduce(0) { $0 + $1.bytes } }

    func enforceBudget() {
        for name in CacheBudget.evictions(from: contents, cap: capBytes) {
            try? FileManager.default.removeItem(at: cacheRoot.appending(path: name))
        }
    }

    /// Marks a file as used now, so the least-recently-used rule has something
    /// to go on. macOS does not reliably update access times by itself.
    func touch(_ file: URL) {
        try? FileManager.default.setAttributes([.modificationDate: Date()],
                                               ofItemAtPath: file.path)
    }
}
