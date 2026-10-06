import Foundation
import MoldClient

/// Byte-free recall information. It grants no authority until the origin is checked again.
nonisolated struct SavedReuse: Codable, Equatable {
    var version = 1
    let origin: UUID
    let instance: String?
    let filename: String
    let model: String?
    let recipe: String?
    let metadata: OutputMetadata
    var archive: String?
    var output: String?
    var invalidated = false
}

nonisolated struct SavedReuseFile: Sendable {
    let url: URL
    init(directory: URL = SecretStore.applicationSupport()) {
        url = directory.appending(path: "generate-retained-recipe.json")
    }
    var exists: Bool { FileManager.default.fileExists(atPath: url.path) }
    func load() -> SavedReuse? {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              size <= 1024 * 1024, let text = try? String(contentsOf: url, encoding: .utf8),
              let saved = try? MoldJSON.localDecoder.decode(SavedReuse.self, from: Data(text.utf8)) else { return nil }
        return saved
    }
    func save(_ saved: SavedReuse?) {
        guard let saved else { try? FileManager.default.removeItem(at: url); return }
        guard let data = try? MoldJSON.localEncoder.encode(saved), data.count <= 1024 * 1024 else { return }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: [.atomic, .completeFileProtection])
    }
}
