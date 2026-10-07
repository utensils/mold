import Foundation
import CryptoKit
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

    /// A matching locator alone cannot authorize reordered or changed remote
    /// descriptor slots. Locally held inline media remains explicit input.
    func acceptsSnapshot(_ inputs: DraftInputSnapshot, model: String?, recipe: String?) -> Bool {
        guard version == 1, !invalidated, self.model == model, self.recipe == recipe,
              inputs.retainedReuseFingerprint == fingerprint else { return false }
        let references = inputs.active.generationReferences
        return !references.contains { $0.media.authority == "descriptor" }
            || references == RenderDraft(reusing: metadata).media.generationReferences
    }

    /// Binds a local input snapshot to this exact byte-free recall context.
    var fingerprint: String? {
        guard let value = try? MoldJSON.localEncoder.encode(self),
              let object = try? JSONSerialization.jsonObject(with: value),
              let canonical = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else { return nil }
        return SHA256.hash(data: canonical).map { String(format: "%02x", $0) }.joined()
    }

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
