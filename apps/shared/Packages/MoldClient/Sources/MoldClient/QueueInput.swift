import Foundation

/// Sealed conditioning descriptors, never filenames used as fetch authority.
public struct QueueInput: Codable, Hashable, Sendable, Identifiable {
    public let index: Int?
    public let label: String
    public let preview: Bool
    public var id: String { index.map(String.init) ?? "legacy" }

    public init(index: Int? = nil, label: String, preview: Bool) {
        self.index = index; self.label = label; self.preview = preview
    }
}

public struct QueueInputPreview: Sendable, Identifiable {
    public let input: QueueInput
    public let bytes: Data?
    public var id: String { input.id }
    public init(input: QueueInput, bytes: Data?) { self.input = input; self.bytes = bytes }
}

public extension MoldQueueBackend {
    /// Load each member independently: a missing preview does not hide later references.
    func queueInputPreviews(id: String, firstOnly: Bool = false, cached: [QueueInputPreview] = [], maximumBytes: Int = .max) async throws -> [QueueInputPreview] {
        let inputs = try await queueInputs(id: id)
        guard inputs.count <= 256, Set(inputs.map(\.id)).count == inputs.count, inputs.allSatisfy({ ($0.index ?? 0) >= 0 && $0.label.utf8.count <= 1024 }) else { throw MoldClientError.malformedResponse }
        var previews: [QueueInputPreview] = []
        var foundImage = false
        var remaining = max(0, maximumBytes)
        for input in inputs {
            try Task.checkCancellation()
            var bytes: Data?
            if let existing = cached.first(where: { $0.input == input })?.bytes {
                bytes = existing
            } else if input.preview && remaining > 0 && (!firstOnly || !foundImage) {
                do { bytes = try await queueInputThumbnail(id: id, index: input.index) }
                catch is CancellationError { throw CancellationError() }
                catch { bytes = nil }
            } else { bytes = nil }
            try Task.checkCancellation()
            if let loaded = bytes, loaded.count > remaining { bytes = nil }
            remaining -= bytes?.count ?? 0
            if bytes != nil { foundImage = true }
            if input.index == nil && bytes == nil { continue }
            previews.append(QueueInputPreview(input: input, bytes: bytes))
        }
        return previews
    }
}
