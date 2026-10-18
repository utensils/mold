import Foundation

/// Read-only authoring inspection. Nothing here conforms, fits or attaches media.
public enum InputMediaPreview {
    public static func image(original: String?, submitted: String?) -> String? { original ?? submitted }

    public static func file(base64: String) async throws -> SourceMediaFile {
        try await work {
            try Task.checkCancellation()
            guard base64.utf8.count <= (SourceMediaFile.previewLimit + 2) / 3 * 4,
                  let bytes = Data(base64Encoded: base64) else { throw MoldClientError.malformedResponse }
            return try stage(bytes)
        }
    }
    public static func file(bytes: Data) async throws -> SourceMediaFile {
        try await work { try stage(bytes) }
    }
    public static func file(url: URL) async throws -> SourceMediaFile {
        try await work {
            let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            try SourceMediaFile.checkPreviewSize(size)
            return try stage(Data(contentsOf: url, options: .mappedIfSafe))
        }
    }
    private static func work(_ operation: @escaping @Sendable () throws -> SourceMediaFile) async throws -> SourceMediaFile {
        try Task.checkCancellation()
        let worker = Task.detached(priority: .utility, operation: operation)
        return try await withTaskCancellationHandler {
            let file = try await worker.value
            try Task.checkCancellation()
            return file
        } onCancel: { worker.cancel() }
    }
    private static func stage(_ bytes: Data) throws -> SourceMediaFile {
        try Task.checkCancellation()
        try SourceMediaFile.checkPreviewSize(bytes.count)
        let file = try SourceMediaFile(extension: "")
        try bytes.write(to: file.url)
        try Task.checkCancellation()
        return file
    }
}

public extension InputMediaPreview {
    static func retainedFile(reference: AuthoringReference, index: Int, references: [AuthoringReference],
                             original: [AuthoringReference], members: [RetainedSourceMedia.Member],
                             filename: String, backend: any MoldBackend) async throws -> SourceMediaFile {
        guard let reference = reference.canonicalReference,
              let references = references.canonicalReferences, let original = original.canonicalReferences else {
            throw MoldClientError.malformedResponse
        }
        return try await retainedFile(reference: reference, index: index, references: references,
            original: original, members: members, filename: filename, backend: backend)
    }
    static func retainedFile(reference: GenerationReference, index: Int, references: [GenerationReference],
                             original: [GenerationReference], members: [RetainedSourceMedia.Member],
                             filename: String, backend: any MoldBackend) async throws -> SourceMediaFile {
        guard references.indices.contains(index), references[index] == reference,
              RetainedReferenceGuard.canHydrate(references: references, original: original, members: members) else {
            throw MoldClientError.malformedResponse
        }
        let ordered = members.filter { $0.role == "references" }
        guard ordered.indices.contains(index) else { throw MoldClientError.malformedResponse }
        let member = ordered[index]
        let inventory = try await backend.retainedSourceMedia(for: filename)
        guard inventory.availability == .available, inventory.members.contains(member) else { throw MoldClientError.malformedResponse }
        return try await backend.retainedSourceMediaFile(for: filename, member: member)
    }
}

/// Captured route authority for a read-only picker presentation.
public struct InputMediaPreviewAuthority: Hashable, Sendable {
    public let host: MoldHost
    public let instance: String?
    public init(host: MoldHost, instance: String?) { self.host = host; self.instance = instance }
    public func isCurrent(host: MoldHost?, instance: String?) -> Bool {
        host == self.host && instance == self.instance
    }
}
