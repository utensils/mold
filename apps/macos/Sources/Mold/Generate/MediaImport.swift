import Foundation

/// A file mold takes as OPAQUE bytes -- a continuation clip, conditioning
/// audio, a source video -- read and base64-encoded off the main actor.
///
/// Deliberately not `PictureImport`: that one conforms what it reads to a
/// format the engine's image decoders can read, and pushing a video or a
/// soundtrack through an image transcoder would destroy it. What the two
/// share is the reason they exist at all -- `Data(contentsOf:)` plus
/// `base64EncodedString()` for a 200 MB clip ran on the main thread from a
/// `View` method and froze the window (finding 02#10), and under this app's
/// MainActor-by-default isolation the only way off it is an explicit
/// `Task.detached`.
nonisolated enum MediaImport {
    /// What a well is holding: the bytes as they will be sent, and the name
    /// to show for them.
    struct File: Sendable {
        let base64: String
        let name: String
    }

    static func load(_ url: URL) async throws -> File {
        let data = try await bytes(of: url)
        return await Task.detached(priority: .userInitiated) {
            File(base64: data.base64EncodedString(), name: url.lastPathComponent)
        }.value
    }

    /// The bytes alone, for a caller that sends them as bytes -- a Library
    /// import, which hands a whole file to `GalleryImport`.
    static func bytes(of url: URL) async throws -> Data {
        try await Task.detached(priority: .userInitiated) {
            try Data(contentsOf: url)
        }.value
    }
}
