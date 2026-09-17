import Foundation

/// How a print was made. mold records far more than this; these are the fields
/// the app shows or reuses.
///
/// Every field but `prompt` and `model` is optional on purpose: this metadata
/// spans years of mold versions, and a print made before a field existed is a
/// normal print, not a corrupt one.
public struct OutputMetadata: Codable, Hashable, Sendable {
    public let prompt: String?
    public let negativePrompt: String?
    public let model: String?
    public let family: String?
    public let seed: UInt64?
    public let steps: Int?
    public let guidance: Double?
    public let width: Int?
    public let height: Int?
    /// What was actually rendered, when it differs from the delivered size --
    /// an upscaled print's `width` is the final one, not the one to reuse.
    public let generationWidth: Int?
    public let generationHeight: Int?
    public let frames: Int?
    public let fps: Double?
    public let generationTimeMs: Int?
    public let jobId: String?
    public let outputFormat: String?
    /// `one-shot` or `sequence`. A sequence's `prompt` is every stage joined by
    /// newlines, which is why reuse must never restore it wholesale.
    public let outputMode: String?
    public let chainJobId: String?
}

/// One finished piece of work on one host.
public struct GalleryPrint: Codable, Hashable, Sendable {
    public let filename: String
    public let metadata: OutputMetadata
    /// Unix seconds.
    public let timestamp: UInt64
    public let format: String?
    public let sizeBytes: Int?
    /// Changes when the bytes change. It is the cache key for thumbnails and
    /// the basis of the ETag, so a re-rendered poster invalidates cleanly.
    public let mediaVersion: String?
    public let title: String?
    public let tags: [String]?
    public let favorite: Bool?
    public let collections: [String]?
    public let trashedAt: UInt64?
    public let purgeAt: UInt64?

    public var createdAt: Date { Date(timeIntervalSince1970: TimeInterval(timestamp)) }
    public var isFavorite: Bool { favorite ?? false }
    public var tagList: [String] { tags ?? [] }

    /// Containers mold only ever writes for something that MOVES. `format` is
    /// the serialized `OutputFormat` (`types.rs:3667-3689`), not a file
    /// extension, so `apng` arrives spelled out even though the file on disk
    /// is a `.png`. `webm` and `mov` used to be in this set and are formats
    /// mold has never produced (`metadata_io.rs:36-56` is the closed set).
    private static let animatedFormats: Set<String> = ["mp4", "gif", "apng"]

    /// Whether this print moves.
    ///
    /// The container answers for every format but one. `webp` is offered for a
    /// still recipe AND for a temporal one (`generation_profile.rs:2140-2157`),
    /// so a WebP alone says nothing and the print's own frame count decides.
    /// Picking GIF or WebP for an LTX-2 or Wan render is a one-click choice
    /// the inspector offers from `capabilities.output.formats`, so this is an
    /// ordinary print, not a corner case.
    public var isVideo: Bool {
        guard let format else { return false }
        if Self.animatedFormats.contains(format) { return true }
        return format == "webp" && (metadata.frames ?? 1) > 1
    }

    /// mold stores every 3-D artifact as one GLB.
    public var isMesh: Bool { format == "glb" }

    /// What a person would call this.
    public var kind: PrintKind {
        if isVideo { return .clip }
        if isMesh { return .mesh }
        return .picture
    }

    /// What to put under the tile, in Quick Look's title bar, and on the
    /// file a share hands over: the person's own title when they gave one,
    /// and otherwise the name the machine chose. An empty title is not a
    /// title -- it is a field someone cleared.
    public var displayName: String {
        if let title, !title.trimmingCharacters(in: .whitespaces).isEmpty { return title }
        return filename
    }

    /// The collections this print is in, as ids ON ITS OWN MACHINE.
    public var collectionList: [String] { collections ?? [] }
}
