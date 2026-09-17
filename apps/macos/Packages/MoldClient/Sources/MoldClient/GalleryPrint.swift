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
    ///
    /// **Known limit: an animated WebP brought in through `GalleryImport`
    /// reads as a still.** There is no honest signal for it anywhere on the
    /// wire: the import descriptor carries no `frames` (nothing on this Mac
    /// counts them, and inventing a number would be worse than reading none),
    /// and the host's own `format_from_path` (`metadata_io.rs:36-56`) maps the
    /// extension to `Webp` without opening the file. A GIF or an MP4 imported
    /// the same way is classified correctly, because for those the container
    /// IS the answer. Closing this needs a frame count at the import call
    /// site, which is a change to what `GalleryImport` sends, not to this
    /// rule.
    public var isVideo: Bool {
        guard let format else { return false }
        if Self.animatedFormats.contains(format) { return true }
        return format == "webp" && (metadata.frames ?? 1) > 1
    }

    /// mold stores every 3-D artifact as one GLB.
    public var isMesh: Bool { format == "glb" }

    /// What a person would call this.
    public var kind: PrintKind {
        // No `format` (an older host's listing): the name is all there is,
        // and the one extension table answers for it.
        if format == nil { return PrintKind(playbackOf: filename) }
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

// In an extension rather than in the body above, so the memberwise
// initializer every construction site uses survives.
public extension GalleryPrint {
    /// Refuses a filename that is not a single safe path component.
    ///
    /// This app writes that name into its media cache and `removeItem`s at the
    /// same path on a save, unsandboxed -- so the check belongs at the door,
    /// where a hostile name never becomes a `GalleryPrint` at all, rather than
    /// at each of the places that later builds a path out of one. Decoding a
    /// LISTING drops such a row and keeps the rest; see `GalleryListing`.
    init(from decoder: any Decoder) throws {
        let row = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            filename: try SafeFilename.validated(row.decode(String.self, forKey: .filename)),
            metadata: try row.decode(OutputMetadata.self, forKey: .metadata),
            timestamp: try row.decode(UInt64.self, forKey: .timestamp),
            format: try row.decodeIfPresent(String.self, forKey: .format),
            sizeBytes: try row.decodeIfPresent(Int.self, forKey: .sizeBytes),
            mediaVersion: try row.decodeIfPresent(String.self, forKey: .mediaVersion),
            title: try row.decodeIfPresent(String.self, forKey: .title),
            tags: try row.decodeIfPresent([String].self, forKey: .tags),
            favorite: try row.decodeIfPresent(Bool.self, forKey: .favorite),
            collections: try row.decodeIfPresent([String].self, forKey: .collections),
            trashedAt: try row.decodeIfPresent(UInt64.self, forKey: .trashedAt),
            purgeAt: try row.decodeIfPresent(UInt64.self, forKey: .purgeAt)
        )
    }
}
