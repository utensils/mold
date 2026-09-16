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

    /// mold stores video as mp4 and meshes as glb; everything else is a still.
    public var isVideo: Bool { ["mp4", "webm", "mov"].contains(format ?? "") }
    public var isMesh: Bool { format == "glb" }
}

/// A print's identity in a library merged across machines.
///
/// A filename alone is NOT an identity: two hosts generate names from the same
/// scheme and will collide, and mold's own rule is that a print belongs to the
/// machine that made it.
public struct PrintID: Hashable, Codable, Sendable {
    public let host: MoldHost.ID
    public let filename: String

    public init(host: MoldHost.ID, filename: String) {
        self.host = host
        self.filename = filename
    }
}

/// A print paired with the machine that owns it -- what the merged Library
/// actually holds.
public struct LibraryEntry: Identifiable, Hashable, Sendable {
    public let hostID: MoldHost.ID
    public let hostName: String
    public let print: GalleryPrint
    /// Everything searchable, folded once at construction.
    ///
    /// A library holds thousands of prints and the search field filters on
    /// every keystroke; folding each row's text again per keystroke is work
    /// proportional to the library, repeated for every character typed.
    public let searchKey: String

    public init(hostID: MoldHost.ID, hostName: String, print: GalleryPrint) {
        self.hostID = hostID
        self.hostName = hostName
        self.print = print
        self.searchKey = Self.fold([
            print.metadata.prompt, print.metadata.model, print.metadata.family,
            print.title, print.filename, hostName,
            print.metadata.seed.map(String.init),
        ].compactMap(\.self).joined(separator: " ") + " " + print.tagList.joined(separator: " "))
    }

    public var id: PrintID { PrintID(host: hostID, filename: print.filename) }
    public var createdAt: Date { print.createdAt }

    /// Case-, diacritic- and width-insensitive, so "cafe" finds "Café".
    static func fold(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
                     locale: .current)
    }

    /// Every whitespace-separated token must appear, so more words narrow.
    public func matches(_ query: String) -> Bool {
        let tokens = Self.fold(query).split(separator: " ")
        guard !tokens.isEmpty else { return true }
        return tokens.allSatisfy { searchKey.contains($0) }
    }
}

public extension GalleryPrint {
    /// A print with the few fields a client may change locally.
    ///
    /// `GalleryPrint` is a wire type and stays immutable; this exists so an
    /// optimistic update can turn a star on without inventing a second model
    /// of what a print is.
    struct Mutable {
        public var favorite: Bool?
        public var tags: [String]?
        public var title: String?
        private let base: GalleryPrint

        public init(_ print: GalleryPrint) {
            self.base = print
            self.favorite = print.favorite
            self.tags = print.tags
            self.title = print.title
        }

        public func build() -> GalleryPrint {
            GalleryPrint(
                filename: base.filename, metadata: base.metadata, timestamp: base.timestamp,
                format: base.format, sizeBytes: base.sizeBytes, mediaVersion: base.mediaVersion,
                title: title, tags: tags, favorite: favorite, collections: base.collections,
                trashedAt: base.trashedAt, purgeAt: base.purgeAt)
        }
    }
}
