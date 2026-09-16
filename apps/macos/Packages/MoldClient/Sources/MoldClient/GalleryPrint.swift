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

    /// mold stores video as mp4 and meshes as glb; everything else is a still.
    public var isVideo: Bool { ["mp4", "webm", "mov"].contains(format ?? "") }
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
    ///
    /// Public because anything OFFERING a filter has to fold the same way the
    /// filter itself does, or a suggestion appears that then matches nothing.
    public static func fold(_ text: String) -> String {
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
