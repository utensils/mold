import Foundation

/// Editing one print.
public struct GalleryPatch: Codable, Sendable {
    public var title: String?
    public var favorite: Bool?
    public var addTags: [String]?
    public var removeTags: [String]?

    public init(title: String? = nil, favorite: Bool? = nil,
                addTags: [String]? = nil, removeTags: [String]? = nil) {
        self.title = title
        self.favorite = favorite
        self.addTags = addTags
        self.removeTags = removeTags
    }
}

/// A collection named in a request.
///
/// The app ALWAYS sends a name. The host resolves it by slug and creates it
/// when it has never seen it, which is what makes one request work against
/// every machine in a fleet -- an id is only ever right on one of them, and
/// sending one is how a single shelf becomes two.
public struct CollectionRef: Codable, Hashable, Sendable {
    public let name: String?

    public static func named(_ name: String) -> CollectionRef { CollectionRef(name: name) }
}

/// Editing several prints at once, replay-safely.
///
/// `operationId` is the fence: the host applies a given id once, so a retry
/// after a dropped connection cannot double-apply a change. Mint it ONCE per
/// intended change and reuse it for every attempt -- a fresh id per attempt is
/// exactly the double-apply the fence exists to prevent.
public struct GalleryBulkMutation: Codable, Sendable {
    public let operationId: String
    public let filenames: [String]
    public var favorite: Bool?
    public var addTags: [String]
    public var removeTags: [String]
    /// Ensure this collection exists on the serving host, then add every
    /// filename to it.
    public var addToCollection: CollectionRef?
    /// Removal names the slug, because that is the identity the app and the
    /// host agree on.
    public var removeFromCollectionSlug: String?

    public init(filenames: [String], favorite: Bool? = nil,
                addTags: [String] = [], removeTags: [String] = [],
                addToCollection: CollectionRef? = nil,
                removeFromCollectionSlug: String? = nil,
                operationId: String = UUID().uuidString) {
        self.operationId = operationId
        self.filenames = filenames
        self.favorite = favorite
        self.addTags = addTags
        self.removeTags = removeTags
        self.addToCollection = addToCollection
        self.removeFromCollectionSlug = removeFromCollectionSlug
    }
}

public struct TrashRequest: Codable, Sendable {
    public let filenames: [String]
    public init(filenames: [String]) { self.filenames = filenames }
}

/// A named group of prints, as ONE machine holds it. `CollectionShelf.merge`
/// is what turns several of these into the one shelf a person sees.
public struct Collection: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let slug: String
    public let description: String?
    /// The tile to show. Absent lets the app pick the newest member.
    public let coverFilename: String?
    /// Trashed members are still counted -- they keep their membership until
    /// they are purged, so a restored print returns to its shelf.
    public let count: Int?
    public let hidden: Bool?

    public init(id: String, name: String, slug: String, description: String? = nil,
                coverFilename: String? = nil, count: Int? = nil, hidden: Bool? = nil) {
        self.id = id
        self.name = name
        self.slug = slug
        self.description = description
        self.coverFilename = coverFilename
        self.count = count
        self.hidden = hidden
    }
}

public struct TagCount: Codable, Hashable, Sendable, Identifiable {
    public let name: String
    public let count: Int
    public var id: String { name }

    public init(name: String, count: Int) {
        self.name = name
        self.count = count
    }
}

/// What a host will convert a stored print into.
///
/// One flat list covering both kinds: `gif`/`apng`/`webp` are what a clip
/// becomes, and `obj`/`stl`/`ply`/`zip` are what a mesh becomes — a mesh can
/// also become an animated turntable, which is why the animated formats are
/// not video-only.
public struct ExportOptions: Codable, Hashable, Sendable {
    public let formats: [String]

    private static let animated: Set<String> = ["gif", "apng", "webp"]
    private static let geometry: Set<String> = ["obj", "stl", "ply", "zip"]

    /// What a clip can be turned into.
    public var forVideo: [String] { formats.filter(Self.animated.contains) }

    /// What a mesh can be turned into: geometry files, plus a turntable.
    public var forMesh: [String] {
        formats.filter { Self.geometry.contains($0) || Self.animated.contains($0) }
    }
}
