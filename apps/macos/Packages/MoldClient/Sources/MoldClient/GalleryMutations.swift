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

/// Editing several prints at once, replay-safely.
///
/// `operationId` is the fence: the host applies a given id once, so a retry
/// after a dropped connection cannot double-apply a change.
public struct GalleryBulkMutation: Codable, Sendable {
    public let operationId: String
    public let filenames: [String]
    public var favorite: Bool?
    public var addTags: [String]
    public var removeTags: [String]

    public init(filenames: [String], favorite: Bool? = nil,
                addTags: [String] = [], removeTags: [String] = [],
                operationId: String = UUID().uuidString) {
        self.operationId = operationId
        self.filenames = filenames
        self.favorite = favorite
        self.addTags = addTags
        self.removeTags = removeTags
    }
}

public struct TrashRequest: Codable, Sendable {
    public let filenames: [String]
    public init(filenames: [String]) { self.filenames = filenames }
}

/// A named group of prints. Collections merge across machines by `slug`, so
/// the same name on two hosts is one shelf.
public struct Collection: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let slug: String
    public let count: Int?
    public let hidden: Bool?
}

public struct TagCount: Codable, Hashable, Sendable, Identifiable {
    public let name: String
    public let count: Int
    public var id: String { name }
}

/// What a host will convert a stored print into.
public struct ExportOptions: Codable, Hashable, Sendable {
    public let image: [String]?
    public let video: [String]?
    public let mesh: [String]?
}
