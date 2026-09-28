import Foundation

/// Which shelf of the library is showing. Shared by both apps; each keeps its
/// own empty-shelf copy, because the ways in differ by device.
///
/// A collection is named by its SLUG, never by an id: an id belongs to one
/// machine, and this is remembered across launches and shared by machines that
/// come and go. The shelf itself is looked up when it is needed.
public enum LibraryScope: Hashable, Identifiable, Codable, Sendable {
    case all
    case favorites
    case collection(slug: String)
    case trash

    public var id: Self { self }

    /// The three fixed shelves. Collections are appended from the store.
    public static let fixed: [LibraryScope] = [.all, .favorites, .trash]

    public func title(in shelves: [CollectionShelf]) -> String {
        switch self {
        case .all: "All Prints"
        case .favorites: "Favourites"
        case .trash: "Recently Deleted"
        case let .collection(slug):
            shelves.first { $0.slug == slug }?.name ?? "Collection"
        }
    }

    public var symbol: String {
        switch self {
        case .all: "photo.on.rectangle.angled"
        case .favorites: "star"
        case .collection: "rectangle.stack"
        case .trash: "trash"
        }
    }

    /// Trash is a different LISTING on the host, not a filter over the live
    /// one -- which is why it cannot be expressed as a search token.
    public var isTrash: Bool { self == .trash }

    public var collectionSlug: String? {
        if case let .collection(slug) = self { return slug }
        return nil
    }

    /// The narrowing this shelf adds on top of whatever was searched for.
    public func token(in shelves: [CollectionShelf]) -> LibraryToken? {
        switch self {
        case .favorites:
            .favorite
        case let .collection(slug):
            shelves.first { $0.slug == slug }.map {
                .collection(slug: $0.slug, name: $0.name, ids: $0.hosts)
            }
        case .all, .trash:
            nil
        }
    }
}
