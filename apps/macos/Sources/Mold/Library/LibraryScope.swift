import Foundation
import MoldClient

/// Which shelf of the library is showing.
///
/// A collection is named by its SLUG, never by an id: an id belongs to one
/// machine, and this is remembered across launches and shared by machines that
/// come and go. The shelf itself is looked up when it is needed.
enum LibraryScope: Hashable, Identifiable, Codable {
    case all
    case favorites
    case collection(slug: String)
    case trash

    var id: Self { self }

    /// The three fixed shelves. Collections are appended from the store.
    static let fixed: [LibraryScope] = [.all, .favorites, .trash]

    func title(in shelves: [CollectionShelf]) -> String {
        switch self {
        case .all: "All Prints"
        case .favorites: "Favourites"
        case .trash: "Recently Deleted"
        case let .collection(slug):
            shelves.first { $0.slug == slug }?.name ?? "Collection"
        }
    }

    var symbol: String {
        switch self {
        case .all: "photo.on.rectangle.angled"
        case .favorites: "star"
        case .collection: "rectangle.stack"
        case .trash: "trash"
        }
    }

    /// Trash is a different LISTING on the host, not a filter over the live
    /// one -- which is why it cannot be expressed as a search token.
    var isTrash: Bool { self == .trash }

    var collectionSlug: String? {
        if case let .collection(slug) = self { return slug }
        return nil
    }

    /// The narrowing this shelf adds on top of whatever was searched for.
    func token(in shelves: [CollectionShelf]) -> LibraryToken? {
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

    var emptyMessage: String {
        switch self {
        case .all: "Prints from every machine appear here."
        case .favorites: "Stars you add show up here."
        case .collection: "Drag prints onto this collection to file them here."
        case .trash: "Deleted prints wait here until their machine purges them."
        }
    }
}
