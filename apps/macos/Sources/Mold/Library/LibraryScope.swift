import Foundation

/// Which shelf of the library is showing.
enum LibraryScope: Hashable, Identifiable, CaseIterable {
    case all
    case favorites
    case trash

    var id: Self { self }

    var title: String {
        switch self {
        case .all: "All Prints"
        case .favorites: "Favorites"
        case .trash: "Recently Deleted"
        }
    }

    var symbol: String {
        switch self {
        case .all: "photo.on.rectangle.angled"
        case .favorites: "star"
        case .trash: "trash"
        }
    }

    /// Trash is a different listing on the host, not a filter over the live one.
    var isTrash: Bool { self == .trash }
}
