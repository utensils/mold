import Foundation

/// What kind of thing a print is, as a person would say it.
public enum PrintKind: String, Hashable, Sendable, CaseIterable {
    case picture, clip, mesh
}

/// One chip in the search field.
///
/// A token is a *fact* about a print, and the field holds as many as you like.
/// How several of them combine is the interesting part -- see `LibraryQuery`.
public enum LibraryToken: Hashable, Sendable, Identifiable {
    case tag(String)
    case machine(id: MoldHost.ID, name: String)
    case kind(PrintKind)
    case favorite
    /// A shelf. It carries the collection's id on EVERY machine that has it,
    /// because a print's `collections` are its own host's ids -- filtering
    /// with another machine's id matches nothing and reads as an empty shelf.
    case collection(slug: String, name: String, ids: [MoldHost.ID: String])

    public var id: String {
        switch self {
        case let .tag(name): "tag:\(name)"
        case let .machine(id, _): "machine:\(id)"
        case let .kind(kind): "kind:\(kind.rawValue)"
        case .favorite: "favorite"
        case let .collection(slug, _, _): "collection:\(slug)"
        }
    }

    /// What the chip reads as.
    public var label: String {
        switch self {
        case let .tag(name): name
        case let .machine(_, name): name
        case let .kind(kind): kind.rawValue.capitalized
        case .favorite: "Favourite"
        case let .collection(_, name, _): name
        }
    }
}

public enum LibrarySort: String, Hashable, Sendable, CaseIterable {
    case newest, oldest, largest, name

    public var title: String {
        switch self {
        case .newest: "Newest First"
        case .oldest: "Oldest First"
        case .largest: "Largest First"
        case .name: "Name"
        }
    }
}
