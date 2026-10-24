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
    /// A model family, by canonical slug (`GalleryPrint.familySlug`).
    case family(slug: String, name: String)
    /// In any collection (true) or in none (false), on any copy.
    case filed(Bool)
    /// Made from existing media (`OutputMetadata.usedSourceMedia`).
    case fromSource
    /// Carries a tag (true) or none (false).
    case tagged(Bool)
    case orientation(PrintOrientation)

    public var id: String {
        switch self {
        case let .tag(name): "tag:\(name)"
        case let .machine(id, _): "machine:\(id)"
        case let .kind(kind): "kind:\(kind.rawValue)"
        case .favorite: "favorite"
        case let .collection(slug, _, _): "collection:\(slug)"
        case let .family(slug, _): "family:\(slug)"
        case let .filed(filed): filed ? "filed" : "unfiled"
        case .fromSource: "from-source"
        case let .tagged(tagged): tagged ? "tagged" : "untagged"
        case let .orientation(orientation): "orientation:\(orientation.rawValue)"
        }
    }

    /// The chip's icon. A chip is small and a person scanning a row of them
    /// reads the shapes before the words.
    public var symbol: String {
        switch self {
        case .tag: "tag"
        case .machine: "server.rack"
        case .favorite: "star"
        case let .kind(kind):
            switch kind {
            case .picture: "photo"
            case .clip: "film"
            case .mesh: "cube"
            }
        case .collection: "rectangle.stack"
        case .family: "circle.fill"
        case let .filed(filed): filed ? "rectangle.stack" : "rectangle.stack.badge.minus"
        case .fromSource: "photo.badge.arrow.down"
        case let .tagged(tagged): tagged ? "tag" : "tag.slash"
        case let .orientation(orientation): LibraryFilter.symbol(for: orientation)
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
        case let .family(_, name): name
        case let .filed(filed): filed ? "In a Collection" : "Not in a Collection"
        case .fromSource: "Made from a Source"
        case let .tagged(tagged): tagged ? "Has Tags" : "Untagged"
        case let .orientation(orientation): orientation.rawValue.capitalized
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
