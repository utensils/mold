import Foundation

/// Everything you can do to what is selected in the Library, declared ONCE.
///
/// The menu bar and the tile's right-click menu were two hand-written lists,
/// and they disagreed: Move to Collection and Remove from ⟨shelf⟩ existed only
/// in the menu bar, Open, Use These Settings and Copy only on the tile, and
/// the same action was called "Favorite" in one and "Add to Favorites" in the
/// other. A person who learns a name in one place should find it in the other,
/// and a reviewer should be able to read the whole offer in one file.
///
/// Pure, so the order, the wording and the gating are a test rather than
/// something you check by right-clicking. `Share` is deliberately absent: it
/// is a `ShareLink`, a system control rather than an action this app performs,
/// and both surfaces already render the same one.
public enum LibraryAction: Hashable, Sendable {
    case open
    case reuse
    case quickLook
    case favorite(Bool)
    case file(slug: String)
    case unfile(slug: String)
    case copy
    case save
    case export(format: String)
    case trash
    case putBack
    case deleteForever
    case emptyTrash
    case renameCollection
    case setCollectionHidden(Bool)
    case deleteCollection
}

/// One row of the offer: a command, a submenu, or a divider.
public struct LibraryMenuItem: Hashable, Sendable, Identifiable {
    public let id: String
    public let title: String
    public let action: LibraryAction?
    public let children: [LibraryMenuItem]
    public let isDestructive: Bool

    public var isDivider: Bool { action == nil && children.isEmpty && title.isEmpty }
    public var isSubmenu: Bool { !children.isEmpty }

    public static let divider = LibraryMenuItem(id: "divider", title: "", action: nil,
                                                children: [], isDestructive: false)

    public init(id: String, title: String, action: LibraryAction? = nil,
                children: [LibraryMenuItem] = [], isDestructive: Bool = false) {
        self.id = id
        self.title = title
        self.action = action
        self.children = children
        self.isDestructive = isDestructive
    }
}

/// What to offer for a given selection.
public struct LibraryMenuPlan: Sendable {
    public let scope: LibraryScopeKind
    public let count: Int
    public let allFavorite: Bool
    public let name: String?
    public let shelves: [CollectionShelf]
    public let enclosingShelf: CollectionShelf?
    public let exportFormats: [String]
    public let canReuse: Bool
    public let trashCount: Int

    public init(scope: LibraryScopeKind, count: Int, allFavorite: Bool = false,
                name: String? = nil, shelves: [CollectionShelf] = [],
                enclosingShelf: CollectionShelf? = nil, exportFormats: [String] = [],
                canReuse: Bool = false, trashCount: Int = 0) {
        self.scope = scope
        self.count = count
        self.allFavorite = allFavorite
        self.name = name
        self.shelves = shelves
        self.enclosingShelf = enclosingShelf
        self.exportFormats = exportFormats
        self.canReuse = canReuse
        self.trashCount = trashCount
    }
}

/// Which shelf the Library is showing, as the plan needs to know it.
public enum LibraryScopeKind: Hashable, Sendable {
    case prints
    case collection
    case trash
}
