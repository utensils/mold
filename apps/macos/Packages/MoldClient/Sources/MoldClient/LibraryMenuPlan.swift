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
/// and both surfaces already render the same one -- it rides along as
/// `rowActionMenu`'s `extra`.
///
/// The offer is a `[RowAction<LibraryAction>]` like every other menu in the
/// app; this type is only what the list is resolved FROM.
public enum LibraryAction: Hashable, Sendable {
    case open
    case reuse
    case quickLook
    case favorite(Bool)
    case file(slug: String)
    case unfile(slug: String)
    /// Upscale this print on the machine that holds it. `nil` means the
    /// machine's own default -- what the plain item sends when this app has
    /// not read that machine's upscalers.
    case upscale(model: String?)
    case copy
    case save
    case export(format: String)
    /// A mesh's animated containers share ONE entry, which opens the sheet
    /// that carries their frames, rate and size -- a turntable is a RENDER of
    /// the mesh, not a transcode, and has options a transcode does not.
    case exportTurntable
    case trash
    case putBack
    case deleteForever
    case emptyTrash
    case renameCollection
    case setCollectionHidden(Bool)
    case deleteCollection
}

/// What to offer for a given selection.
public struct LibraryMenuPlan: Sendable {
    public let scope: LibraryScopeKind
    public let count: Int
    public let allFavorite: Bool
    public let name: String?
    public let shelves: [CollectionShelf]
    public let enclosingShelf: CollectionShelf?
    /// A CLIP's containers, one menu entry each.
    public let exportFormats: [String]
    /// A MESH's containers, from the holding host's own advertised list. The
    /// geometry files are one entry each and the animated ones collapse into
    /// a single Turntable… that opens the sheet.
    public let meshExports: MeshExport.Split?
    public let canReuse: Bool
    /// Whether the machine holding this print advertises upscaling it -- a
    /// clip needs `video_upscale`, a still `gallery_image` as well. Absence
    /// is a definitive no, and the item is then ABSENT rather than inert.
    public let canUpscale: Bool
    /// The installed upscalers to choose between, the default first. Fewer
    /// than two is one plain item -- a submenu with one row in it is a door
    /// onto a corridor.
    public let upscalers: [UpscalerOption]
    public let trashCount: Int

    public init(scope: LibraryScopeKind, count: Int, allFavorite: Bool = false,
                name: String? = nil, shelves: [CollectionShelf] = [],
                enclosingShelf: CollectionShelf? = nil, exportFormats: [String] = [],
                meshExports: MeshExport.Split? = nil,
                canReuse: Bool = false, canUpscale: Bool = false,
                upscalers: [UpscalerOption] = [], trashCount: Int = 0) {
        self.meshExports = meshExports
        self.scope = scope
        self.count = count
        self.allFavorite = allFavorite
        self.name = name
        self.shelves = shelves
        self.enclosingShelf = enclosingShelf
        self.exportFormats = exportFormats
        self.canReuse = canReuse
        self.canUpscale = canUpscale
        self.upscalers = upscalers
        self.trashCount = trashCount
    }
}

public extension LibraryMenuPlan {
    /// The ONE name for restoring a print's recipe, wherever it is read: the
    /// tile's menu, the menu bar, the inspector, and the mesh viewer's own
    /// controls. A second literal is a second name waiting to drift.
    static let reuseTitle = "Use These Settings"
}

/// Which shelf the Library is showing, as the plan needs to know it.
public enum LibraryScopeKind: Hashable, Sendable {
    case prints
    case collection
    case trash
}
