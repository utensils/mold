import Foundation

// The offer itself. Split from the types for size; this is the list both
// menus render, in this order, with these words.
public extension LibraryMenuPlan {

    /// The items, in the order they are shown. Empty when there is nothing
    /// applicable -- a row with no action gets NO menu, never an empty one.
    ///
    /// The Library groups its own list, so the separators below are where
    /// they are on purpose and `RowAction.rendered` leaves the order alone;
    /// it is still what drops an empty submenu and trims a stray divider.
    var items: [Item] {
        RowAction.rendered((scope == .trash ? trashItems : printItems) + collectionItems)
    }

    /// The Library's own row: a `RowAction` like every other menu's.
    typealias Item = RowAction<LibraryAction>

    private var printItems: [Item] {
        guard count > 0 else { return [] }
        var items: [Item] = [
            Item(kind: .open, title: "Open"),
            Item(kind: .quickLook, title: quickLookTitle),
        ]
        if canReuse, count == 1 {
            items += [.separator, Item(kind: .reuse, title: "Use These Settings")]
        }
        items += [
            .separator,
            Item(kind: .favorite(!allFavorite), title: favoriteTitle),
            Item(title: "Move to Collection", children: shelves.map {
                Item(kind: .file(slug: $0.slug), title: $0.name)
            }),
        ]
        if let shelf = enclosingShelf {
            items.append(Item(kind: .unfile(slug: shelf.slug), title: "Remove from \(shelf.name)"))
        }
        items += [
            .separator,
            Item(kind: .copy, title: "Copy"),
            Item(kind: .save, title: saveTitle),
        ]
        if count == 1 {
            items.append(Item(title: "Export…", children: exportFormats.map {
                Item(kind: .export(format: $0), title: $0.uppercased())
            }))
        }
        return items + [.separator, Item(kind: .trash, title: "Move to Trash", isDestructive: true)]
    }

    /// Recently Deleted. `Empty Trash…` belongs here too -- it was on the
    /// sidebar row and in the menu bar, and nowhere on a trashed tile.
    private var trashItems: [Item] {
        var items: [Item] = []
        if count > 0 {
            items += [
                Item(kind: .putBack, title: "Put Back"),
                .separator,
                Item(kind: .deleteForever, title: "Delete Immediately…", isDestructive: true),
            ]
        }
        if trashCount > 0 {
            items.append(Item(kind: .emptyTrash, title: "Empty Trash…", isDestructive: true))
        }
        return items
    }

    /// The shelf itself, when one is being shown. These three lived only in
    /// the sidebar's right-click menu: unreachable from the keyboard and
    /// invisible to Help ▸ Search.
    private var collectionItems: [Item] {
        guard let shelf = enclosingShelf, scope == .collection else { return [] }
        return [
            .separator,
            Item(kind: .renameCollection, title: "Rename “\(shelf.name)”…"),
            Item(kind: .setCollectionHidden(!shelf.hidden),
                 title: shelf.hidden ? "Show in All Prints" : "Hide from All Prints"),
            Item(kind: .deleteCollection, title: "Delete Collection…", isDestructive: true),
        ]
    }

    /// Quick Look names what it is about, the way the Finder does.
    private var quickLookTitle: String {
        guard count == 1 else { return "Quick Look \(count.formatted()) Prints" }
        return name.map { "Quick Look “\($0)”" } ?? "Quick Look"
    }

    /// One wording, and the sidebar's spelling of the shelf it files into.
    private var favoriteTitle: String {
        allFavorite ? "Remove from Favourites" : "Add to Favourites"
    }

    private var saveTitle: String { Self.saveTitle(count: count) }

    /// Also the File menu's, which said "Save a Copy…" over four selected
    /// prints and then wrote four files. Nothing selected keeps the singular:
    /// the item is disabled there, and "Save 0 Copies…" is not a sentence.
    static func saveTitle(count: Int) -> String {
        count <= 1 ? "Save a Copy…" : "Save \(count.formatted()) Copies…"
    }
}
