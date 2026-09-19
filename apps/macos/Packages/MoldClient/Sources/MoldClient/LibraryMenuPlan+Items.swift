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
        var items: [Item] = []
        if canOpen { items.append(Item(kind: .open, title: "Open")) }
        items.append(Item(kind: .quickLook, title: quickLookTitle))
        if canReuse, count == 1 {
            items += [.separator, Item(kind: .reuse, title: Self.reuseTitle)]
        }
        if count == 1, canUseAsSource || canAddReference {
            items.append(.separator)
        }
        if canUseAsSource, count == 1 {
            items.append(Item(kind: .useAsSourceImage, title: "Use as Source Image"))
        }
        if canAddReference, count == 1 {
            items.append(Item(kind: .addAsReference, title: "Add as Reference"))
        }
        // Beside Use These Settings, because both make a NEW print out of
        // this one. One print at a time: the clip half is a durable job per
        // print, and starting several at once would queue a machine full of
        // work from one click.
        if canUpscale, count == 1 {
            items += [.separator, upscaleItem]
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
        if count == 1 { items.append(Item(title: "Export…", children: exportItems)) }
        return items + [.separator, Item(kind: .trash, title: "Move to Trash", isDestructive: true)]
    }

    /// What Export… holds. A clip's containers are one entry each; a mesh's
    /// come from its host's advertised list, with the animated ones collapsed
    /// into the ONE entry that opens the turntable sheet. An empty submenu is
    /// dropped by `rendered`, so a host advertising nothing leaves no row.
    private var exportItems: [Item] {
        guard let meshExports else {
            return exportFormats.map { Item(kind: .export(format: $0), title: $0.uppercased()) }
        }
        var items = meshExports.files.map {
            Item(kind: .export(format: $0), title: $0.uppercased())
        }
        if !meshExports.animations.isEmpty {
            items.append(Item(kind: .exportTurntable, title: "Turntable…"))
        }
        return items
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

    /// One item, or a submenu naming each installed upscaler.
    ///
    /// Desktop opens a dialog with a model picker; this app has no dialog, so
    /// the choice is where every other choice in this menu is. Fewer than two
    /// installed upscalers is one plain item that sends no model name at all,
    /// and the machine resolves its own default.
    private var upscaleItem: Item {
        guard upscalers.count > 1 else {
            return Item(kind: .upscale(model: nil), title: "Make Bigger…")
        }
        return Item(title: "Make Bigger", children: upscalers.map {
            Item(kind: .upscale(model: $0.name), title: $0.title)
        })
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
