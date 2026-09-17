import Foundation

// The offer itself. Split from the types for size; this is the list both
// menus render, in this order, with these words.
public extension LibraryMenuPlan {

    /// The items, in the order they are shown. Empty when there is nothing
    /// applicable -- a row with no action gets NO menu, never an empty one.
    var items: [LibraryMenuItem] {
        let body = scope == .trash ? trashItems : printItems
        return collapsingDividers(body + collectionItems)
    }

    private var printItems: [LibraryMenuItem] {
        guard count > 0 else { return [] }
        var items: [LibraryMenuItem] = [
            .init(id: "open", title: "Open", action: .open),
            .init(id: "quickLook", title: quickLookTitle, action: .quickLook),
        ]
        if canReuse, count == 1 {
            items += [.divider, .init(id: "reuse", title: "Use These Settings", action: .reuse)]
        }
        items += [
            .divider,
            .init(id: "favorite", title: favoriteTitle, action: .favorite(!allFavorite)),
        ]
        if !shelves.isEmpty {
            items.append(.init(id: "file", title: "Move to Collection", children: shelves.map {
                .init(id: "file.\($0.slug)", title: $0.name, action: .file(slug: $0.slug))
            }))
        }
        if let shelf = enclosingShelf {
            items.append(.init(id: "unfile", title: "Remove from \(shelf.name)",
                               action: .unfile(slug: shelf.slug)))
        }
        items += [
            .divider,
            .init(id: "copy", title: "Copy", action: .copy),
            .init(id: "save", title: saveTitle, action: .save),
        ]
        if count == 1, !exportFormats.isEmpty {
            items.append(.init(id: "export", title: "Export…", children: exportFormats.map {
                .init(id: "export.\($0)", title: $0.uppercased(), action: .export(format: $0))
            }))
        }
        items += [
            .divider,
            .init(id: "trash", title: "Move to Trash", action: .trash, isDestructive: true),
        ]
        return items
    }

    /// Recently Deleted. `Empty Trash…` belongs here too -- it was on the
    /// sidebar row and in the menu bar, and nowhere on a trashed tile.
    private var trashItems: [LibraryMenuItem] {
        var items: [LibraryMenuItem] = []
        if count > 0 {
            items += [
                .init(id: "putBack", title: "Put Back", action: .putBack),
                .divider,
                .init(id: "deleteForever", title: "Delete Immediately…",
                      action: .deleteForever, isDestructive: true),
            ]
        }
        if trashCount > 0 {
            items.append(.init(id: "emptyTrash", title: "Empty Trash…",
                               action: .emptyTrash, isDestructive: true))
        }
        return items
    }

    /// The shelf itself, when one is being shown. These three lived only in
    /// the sidebar's right-click menu: unreachable from the keyboard and
    /// invisible to Help ▸ Search.
    private var collectionItems: [LibraryMenuItem] {
        guard let shelf = enclosingShelf, scope == .collection else { return [] }
        return [
            .divider,
            .init(id: "renameCollection", title: "Rename “\(shelf.name)”…",
                  action: .renameCollection),
            .init(id: "hideCollection",
                  title: shelf.hidden ? "Show in All Prints" : "Hide from All Prints",
                  action: .setCollectionHidden(!shelf.hidden)),
            .init(id: "deleteCollection", title: "Delete Collection…",
                  action: .deleteCollection, isDestructive: true),
        ]
    }

    /// No leading, trailing or doubled dividers, whatever the gating left out.
    private func collapsingDividers(_ items: [LibraryMenuItem]) -> [LibraryMenuItem] {
        var kept: [LibraryMenuItem] = []
        for item in items where !(item.isDivider && (kept.isEmpty || kept.last?.isDivider == true)) {
            kept.append(item)
        }
        while kept.last?.isDivider == true { kept.removeLast() }
        return kept
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

    private var saveTitle: String {
        count == 1 ? "Save a Copy…" : "Save \(count.formatted()) Copies…"
    }
}
