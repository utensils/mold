import MoldClient
import SwiftUI

/// Everything a print (or a selection) can do, in the Mac tile menu's order,
/// destructive last behind a divider (DESIGN.md §5.2). Items appear only where
/// every machine involved can do them -- organizing needs the machine's
/// `gallery.organize` capability.
struct PrintMenu: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    @Environment(PrintActions.self) private var actions
    @Environment(AppRouter.self) private var router
    let entries: [LibraryEntry]
    let trashed: Bool
    var compact = false

    var body: some View {
        if compact {
            Menu { items } label: { Label("More", systemImage: "ellipsis.circle") }
        } else {
            items
        }
    }

    @ViewBuilder private var items: some View {
        if trashed {
            Button { Task { await library.putBack(entries) } } label: {
                Label("Put Back", systemImage: "arrow.uturn.backward")
            }
            Divider()
            Button(role: .destructive) { Task { await library.deleteImmediately(entries) } } label: {
                Label("Delete Immediately", systemImage: "trash.slash")
            }
        } else {
            if single, let entry = entries.first, entry.print.kind != .mesh {
                Button { router.reuse(entry) } label: {
                    Label("Use These Settings", systemImage: "arrow.uturn.left.circle")
                }
            }
            if canOrganize {
                Button { library.apply(.favorite(!allFavourite), to: entries) } label: {
                    Label(allFavourite ? "Unfavourite" : "Favourite", systemImage: allFavourite ? "star.slash" : "star")
                }
                collectionMenu
                Button { actions.sheet = .tags(entries) } label: { Label("Tags…", systemImage: "tag") }
                if single, let entry = entries.first {
                    Button { actions.sheet = .rename(entry) } label: { Label("Rename…", systemImage: "pencil") }
                }
            }
            Button { actions.share(entries) } label: { Label("Share…", systemImage: "square.and.arrow.up") }
            if entries.contains(where: { $0.print.kind != .mesh }) {
                Button { actions.saveToPhotos(entries) } label: {
                    Label("Save to Photos", systemImage: "square.and.arrow.down")
                }
            }
            if single, let entry = entries.first, entry.print.kind == .picture {
                Button { actions.copy(entry) } label: { Label("Copy", systemImage: "doc.on.doc") }
            }
            Divider()
            Button(role: .destructive) { Task { await library.trash(entries) } } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    @ViewBuilder private var collectionMenu: some View {
        Menu {
            ForEach(library.shelves.filter { !$0.hidden }) { shelf in
                let filed = entries.allSatisfy { shelf.count(in: [$0]) > 0 }
                Button {
                    library.apply(.collection(name: shelf.name, slug: shelf.slug, filing: !filed), to: entries)
                } label: {
                    Label(shelf.name, systemImage: filed ? "checkmark" : "rectangle.stack")
                }
            }
            Divider()
            Button { actions.sheet = .newCollection(entries) } label: {
                Label("New Collection…", systemImage: "plus")
            }
        } label: {
            Label("Add to Collection", systemImage: "rectangle.stack.badge.plus")
        }
    }

    private var single: Bool { entries.count == 1 }
    private var allFavourite: Bool { !entries.isEmpty && entries.allSatisfy(\.print.isFavorite) }
    private var canOrganize: Bool {
        entries.flatMap(\.everyCopy).allSatisfy { hosts.capabilities[$0.hostID]?.canOrganize == true }
    }
}

/// The sheets `PrintActions` asks for, presented once per window.
struct PrintSheets: ViewModifier {
    @Environment(PrintActions.self) private var actions

    func body(content: Content) -> some View {
        @Bindable var actions = actions
        content.sheet(item: $actions.sheet, onDismiss: { actions.shareFinished() }) { sheet in
            switch sheet {
            case let .share(urls): ShareSheet(items: urls).presentationDetents([.medium, .large])
            case let .tags(entries): TagsSheet(entries: entries)
            case let .newCollection(entries): NewCollectionSheet(entries: entries)
            case let .rename(entry): RenameSheet(entry: entry)
            }
        }
    }
}

extension View {
    func printSheets() -> some View { modifier(PrintSheets()) }
}
