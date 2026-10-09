import MoldClient
import SwiftUI

/// The title menu: which shelf the Library shows (DESIGN.md §5.2).
struct ShelfMenu: View {
    @Environment(LibraryStore.self) private var library
    let scope: LibraryScope
    var machineIDs: Set<MoldHost.ID>? = nil
    let choose: (LibraryScope) -> Void

    private func shelfLabel(_ shelf: CollectionShelf) -> String {
        let presence = library.shelfPresence(shelf, on: machineIDs ?? library.machineIDs)
        if presence == .unavailable { return "\(shelf.name) · Unavailable" }
        if presence == .absent { return "\(shelf.name) · Not on machine" }
        return "\(shelf.name) · \(shelf.count(in: library.scopedPool(on: machineIDs ?? library.machineIDs)))"
    }

    var body: some View {
        Picker("Shelf", selection: Binding(get: { scope }, set: choose)) {
            ForEach([LibraryScope.all, .favorites], id: \.self) { shelf in
                Label(shelf.title(in: library.shelves), systemImage: shelf.symbol).tag(shelf)
            }
            if !library.shelves.isEmpty {
                Section("Collections") {
                    ForEach(library.shelves) { shelf in
                        Label(shelfLabel(shelf), systemImage: shelf.hidden ? "rectangle.stack.badge.minus" : "rectangle.stack")
                            .tag(LibraryScope.collection(slug: shelf.slug))
                    }
                }
            }
            Label(LibraryScope.trash.title(in: library.shelves), systemImage: LibraryScope.trash.symbol)
                .tag(LibraryScope.trash)
        }
    }
}

/// Tokens offered while typing: videos, 3-D, a tag, a machine, favourites.
struct SearchSuggestions: View {
    @Binding var query: LibraryQuery
    let machines: [(id: MoldHost.ID, name: String)]
    let tags: [String]

    var body: some View {
        let offered = LibrarySearchSyntax.suggestions(
            for: query.text, machines: machines, tags: tags, applied: Set(query.tokens.map(\.id)))
        ForEach(offered) { token in
            Button {
                query.tokens.append(token)
                query.text = ""
            } label: {
                Label(token.label, systemImage: token.symbol)
            }
        }
    }
}

/// What a selection can do, along the bottom (DESIGN.md §5.2). Trash swaps in Put Back and Delete Immediately.
struct SelectionBar: View {
    @Environment(LibraryStore.self) private var library
    @Environment(PrintActions.self) private var actions
    let scope: LibraryScope
    let selected: [LibraryEntry]
    let cleared: () -> Void
    @State private var confirmDelete = false

    var body: some View {
        HStack {
            if scope.isTrash {
                Button("Put Back") { Task { await library.putBack(selected); cleared() } }
                Spacer()
                Button("Delete Immediately", role: .destructive) { confirmDelete = true }
            } else {
                Button { actions.share(selected) } label: {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                Spacer()
                Button { library.apply(.favorite(!allFavourite), to: selected) } label: {
                    Label(allFavourite ? "Unfavourite" : "Favourite", systemImage: allFavourite ? "star.slash" : "star")
                }
                Spacer()
                PrintMenu(entries: selected, trashed: false, compact: true)
                Spacer()
                Button(role: .destructive) { Task { await library.trash(selected); cleared() } } label: {
                    Label("Delete", systemImage: "trash")
                }
            }
        }
        .labelStyle(.iconOnly)
        .disabled(selected.isEmpty)
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
        .background(.bar)
        .confirmationDialog(String(localized: "Delete \(selected.count) prints immediately?"),
                            isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete Immediately", role: .destructive) {
                Task { await library.deleteImmediately(selected); cleared() }
            }
        } message: {
            Text("They are removed from \(LibraryEntry.soleMachineName(of: selected) ?? "all machines holding these copies") for good. This can't be undone.")
        }
    }

    private var allFavourite: Bool { !selected.isEmpty && selected.allSatisfy(\.isFavorite) }
}

/// Empty Trash, after asking.
struct EmptyTrashButton: View {
    var machineIDs: Set<MoldHost.ID> = []
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @State private var confirm = false

    private var targetMachines: String {
        hosts.hosts.filter { machineIDs.isEmpty || machineIDs.contains($0.id) }.map(\.name).joined(separator: ", ")
    }

    var body: some View {
        Button("Empty Trash…", role: .destructive) { confirm = true }
            .confirmationDialog("Empty Trash?", isPresented: $confirm, titleVisibility: .visible) {
                Button("Empty", role: .destructive) { Task { await library.emptyTrash(on: machineIDs) } }
            } message: {
                Text("Every print in Trash on \(targetMachines) is removed for good. Other machines keep their copies.")
            }
    }
}

struct LibraryMachinePicker: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts

    var body: some View {
        @Bindable var library = library
        Picker("Machine", selection: $library.machineID) {
            Text("All Machines").tag(nil as MoldHost.ID?)
            ForEach(hosts.hosts) { host in
                Text(host.name).tag(Optional(host.id))
            }
        }
    }
}
