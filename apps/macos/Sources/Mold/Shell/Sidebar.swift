import MoldClient
import SwiftUI

/// The window's one list: where you are, what you are looking at, and what you
/// are looking at it on.
///
/// Collections are ROWS here rather than a shelf screen of their own, because
/// that is what a collection is on a Mac -- the same idiom Photos, Music and
/// Mail use for the same thing. Filing is a drag onto a row.
struct Sidebar: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @Environment(LibraryNavigation.self) private var navigation
    @Binding var destination: Destination
    /// The same key `MachinesPane` declares, over the same suite. Two views
    /// sharing one preference by name stay in sync with no plumbing -- the
    /// arrangement `destination` itself already uses.
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) private var selectedMachine = ""

    @State private var renaming: CollectionShelf?
    @State private var isCreating = false
    @State private var pendingDestruction: LibraryActions.Destruction?

    private func confirmDestruction(_ destruction: LibraryActions.Destruction) {
        pendingDestruction = destruction
    }

    var body: some View {
        List(selection: selection) {
            Section {
                ForEach(SidebarRows.destinations) { item in
                    Label(item.title, systemImage: item.symbol).tag(SidebarRow.destination(item))
                }
            }

            // The group label, not a destination: the rows under it ARE the
            // library, each one a shelf of it.
            Section("Library") {
                shelfRow(.all)
                shelfRow(.favorites)
                ForEach(library.shelves) { shelf in
                    CollectionRow(shelf: shelf, renaming: $renaming)
                        .tag(SidebarRow.shelf(.collection(slug: shelf.slug)))
                }
                shelfRow(.trash)
                    // Present and inert on an empty trash, not absent: the
                    // Library menu's own `Empty Trash…` says the same thing
                    // the same way.
                    .rowActionMenu([RowAction(kind: LibraryAction.emptyTrash,
                                              title: "Empty Trash…", isDestructive: true,
                                              isDisabled: library.trashed.isEmpty)]) { _ in
                        LibraryActions(hosts: hosts, library: library,
                                       confirmDestruction: confirmDestruction).emptyTrash()
                    }
                Button("New Collection…", systemImage: "plus") { isCreating = true }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
            }

            Section("Machines") {
                ForEach(hosts.hosts) { host in
                    MachineRow(host: host, reachability: hosts.reachability(of: host),
                               destination: $destination)
                        .tag(SidebarRow.machine(host.id))
                }
            }
        }
        .listStyle(.sidebar)
        .refreshable { await hosts.refreshAll() }
        .sheet(isPresented: $isCreating) { ShelfNameSheet(shelf: nil) }
        .sheet(item: $renaming) { ShelfNameSheet(shelf: $0) }
        .destructionDialog($pendingDestruction)
    }

    private func shelfRow(_ scope: LibraryScope) -> some View {
        Label {
            HStack {
                Text(scope.title(in: library.shelves)).lineLimit(1)
                if let count = count(of: scope) {
                    Spacer(minLength: 6)
                    Text(count.formatted())
                        .font(.caption)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
        } icon: {
            Image(systemName: scope.symbol)
        }
        .tag(SidebarRow.shelf(scope))
    }

    /// Every badge is a promise about what opening the row shows, so each one
    /// counts the rows that row would draw. All Prints is deliberately without
    /// one: the number is in the pane's subtitle, and repeating four figures in
    /// a sidebar row is noise.
    private func count(of scope: LibraryScope) -> Int? {
        switch scope {
        case .favorites: library.items.count { $0.print.isFavorite }
        case .trash: library.trashed.isEmpty ? nil : library.trashed.count
        case .all, .collection: nil
        }
    }

    /// One selection over three kinds of row, both ways, decided by
    /// `SidebarRows` -- so the highlighted row and what picking one does can
    /// never disagree about where the window is.
    private var selection: Binding<SidebarRow?> {
        Binding(
            get: {
                SidebarRows.selected(destination: destination, scope: navigation.scope,
                                     machine: hosts.machine(selected: selectedMachine)?.id)
            },
            set: { row in
                guard let pick = SidebarRows.pick(row) else { return }
                if let scope = pick.scope { navigation.scope = scope }
                if let machine = pick.machine {
                    selectedMachine = machine.uuidString
                } else if pick.destination == .machines {
                    // The top-level row is the FLEET; a machine still selected
                    // here would map the row straight back to that machine's page.
                    selectedMachine = ""
                }
                destination = pick.destination
            }
        )
    }
}
