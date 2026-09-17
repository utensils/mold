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

    @State private var renaming: CollectionShelf?
    @State private var isCreating = false
    @State private var pendingDestruction: LibraryActions.Destruction?

    private func confirmDestruction(_ destruction: LibraryActions.Destruction) {
        pendingDestruction = destruction
    }

    var body: some View {
        List(selection: selection) {
            Section {
                ForEach(Destination.allCases) { item in
                    Label(item.title, systemImage: item.symbol).tag(Row.destination(item))
                }
            }

            Section("Library") {
                shelfRow(.all)
                shelfRow(.favorites)
                ForEach(library.shelves) { shelf in
                    CollectionRow(shelf: shelf, renaming: $renaming)
                        .tag(Row.shelf(.collection(slug: shelf.slug)))
                }
                shelfRow(.trash)
                    .contextMenu {
                        Button("Empty Trash…", role: .destructive) {
                            LibraryActions(hosts: hosts, library: library,
                                           confirmDestruction: confirmDestruction).emptyTrash()
                        }
                        .disabled(library.trashed.isEmpty)
                    }
                Button("New Collection…", systemImage: "plus") { isCreating = true }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
            }

            Section("Machines") {
                ForEach(hosts.hosts) { host in
                    HostRow(host: host, reachability: hosts.reachability(of: host))
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
        .tag(Row.shelf(scope))
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

    /// One selection over two kinds of row. Picking a shelf also moves to the
    /// Library, because choosing what to look at and choosing to look are the
    /// same act -- making them two clicks would be a bug people report.
    private var selection: Binding<Row?> {
        Binding(
            get: { destination == .library ? .shelf(navigation.scope) : .destination(destination) },
            set: { row in
                switch row {
                case let .destination(item): destination = item
                case let .shelf(scope):
                    navigation.scope = scope
                    destination = .library
                case nil: break
                }
            }
        )
    }

    private enum Row: Hashable {
        case destination(Destination)
        case shelf(LibraryScope)
    }
}

/// One machine in the sidebar. Not selectable: a machine is something the
/// library is filtered BY, not a place to go, and the filter is a search chip.
private struct HostRow: View {
    let host: MoldHost
    let reachability: HostStore.Reachability

    var body: some View {
        HStack(spacing: 8) {
            HostStatusDot(reachability: reachability)
            VStack(alignment: .leading, spacing: 1) {
                Text(host.name)
                if let detail = reachability.summary {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
        .help(HostAddress.displayString(for: host.baseURL))
    }
}
