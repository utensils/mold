import CoreTransferable
import MoldClient
import SwiftUI
import UniformTypeIdentifiers

// The iPad sidebar's two sections (DESIGN.md §4), under the five
// destinations: the Library's other shelves -- Favourites, each collection,
// Recently Deleted -- and one row per machine, saying when it is not
// answering. Sidebar only (never in the floating tab bar); on iPhone the
// Library's title menu and the Machines list do this.

extension RootView {
    @TabContentBuilder<TabSelection>
    func librarySection(_ library: LibraryStore) -> some TabContent<TabSelection> {
        TabSection {
            shelfTab(.favorites, library)
            ForEach(library.shelves, id: \.slug) { collection in
                shelfTab(.collection(slug: collection.slug), library)
                    // Drop prints on a collection to file them there.
                    .dropDestination(for: PrintDrop.self) { drops in
                        let ids = Set(drops.map(\.id))
                        let entries = library.pool.filter { entry in entry.everyCopy.contains { ids.contains($0.id) } }
                        library.apply(.collection(name: collection.name, slug: collection.slug, filing: true), to: entries)
                    }
            }
            shelfTab(.trash, library)
        } header: {
            Text("Shelves")
        }
        .defaultVisibility(.hidden, for: .tabBar)
    }

    /// Sidebar only: in the floating tab bar every shelf and machine became
    /// an item, and UIKit paged and re-laid that bar out for so long on a
    /// text-size change that the app stopped answering (the audit's
    /// Dynamic Type pass caught it). The bar keeps All Prints and Machines.
    private func shelfTab(_ scope: LibraryScope, _ library: LibraryStore) -> some TabContent<TabSelection> {
        Tab(value: TabSelection.shelf(scope)) {
            NavigationStack { LibraryView(fixedScope: scope) }
        } label: {
            Label { Text(scope.title(in: library.shelves)) } icon: { Image(systemName: scope.symbol) }
        }
        .defaultVisibility(.hidden, for: .tabBar)
    }

    @TabContentBuilder<TabSelection>
    func machinesSection(_ hosts: HostStore) -> some TabContent<TabSelection> {
        TabSection {
            ForEach(hosts.hosts) { host in
                Tab(value: TabSelection.machine(host.id)) {
                    NavigationStack { MachineDetailView(id: host.id).machineRoutes() }
                } label: {
                    Label(host.name, systemImage: "desktopcomputer")
                }
                // Words beside the name, never colour alone.
                .badge(Self.badge(hosts.reachability(of: host)))
                .defaultVisibility(.hidden, for: .tabBar)
            }
        } header: {
            Text("Your Machines")
        }
        .defaultVisibility(.hidden, for: .tabBar)
    }

    static func badge(_ state: HostStore.Reachability) -> Text? {
        switch state {
        case .down: Text("Offline")
        case .needsKey: Text("Key")
        case .unknown, .checking, .up: nil
        }
    }
}

/// A print's identity, dragged inside the app (onto a sidebar collection).
/// Other apps get the file itself from `DraggedPrint`.
nonisolated struct PrintDrop: Codable, Transferable {
    let host: UUID
    let filename: String
    var id: PrintID { PrintID(host: host, filename: filename) }

    static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .moldPrint)
    }
}

extension UTType {
    nonisolated static let moldPrint = UTType(exportedAs: "io.utensils.mold.print-id")
}
