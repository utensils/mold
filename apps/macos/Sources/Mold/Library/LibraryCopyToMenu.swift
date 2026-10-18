import MoldClient
import SwiftUI

struct LibraryCopyToMenu: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    let entries: [LibraryEntry]
    var body: some View {
        let destinations = hosts.hosts.filter { host in
            hosts.isUp(host) && entries.contains { $0.hostID != host.id }
        }
        if !entries.isEmpty, !destinations.isEmpty {
            Menu("Copy to") {
                ForEach(destinations) { host in
                    Button(host.name) { library.syncSession.copy(entries, to: host, in: library) }
                }
            }
        }
    }
}

struct LibrarySelectionCopyToMenu: View {
    @Environment(LibraryStore.self) private var library
    let selection: LibrarySelection
    var body: some View {
        if !selection.scope.isTrash {
            LibraryCopyToMenu(entries: selection.targets.compactMap(library.entry))
        }
    }
}
