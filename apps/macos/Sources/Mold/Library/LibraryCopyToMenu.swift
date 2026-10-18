import MoldClient
import SwiftUI

/// Native menu builders and Commands do not inherit the window environment.
/// Capture the offered destinations and owning action before entering either.
struct LibraryCopyToMenu: View {
    let destinations: [MoldHost]
    let copy: (MoldHost) -> Void
    var body: some View {
        if !destinations.isEmpty {
            Menu("Copy to") {
                ForEach(destinations) { host in
                    Button(host.name) { copy(host) }
                }
            }
        }
    }
}

extension LibraryActions {
    func copyDestinations(for entries: [LibraryEntry], scope: LibraryScope) -> [MoldHost] {
        guard !scope.isTrash else { return [] }
        return hosts.hosts.filter { host in
            hosts.isUp(host) && entries.contains { $0.hostID != host.id }
        }
    }
    func copy(_ entries: [LibraryEntry], to host: MoldHost) {
        library.syncSession.copy(entries, to: host, in: library)
    }
}
