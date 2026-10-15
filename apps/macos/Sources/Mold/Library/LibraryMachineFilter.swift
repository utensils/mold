import MoldClient
import SwiftUI

/// A native single-choice menu over the Library's machine search tokens.
/// A multi-machine query keeps its summary and has no single checked host.
struct LibraryMachineFilter: View {
    let query: LibraryQuery
    let hosts: [MoldHost]
    let choose: (MoldHost?) -> Void

    var body: some View {
        let title = Self.title(for: query, hosts: hosts)
        Menu {
            Picker("Machine", selection: Binding(get: { query.machineIDs }, set: { ids in
                choose(hosts.first { ids.contains($0.id) })
            })) {
                Text("All Machines").tag(Set<MoldHost.ID>())
                ForEach(hosts) { host in
                    Text(host.name).tag(Set([host.id]))
                }
            }
            .pickerStyle(.inline)
        } label: {
            LibraryFilterLabel(title: title, systemImage: "server.rack")
        }
        .accessibilityLabel("Machine: \(title)")
        .help("Show prints from one machine (\(title))")
    }

    static func title(for query: LibraryQuery, hosts: [MoldHost]) -> String {
        let ids = query.machineIDs
        guard ids.count == 1, let id = ids.first else {
            return ids.isEmpty ? "All Machines" : "\(ids.count) Machines"
        }
        if let host = hosts.first(where: { $0.id == id }) { return host.name }
        // Search tokens retain the name even if a machine has been removed.
        return query.tokens.compactMap {
            if case let .machine(tokenID, name) = $0, tokenID == id { return name }
            return nil
        }.first ?? "One Machine"
    }
}
