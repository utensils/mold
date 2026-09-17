import MoldClient
import SwiftUI

// Which machine the pane is about, which of its models match, and how those
// rows collect into the table's sections. Split from the view for size --
// `private` does not cross a file boundary, so what the pane reads is
// `internal` here.
extension ModelsPane {
    var host: MoldHost? { hosts.machine(selected: selectedMachine) }

    /// Discover only where the CURRENT machine says it browses a catalog --
    /// switching machines can change which scopes this picker offers
    /// (design S6).
    var availableScopes: [ModelScope] {
        ModelScope.available(capabilities: host.flatMap { hosts.capabilities[$0.id] })
    }

    /// The machine's own answer to `/api/status`, read for its `modelsDisk`
    /// figure -- `nil` off a host that hasn't answered, which the footer
    /// treats the same as one that predates the field.
    var status: ServerStatus? {
        guard let host, case let .up(status) = hosts.reachability(of: host) else { return nil }
        return status
    }

    /// Every installed row on this machine, of every family -- a management
    /// listing, not the picker's `ready(on:)` (design fact 5, M5): a
    /// half-installed model or a non-generator belongs here, because this is
    /// the one place somebody would go to fix or manage it.
    var candidates: [Model] {
        guard let host else { return [] }
        let all = models.installed(on: host.id)
        guard !query.isEmpty else { return all }
        let needle = query.lowercased()
        return all.filter {
            $0.description.lowercased().contains(needle) || $0.name.lowercased().contains(needle)
        }
    }

    /// Grouped by the server's OWN family string (decision 9, M5) and
    /// ordered within each group the way the table's current sort says.
    var sections: [(family: String, rows: [Model])] {
        ModelSort.grouped(candidates, by: sort.wrappedValue)
    }
}
