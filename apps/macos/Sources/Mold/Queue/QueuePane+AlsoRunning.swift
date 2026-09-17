import MoldClient
import SwiftUI

// **Also Running**: what a machine is doing that never becomes a queue row.
//
// `/api/queue` answers for generations and nothing else, so preparation, a
// prompt rewrite, a standalone upscale and a durable sequence left this pane
// looking idle while the GPU was flat out.
extension QueuePane {

    /// Every such row across the fleet -- what the empty state has to ask
    /// about before saying nothing is queued.
    var alsoRunning: [AlsoRunningRow] {
        AlsoRunning.rows(reported: activity.rows, queuedIDs: queuedIDs, upscales: upscales.live)
    }

    /// One machine's, for the section under its name.
    func alsoRunning(on host: MoldHost.ID) -> [AlsoRunningRow] {
        alsoRunning.filter { $0.host == host }
    }

    /// The ids the pane is ALREADY drawing, per machine. A row is excluded
    /// because it is on screen as a queue row, not because of what kind it
    /// is.
    private var queuedIDs: [MoldHost.ID: Set<String>] {
        var ids: [MoldHost.ID: Set<String>] = [:]
        for host in hosts.hosts {
            ids[host.id] = Set(queue.entries(on: host.id).map(\.id))
        }
        return ids
    }

    @ViewBuilder
    func alsoRunningSection(_ host: MoldHost) -> some View {
        let rows = alsoRunning(on: host.id)
        if !rows.isEmpty {
            Section("Also Running on \(host.name)") {
                ForEach(rows) { row in
                    AlsoRunningRowView(row: row) { act($0, on: row) }
                }
            }
        }
    }

    /// Cancel is offered only where the machine confirmed it for that exact
    /// item, and reaching a reported row's cancel is not this app's to invent
    /// -- the only work here it OWNS is the clip upscale it started.
    func act(_ action: AlsoRunningActions.Kind, on row: AlsoRunningRow) {
        guard case let .upscale(filename, _) = row.work else { return }
        let key = UpscaleStore.Key(host: row.host, filename: filename)
        switch action {
        case .pause: Task { await upscales.transition(key, to: .pause) }
        case .resume: Task { await upscales.transition(key, to: .resume) }
        case .cancel: Task { await upscales.transition(key, to: .cancel) }
        case .forget: upscales.forget(key)
        }
    }
}
