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
        AlsoRunning.rows(reported: activity.rows, queuedIDs: queuedIDs,
                         upscales: upscales.live, stills: upscales.liveStills)
    }

    /// One machine's, for the section under its name. Filtered from the
    /// fleet list rather than computed per machine, because the suppression
    /// rules read across the whole answer.
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
                    AlsoRunningRowView(row: row, actions: AlsoRunningActions(row, mutationAllowed: { action in
                        if case let .reported(reported) = row.work { return activity.canAct(action, on: reported) }
                        if case let .upscale(filename, job) = row.work {
                            let key = UpscaleStore.Key(host: row.host, filename: filename)
                            guard upscales.jobs[key]?.id == job.id else { return false }
                            switch action {
                            case .pause: return upscales.canTransition(key, to: .pause)
                            case .resume: return upscales.canTransition(key, to: .resume)
                            case .cancel: return upscales.canTransition(key, to: .cancel)
                            case .forget: return true
                            }
                        }
                        return hosts.isUp(host)
                    })) { act($0, on: row) }
                }
            }
        }
    }

    /// Reported chains recheck the machine's authority before dispatch;
    /// upscales preserve the exact rendered job identity and state.
    func act(_ action: AlsoRunningActions.Kind, on row: AlsoRunningRow) {
        if case let .reported(reported) = row.work {
            Task { await activity.act(action, on: reported) }
            return
        }
        if case let .still(filename, _) = row.work, action == .forget {
            upscales.forgetStill(UpscaleStore.Key(host: row.host, filename: filename))
            return
        }
        guard case let .upscale(filename, job) = row.work else { return }
        let key = UpscaleStore.Key(host: row.host, filename: filename)
        switch action {
        case .pause: Task { await upscales.transition(key, to: .pause, expectedJob: job) }
        case .resume: Task { await upscales.transition(key, to: .resume, expectedJob: job) }
        case .cancel: Task { await upscales.transition(key, to: .cancel, expectedJob: job) }
        case .forget: upscales.forget(key)
        }
    }
}
