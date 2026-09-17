import Foundation
import MoldClient

// A whole batch's Pause/Resume/Cancel. Split from `QueueStore+Batches.swift`
// past the file-size advisory; the hydration it used to sit beside is a READ,
// this is the one thing in that area that WRITES.
@MainActor
extension QueueStore {

    /// One action reaching every child of a group that offers it, serialized,
    /// then a single re-read -- the same "N calls, one re-read" shape
    /// reorder's batch move takes (design decision 4), rather than one re-read
    /// per child. A settled row (complete, failed, cancelled) is left alone
    /// even when it rides along in `group.rows`.
    func act(_ action: QueueRow.Action, onLiveChildrenOf group: QueueGroup, host: MoldHost.ID) async {
        guard !refuseIfFixture(host, doing: groupVerb(action)) else { return }
        guard let client = hosts.backend(for: host) else { return }
        // Each child asked again, through the same authority its own row
        // drew from: a batch with one waiting and one running child pauses
        // the waiting one and leaves the other alone, rather than sending a
        // call the machine refuses by name (`routes.rs:7706-7710`).
        let capabilities = hosts.capabilities[host]
        for entry in group.rows
        where QueueRowActions.resolve(entry, on: capabilities).offers(action) {
            do {
                switch action {
                case .cancel: try await client.cancelJob(id: entry.id)
                case .pause: try await client.pauseJob(id: entry.id)
                case .resume: try await client.resumeJob(id: entry.id)
                // Retry needs a `QueueAuthority` per row, not a bare id, and
                // belongs to `QueueHoldRow` -- not a group-wide action.
                case .retry: continue
                }
                hosts.succeeded(on: host)
            } catch {
                hosts.report(error, on: host, doing: groupVerb(action))
            }
        }
        await poll(host)
    }

    private func groupVerb(_ action: QueueRow.Action) -> String {
        switch action {
        case .cancel: "cancel that job"
        case .pause: "pause that job"
        case .resume: "resume that job"
        case .retry: "retry that job"
        }
    }
}
