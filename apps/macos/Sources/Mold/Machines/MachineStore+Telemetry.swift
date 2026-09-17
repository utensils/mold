import Foundation
import MoldClient

// The 1 Hz resource stream. Split from `MachineStore` for size -- this is the
// half of "what a machine is" that moves every second, the rest of it barely
// moves at all.
@MainActor
extension MachineStore {
    /// Telemetry belongs to the pane that draws it, not the app -- an open
    /// connection per machine for the life of the process costs a frame per
    /// machine per second forever, whatever is on screen. Cancelling any
    /// other machine's stream first is what makes at most one exist in the
    /// process possible by construction, rather than by a reconcile loop.
    func watchResources(on host: MoldHost.ID) {
        if telemetry?.host != host { stopWatchingResources() }
        // Already watching this one -- leave the live stream alone.
        guard telemetry == nil, let client = hosts.backend(for: host) else { return }
        let task = Task { [weak self] in
            do {
                for try await snapshot in client.resourceStream() {
                    guard !Task.isCancelled else { return }
                    self?.resources[host] = snapshot
                }
            } catch {
                guard !Task.isCancelled else { return }
                self?.hosts.report(error, on: host, doing: "read its memory use")
            }
        }
        telemetry = (host, task)
    }

    func stopWatchingResources() {
        telemetry?.task.cancel()
        telemetry = nil
    }
}
