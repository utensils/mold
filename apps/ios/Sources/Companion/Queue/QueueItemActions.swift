import MoldClient
import SwiftUI

/// Cards keep status and transfer controls; details keep explicit recovery actions.
struct QueueItemActions: View {
    @Environment(QueueStore.self) private var queue
    let entry: QueueEntry
    let host: MoldHost
    var detail = false

    var body: some View {
        if queue.isActing(entry, on: host.id) {
            ProgressView("Updating job…")
        } else {
            if let hold = queue.hold(for: entry, on: host.id) {
                QueueHeldActions(entry: entry, hold: hold, host: host, detail: detail)
            } else if detail, queue.canPause(entry, on: host.id) {
                Button(entry.state == .paused ? "Resume" : "Pause",
                       systemImage: entry.state == .paused ? "play" : "pause") {
                    Task { await queue.setPaused(entry.state != .paused, entry, on: host.id) }
                }
                .buttonStyle(.bordered)
            }
        }
    }
}
