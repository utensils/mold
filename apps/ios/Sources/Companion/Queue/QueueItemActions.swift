import MoldClient
import SwiftUI

/// The same state policy serves visible buttons, swipes and detail controls.
struct QueueItemActions: View {
    @Environment(QueueStore.self) private var queue
    @Environment(\.dynamicTypeSize) private var size
    let entry: QueueEntry
    let host: MoldHost
    var detail = false

    var body: some View {
        if queue.isActing(entry, on: host.id) {
            ProgressView("Updating job…")
        } else {
            if let hold = queue.hold(for: entry, on: host.id) {
                QueueHeldActions(entry: entry, hold: hold, host: host, detail: detail)
            } else if queue.canPause(entry, on: host.id) {
                Button(entry.state == .paused ? "Resume" : "Pause",
                       systemImage: entry.state == .paused ? "play" : "pause") {
                    Task { await queue.setPaused(entry.state != .paused, entry, on: host.id) }
                }
                .buttonStyle(.bordered)
            }
            if !detail, queue.canCancel(entry, on: host.id) {
                Button(role: .destructive) {
                    Task { await queue.cancel(entry, on: host.id) }
                } label: {
                    if size.isAccessibilitySize {
                        Text("Cancel Job")
                            .lineLimit(nil)
                            .multilineTextAlignment(.center)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity)
                            .foregroundStyle(.red)
                    } else {
                        Label("Cancel Job", systemImage: "xmark").foregroundStyle(.red)
                    }
                }
                .buttonStyle(.bordered)
                .tint(.red)
                .accessibilityIdentifier("queue-cancel-" + entry.id)
            }
        }
    }
}
