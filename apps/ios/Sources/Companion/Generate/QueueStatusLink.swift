import MoldClient
import SwiftUI

/// Foreground job counts, not the number of batches this window submitted.
struct QueueStatusLink: View {
    @Environment(QueueStore.self) private var queue
    @Environment(HostStore.self) private var hosts
    @Environment(AppRouter.self) private var router

    var body: some View {
        let entries = queue.listings.values.flatMap { $0 }
        let running = entries.filter { $0.state == .running }.count
        let waiting = entries.filter { $0.state == .queued || $0.state == .held || $0.state == .paused }.count
        Button { router.selection = .go(.queue) } label: {
            Label(!queue.unavailableMachines.isEmpty ? "View Queue · Some machines unavailable" : entries.isEmpty ? "View Queue" : "\(running) rendering · \(waiting) waiting",
                  systemImage: "list.bullet.indent")
                .font(.subheadline)
                .frame(minHeight: 44)
        }
        .accessibilityIdentifier("generate-queue-status")
        .task(id: hosts.upHosts.map(\.id)) { await queue.reload() }
    }
}
