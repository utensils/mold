import MoldClient
import SwiftUI

/// The queue (DESIGN.md §5.3): one section per machine; a batch is a parent
/// row with its children; a held row asks in words. Actions are the ones the
/// machine advertises -- absent, never present and inert.
struct QueueView: View {
    @Environment(HostStore.self) private var hosts
    @Environment(QueueStore.self) private var queue
    @Environment(AppRouter.self) private var router
    @State private var emptying: [MoldHost.ID]?

    var body: some View {
        Group {
            if hosts.hosts.isEmpty {
                EmptyState(title: String(localized: "Nothing waiting"), symbol: Destination.queue.symbol,
                           message: String(localized: "Renders you start appear here. Add a machine to begin.")) {
                    Button("Add a Machine…") { router.addMachine() }.prominentAction()
                }
            } else if queue.isEmpty, !queue.unavailableMachines.isEmpty {
                EmptyState(title: String(localized: "Queue unavailable"), symbol: Destination.queue.symbol,
                           message: String(localized: "Some machines could not provide their queues. Check Machines to reconnect, then pull to refresh.")) {
                    Button("Check Machines") { router.selection = .go(.machines) }.prominentAction()
                }
            } else if queue.isEmpty {
                EmptyState(title: String(localized: "Nothing waiting"), symbol: Destination.queue.symbol,
                           message: String(localized: "Renders you start appear here, on every machine."))
            } else {
                list
            }
        }
        .toolbar { QueueToolbar(emptying: $emptying) }
        .confirmationDialog(emptyTitle, isPresented: Binding(get: { emptying != nil }, set: { if !$0 { emptying = nil } }),
                            titleVisibility: .visible) {
            Button("Empty Queue", role: .destructive) {
                let ids = emptying ?? []
                Task { await queue.empty(ids) }
            }
        } message: {
            Text("Anything already rendering keeps going. Waiting and held jobs are cancelled.")
        }
        .refreshable { await queue.reload() }
        // A job just held asks for a decision: a warning, as on Generate.
        .sensoryFeedback(.warning, trigger: queue.heldCount) { old, new in new > old }
        .task(id: hosts.upHosts.map(\.id)) { await queue.reload() }
        .task { await queue.followRunning() }
    }

    private var emptyTitle: String {
        guard let ids = emptying else { return "" }
        if ids.count == 1, let host = hosts.host(ids[0]) { return String(localized: "Empty the queue on \(host.name)?") }
        return String(localized: "Empty the queue on every machine?")
    }

    private var list: some View {
        List {
            FailureBanner()
                .listRowInsets(EdgeInsets())
                .listRowBackground(Color.clear)
            if !queue.unavailableMachines.isEmpty {
                Text("Some machine queues are unavailable. Check Machines to reconnect.")
                    .foregroundStyle(.secondaryText)
            }
            if let summary = queue.summary {
                Text(summary)
                    .foregroundStyle(.secondaryText)
                    .onTapGesture { queue.summary = nil }
                    .task { try? await Task.sleep(for: .seconds(6)); queue.summary = nil }
            }
            ForEach(hosts.hosts) { host in
                let groups = queue.groups(for: host.id)
                if !groups.isEmpty {
                    Section {
                        ForEach(groups) { group in
                            QueueGroupRows(group: group, host: host)
                                .moveDisabled(!queue.canReorder(on: host.id) || !group.rows.allSatisfy(\.state.isReorderable))
                        }
                        .onMove { from, to in move(groups, from: from, to: to, on: host.id) }
                    } header: {
                        QueueSectionHeader(host: host)
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
        .accessibilityIdentifier("queue-list")
        .frame(maxWidth: QueueLayout.readableWidth)
        .frame(maxWidth: .infinity)
        .background(Color(uiColor: .systemGroupedBackground))
    }

    /// A drag in Edit mode, told to the machine as "after this row" -- the
    /// one index space its reorder route reads (`QueueOrder`).
    private func move(_ groups: [QueueGroup], from: IndexSet, to: Int, on id: MoldHost.ID) {
        guard let index = from.first else { return }
        var after = groups
        let moving = after.remove(at: index)
        let landing = to > index ? to - 1 : to
        let neighbour = QueueStore.neighbour(above: landing, in: after)
        Task { await queue.moveGroup(moving.rows.map(\.id), after: neighbour, on: id) }
    }
}

/// A machine's name over its rows, and whether it is dispatching.
private struct QueueSectionHeader: View {
    @Environment(QueueStore.self) private var queue
    let host: MoldHost

    var body: some View {
        HStack(spacing: 6) {
            Text(host.name)
            if queue.isQueuePaused(host.id) {
                Text("· Paused").accessibilityLabel("queue paused")
            }
        }
        .foregroundStyle(.secondaryText)
        .accessibilityAddTraits(.isHeader)
    }
}

/// Edit (reorder) and the whole-queue verbs: Pause / Resume, per machine and
/// for all of them, and Empty Queue….
private struct QueueToolbar: ToolbarContent {
    @Environment(QueueStore.self) private var queue
    @Environment(HostStore.self) private var hosts
    @Binding var emptying: [MoldHost.ID]?

    var body: some ToolbarContent {
        if hosts.upHosts.contains(where: { queue.canReorder(on: $0.id) }), !queue.isEmpty {
            ToolbarItem(placement: .topBarLeading) { EditButton() }
        }
        ToolbarItem(placement: .topBarTrailing) {
            Menu {
                gateItems
                let listed = hosts.upHosts.filter { !(queue.listings[$0.id] ?? []).isEmpty }
                if !listed.isEmpty {
                    Divider()
                    if listed.count == 1 {
                        Button("Empty Queue…", role: .destructive) { emptying = [listed[0].id] }
                    } else {
                        Button("Empty Queue on All Machines…", role: .destructive) { emptying = listed.map(\.id) }
                        ForEach(listed) { host in
                            Button("Empty Queue on \(host.name)…", role: .destructive) { emptying = [host.id] }
                        }
                    }
                }
            } label: {
                Label("Queue Actions", systemImage: "ellipsis")
            }
        }
    }

    @ViewBuilder private var gateItems: some View {
        let machines = queue.gateMachines
        if machines.count == 1 {
            let host = machines[0]
            let paused = queue.isQueuePaused(host.id)
            Button(paused ? "Resume Queue" : "Pause Queue",
                   systemImage: paused ? "play" : "pause") {
                Task { await queue.setQueuePaused(!paused, on: [host.id]) }
            }
        } else if machines.count > 1 {
            if machines.contains(where: { !queue.isQueuePaused($0.id) }) {
                Button("Pause All Machines", systemImage: "pause") {
                    Task { await queue.setQueuePaused(true, on: machines.map(\.id)) }
                }
            }
            if machines.contains(where: { queue.isQueuePaused($0.id) }) {
                Button("Resume All Machines", systemImage: "play") {
                    Task { await queue.setQueuePaused(false, on: machines.map(\.id)) }
                }
            }
            ForEach(machines) { host in
                let paused = queue.isQueuePaused(host.id)
                Button(paused ? "Resume Queue on \(host.name)" : "Pause Queue on \(host.name)") {
                    Task { await queue.setQueuePaused(!paused, on: [host.id]) }
                }
            }
        }
    }
}
