import MoldClient
import SwiftUI

// The queue's toolbar: Refresh, and Empty Queue -- present only for a
// machine that actually advertises it (design M6 decision 5). A single
// machine offering it is a plain button; more than one is a menu naming
// each, the same "absent, not disabled" idiom transfer's own "Move to ▾"
// uses for its machine list.
extension QueuePane {
    @ToolbarContentBuilder var toolbar: some ToolbarContent {
        ToolbarItem {
            Button { Task { await load() } } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .disabled(queue.isLoading)
        }
        if !queueGate.machines.isEmpty {
            ToolbarItem { gateControl }
        }
        if !emptyQueueTargets.isEmpty {
            ToolbarItem { emptyQueueControl }
        }
    }

    /// The whole-queue gate, from the SAME offer the Queue menu draws. One
    /// machine is a plain button; more than one is a menu naming each --
    /// Empty Queue…'s own idiom, so a mixed fleet is never ambiguous.
    @ViewBuilder private var gateControl: some View {
        let gate = queueGate
        if gate.machines.count == 1, let machine = gate.machines.first {
            Button(machine.title) { gate.toggle(.machine(machine.id)) }
        } else {
            Menu("Queue") {
                RowActionMenu(actions: gate.items(), perform: gate.toggle)
            }
        }
    }

    /// The machines whose queue is paused right now, so the pane can say so
    /// whether or not they have rows.
    var pausedMachines: [MoldHost] {
        hosts.hosts.filter { gate.canPause(on: $0.id) && gate.isPaused(on: $0.id) }
    }

    /// Not `private`: `QueuePane+Commands.swift`'s Empty Queue… item reads
    /// this too, and `private` does not cross a file boundary.
    var emptyQueueTargets: [MoldHost] {
        Self.emptyQueueTargets(hosts.hosts, capabilities: hosts.capabilities,
                               entries: queue.byHost)
    }

    @ViewBuilder private var emptyQueueControl: some View {
        if emptyQueueTargets.count == 1, let host = emptyQueueTargets.first {
            Button("Empty Queue…") { confirmEmptyQueue(on: host) }
        } else {
            Menu("Empty Queue…") {
                Button(QueueEmptyConfirm.allMachinesItem) { confirmEmptyAllQueues() }
                Divider()
                ForEach(emptyQueueTargets) { host in
                    Button("Empty Queue on \(host.name)…") { confirmEmptyQueue(on: host) }
                }
            }
        }
    }

    /// Not `private`: `QueuePane+Commands.swift`'s Empty Queue… item calls
    /// this too.
    func confirmEmptyQueue(on host: MoldHost) {
        let counts = QueueEmptyConfirm.Counts(queue.entries(on: host.id))
        pendingDestruction = Destruction(
            title: QueueEmptyConfirm.title(host: host.name),
            message: QueueEmptyConfirm.message(counts),
            verb: "Cancel Jobs"
        ) {
            Task { await queue.empty(on: host.id) }
        }
    }

    /// Every machine at once -- ONE confirm naming the whole count, then each
    /// machine emptied concurrently and reporting its own failures.
    func confirmEmptyAllQueues() {
        let targets = emptyQueueTargets
        let counts = targets.reduce(QueueEmptyConfirm.Counts()) {
            $0 + QueueEmptyConfirm.Counts(queue.entries(on: $1.id))
        }
        pendingDestruction = Destruction(
            title: QueueEmptyConfirm.allMachinesTitle,
            message: QueueEmptyConfirm.message(counts, machines: targets.count),
            verb: "Cancel Jobs"
        ) {
            Task { await queue.emptyAll(targets.map(\.id)) }
        }
    }

    /// Pure: which machines Empty Queue can do something on. The bulk route
    /// is capability-gated, but clearing a HELD row is not -- it is the one
    /// action every hold has -- so a machine holding work is a target even
    /// when it does not advertise the bulk route. Absent otherwise (design
    /// decision 5) -- a test pins the gate without a rendered toolbar.
    static func emptyQueueTargets(
        _ hosts: [MoldHost], capabilities: [MoldHost.ID: Capabilities],
        entries: [MoldHost.ID: [QueueEntry]] = [:]
    ) -> [MoldHost] {
        hosts.filter { host in
            capabilities[host.id]?.canCancelAllQueued == true
                || (entries[host.id] ?? []).contains { $0.state == .held }
        }
    }
}
