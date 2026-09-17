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
            Button(machine.title) { gate.toggle(machine.id) }
        } else {
            Menu("Queue") {
                RowActionMenu(actions: gate.items(), perform: gate.toggle)
            }
        }
    }

    /// The machines whose queue is paused right now, so the pane can say so
    /// whether or not they have rows.
    var pausedMachines: [MoldHost] {
        hosts.hosts.filter { queue.canPauseQueue(on: $0.id) && queue.isQueuePaused(on: $0.id) }
    }

    /// Not `private`: `QueuePane+Commands.swift`'s Empty Queue… item reads
    /// this too, and `private` does not cross a file boundary.
    var emptyQueueTargets: [MoldHost] {
        Self.emptyQueueTargets(hosts.hosts, capabilities: hosts.capabilities)
    }

    @ViewBuilder private var emptyQueueControl: some View {
        if emptyQueueTargets.count == 1, let host = emptyQueueTargets.first {
            Button("Empty Queue…") { confirmEmptyQueue(on: host) }
        } else {
            Menu("Empty Queue…") {
                ForEach(emptyQueueTargets) { host in
                    Button("Empty Queue on \(host.name)…") { confirmEmptyQueue(on: host) }
                }
            }
        }
    }

    /// Not `private`: `QueuePane+Commands.swift`'s Empty Queue… item calls
    /// this too.
    func confirmEmptyQueue(on host: MoldHost) {
        let entries = queue.entries(on: host.id)
        let waiting = entries.filter { $0.state == .queued }.count
        let paused = entries.filter { $0.state == .paused }.count
        pendingDestruction = Destruction(
            title: QueueEmptyConfirm.title(host: host.name),
            message: QueueEmptyConfirm.message(waiting: waiting, paused: paused),
            verb: "Cancel Jobs"
        ) {
            Task { await queue.cancelAll(on: host.id) }
        }
    }

    /// Pure: which machines actually offer this. Absent means `false`
    /// (design decision 5) -- a test pins the gate without a rendered
    /// toolbar.
    static func emptyQueueTargets(
        _ hosts: [MoldHost], capabilities: [MoldHost.ID: Capabilities]
    ) -> [MoldHost] {
        hosts.filter { capabilities[$0.id]?.canCancelAllQueued == true }
    }
}

/// The subtitle's own words -- fleet-wide, and pure so a test can pin the
/// exact wording without a rendered pane. A held row is never "waiting": it
/// is not going anywhere until something about it changes.
enum QueueSummary {
    static func sentence(_ entries: [QueueEntry]) -> String {
        func count(_ state: QueueState) -> Int { entries.filter { $0.state == state }.count }
        let clauses = [
            (count(.queued), "waiting"), (count(.running), "rendering"), (count(.held), "held"),
        ].compactMap { n, word in n > 0 ? "\(n) \(word)" : nil }
        return clauses.isEmpty ? "Idle" : clauses.joined(separator: " · ")
    }
}

/// The Empty Queue confirm's own sentence -- fact 12's whole point: running
/// work is untouched, and the confirm has to say so or it reads as "stop
/// everything".
enum QueueEmptyConfirm {
    static func title(host: String) -> String { "Cancel everything waiting on \(host)?" }

    static func message(waiting: Int, paused: Int) -> String {
        let waitingWord = waiting == 1 ? "1 waiting" : "\(waiting) waiting"
        let pausedWord = paused == 1 ? "1 paused job" : "\(paused) paused jobs"
        return "\(waitingWord) and \(pausedWord) will be cancelled. Anything already rendering keeps going."
    }
}
