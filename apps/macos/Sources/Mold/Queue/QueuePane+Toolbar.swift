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
/// everything". Held rows go too: they are not going anywhere on their own,
/// and leaving them is what made Empty Queue look broken.
enum QueueEmptyConfirm {
    /// The verb every failure about this is keyed on.
    static let verb = "empty its queue"
    static let allMachinesItem = "Empty Queue on All Machines…"
    static let allMachinesTitle = "Empty the queue on every machine?"

    struct Counts: Equatable {
        var waiting = 0, paused = 0, held = 0

        init(waiting: Int = 0, paused: Int = 0, held: Int = 0) {
            (self.waiting, self.paused, self.held) = (waiting, paused, held)
        }

        init(_ entries: [QueueEntry]) {
            func count(_ state: QueueState) -> Int { entries.filter { $0.state == state }.count }
            self.init(waiting: count(.queued), paused: count(.paused), held: count(.held))
        }

        static func + (lhs: Self, rhs: Self) -> Self {
            Self(waiting: lhs.waiting + rhs.waiting, paused: lhs.paused + rhs.paused,
                 held: lhs.held + rhs.held)
        }
    }

    static func title(host: String) -> String { "Empty the queue on \(host)?" }

    /// "3 waiting, 1 paused and 12 held jobs will be cancelled." Zero
    /// clauses are left out; the noun follows the LAST clause's count.
    static func message(_ counts: Counts, machines: Int = 1) -> String {
        let rendering = "Anything already rendering keeps going."
        let clauses = [(counts.waiting, "waiting"), (counts.paused, "paused"), (counts.held, "held")]
            .filter { $0.0 > 0 }
        guard let last = clauses.last else {
            return "Nothing is waiting or held right now. \(rendering)"
        }
        let words = clauses.map { "\($0.0) \($0.1)" }
        let list = words.count == 1
            ? words[0]
            : words.dropLast().joined(separator: ", ") + " and " + words[words.count - 1]
        let noun = last.0 == 1 ? "job" : "jobs"
        let across = machines > 1 ? " across \(machines) machines" : ""
        return "\(list) \(noun)\(across) will be cancelled. \(rendering)"
    }
}
