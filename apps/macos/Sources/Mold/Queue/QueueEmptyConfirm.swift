import MoldClient

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
