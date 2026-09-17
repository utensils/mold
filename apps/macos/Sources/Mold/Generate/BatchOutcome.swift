import MoldClient

/// What a settled batch produced.
///
/// A batch can end part-done -- one child holds for a missing component while
/// three render -- and somebody who asked for four pictures and got three
/// must be told which is which, in one place, rather than shown the first
/// child and left to find the rest in the Queue.
struct BatchOutcome: Equatable {
    /// In child `index` order, which is 1-based on the wire.
    let results: [BatchResult]
    /// One sentence per child that made nothing.
    let failures: [String]

    /// The one answer for a status that has come to rest, or nil while
    /// anything can still move on its own. A HELD child counts as at rest
    /// (`BatchStatus.isAtRest`): the machine has parked it until someone
    /// decides in the Queue, and its sentence says so -- a pane that kept
    /// waiting for it spun "Getting ready…" for ever, and again at every
    /// launch after.
    init?(settling status: BatchStatus) {
        guard status.isAtRest else { return nil }
        let ordered = status.children.sorted { $0.index < $1.index }
        results = ordered.compactMap(\.result)
        failures = ordered.compactMap { child in
            guard child.result == nil else { return nil }
            if child.state == .held { return Self.heldSentence(child.error) }
            return child.error ?? "The render didn't finish."
        }
    }

    /// The machine's own reason, then where the decision lives.
    static func heldSentence(_ reason: String?) -> String {
        let cause = reason.map { "The machine is holding this render: \($0)" }
            ?? "The machine is holding this render"
        let stop = cause.hasSuffix(".") ? "" : "."
        return "\(cause)\(stop) Try it again, move it or cancel it in the Queue."
    }
}

extension BatchOutcome {
    /// "One of four didn't finish." -- built from counts, never from a
    /// child's own reason, because three pictures arrived and the pane's job
    /// here is to show them, not to explain the fourth.
    var failureSummary: String? {
        guard !failures.isEmpty else { return nil }
        return "\(Self.counted(failures.count).capitalized) of "
            + "\(Self.counted(results.count + failures.count)) didn't finish."
    }

    private static let words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ]

    private static func counted(_ n: Int) -> String {
        n < words.count ? words[n] : String(n)
    }
}
