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

    /// The one settled answer for a status, or nil while anything is live.
    init?(settling status: BatchStatus) {
        guard status.isSettled else { return nil }
        let ordered = status.children.sorted { $0.index < $1.index }
        results = ordered.compactMap(\.result)
        failures = ordered.compactMap { child in
            child.result == nil ? (child.error ?? "The render didn't finish.") : nil
        }
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
