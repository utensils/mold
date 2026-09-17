import MoldClient

/// The "Work here" / "Models here" figures on the Machines page.
///
/// Pulled into pure functions so the absent-vs-empty distinction is testable
/// without a live pane: `nil` means the owning store has never answered for
/// this host, which is a different fact than an empty answer, which means the
/// machine truly has none. Reading `nil` as empty said "None installed" about
/// a machine holding a dozen models, just because nobody had opened the
/// Models pane yet this launch.
enum MachineFigures {
    /// Shown while a count is not yet known -- an em dash, not a claim.
    static let notLoaded = "—"

    static func modelFigure(ready: [Model]?) -> String {
        guard let ready else { return notLoaded }
        guard !ready.isEmpty else { return "None installed" }
        let size = ready.compactMap(\.sizeGb).reduce(0, +)
        guard size > 0 else { return "\(ready.count) installed" }
        return "\(ready.count) installed · \(size.formatted(.number.precision(.fractionLength(1)))) GB"
    }

    static func workFigure(live: [QueueEntry]?) -> String {
        guard let live else { return notLoaded }
        guard !live.isEmpty else { return "Nothing queued" }
        let running = live.count { $0.state == .running }
        return "\(live.count - running) queued, \(running) running"
    }
}
