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

    /// `alsoRunning` is what the machine is doing that never becomes a queue
    /// row -- a prompt rewrite, a standalone upscale, a sequence. "Nothing
    /// queued" over a machine mid-rewrite read as idle (2026-09-17).
    static func workFigure(live: [QueueEntry]?, alsoRunning: Int = 0) -> String {
        guard let live else { return notLoaded }
        let queued: String
        if live.isEmpty {
            queued = "Nothing queued"
        } else {
            let running = live.count { $0.state == .running }
            queued = "\(live.count - running) queued, \(running) running"
        }
        guard alsoRunning > 0 else { return queued }
        return "\(queued) · \(alsoRunning) also running"
    }
}
