import Foundation

/// The beat between a batch settling and the next one taking the canvas.
///
/// `settle` used to call `followNext()` in the same turn, on the strength of a
/// comment claiming the `.finished` write "is observed for at least one beat".
/// Nothing guarantees that: `followNext` enqueues a main-actor `Task` and
/// SwiftUI coalesces `@Observable` invalidations into one render pass, so two
/// quick presses of Generate could replace the first batch's picture -- with
/// its `ResultStrip`, its `ResultBar` and any `failureSummary` -- before any
/// of it was ever drawn. `run` is a single slot, so there was no way back to
/// it (finding 02#9).
///
/// So the queue waits for the canvas to say it has the result. The grace
/// period is the belt: the Generate pane can be off screen entirely (the
/// Library is a tab away), and a queue that waits for a view nobody is looking
/// at would never advance at all.
@MainActor
final class ResultHandoff {
    private var release: (() -> Void)?
    private var timeout: Task<Void, Never>?
    private let grace: Duration

    /// A constructor parameter rather than a constant, so a test pins the
    /// behaviour without sleeping through it.
    init(grace: Duration = .milliseconds(1500)) {
        self.grace = grace
    }

    /// Holds `advance` until the canvas acknowledges the result, or until the
    /// grace period runs out. A second hold releases the first: the outcome it
    /// was waiting on has been superseded by this one.
    func hold(_ advance: @escaping () -> Void) {
        timeout?.cancel()
        release?()
        release = advance
        timeout = Task { [weak self] in
            try? await Task.sleep(for: self?.grace ?? .zero)
            guard !Task.isCancelled else { return }
            self?.acknowledge()
        }
    }

    /// The canvas has the finished result on screen. Called from the view, so
    /// the queue advances on a real render rather than on a hope about one.
    func acknowledge() {
        timeout?.cancel()
        timeout = nil
        let advance = release
        release = nil
        advance?()
    }

    /// Nothing is waiting any more -- a Stop, or a pane going away.
    func cancel() {
        timeout?.cancel()
        timeout = nil
        release = nil
    }

    var isHolding: Bool { release != nil }
}
