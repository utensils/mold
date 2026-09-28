import Foundation
import MoldClient

/// What the Live Activity says for a render, derived from the same state the
/// canvas draws -- pure, so each phase is tested without ActivityKit.
enum ActivityProjection {
    /// `nil` while there is nothing to show (idle, still submitting).
    static func state(for run: RunState, machine: String, waiting: Int, remaining: Duration?,
                      preview: String?, now: Date = .now) -> GenerationActivityAttributes.ContentState? {
        switch run {
        case .idle, .submitting:
            return nil
        case let .running(status, progress):
            // Still waiting its turn only while nothing in it has started.
            let position = status.children.allSatisfy { $0.state == .accepted } ? progress?.queuePosition : nil
            return .init(phase: .running,
                         sentence: ProgressWords.sentence(progress, position: position, remaining: nil),
                         figure: ProgressWords.figure(progress).map { "\($0) · \(machine)" } ?? machine,
                         step: progress?.step, total: progress?.total,
                         endsAt: remaining.map { now.addingTimeInterval(TimeInterval($0.components.seconds)) },
                         preview: preview, waiting: waiting)
        case let .finished(outcome, _):
            let count = outcome.results.count
            return .init(phase: .finished,
                         sentence: count > 1 ? String(localized: "\(count) finished on \(machine)")
                                             : String(localized: "Finished on \(machine)"),
                         figure: nil, step: nil, total: nil, endsAt: nil, preview: preview, waiting: waiting,
                         print: outcome.results.first?.filename)
        case let .failed(reason):
            return .init(phase: .failed, sentence: reason, figure: nil, step: nil, total: nil,
                         endsAt: nil, preview: nil, waiting: waiting)
        }
    }

    /// When the Lock Screen should call it stale if nothing updates it: five
    /// minutes past the estimate, or ten minutes from now without one. The
    /// server cannot push, so this is the honest limit (DESIGN.md §5.7).
    static func staleDate(for state: GenerationActivityAttributes.ContentState, now: Date = .now) -> Date? {
        guard state.phase == .running else { return nil }
        return (state.endsAt ?? now.addingTimeInterval(5 * 60)).addingTimeInterval(5 * 60)
    }

    /// A finished or failed render leaves the Lock Screen after 15 minutes.
    static let dismissAfter: TimeInterval = 15 * 60
}
