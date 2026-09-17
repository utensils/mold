import AppKit
import Foundation

/// The number on the Dock icon, kept in step with `LandedPrints`.
///
/// It used to be an `.onChange(of:)` on `RootView` -- so the badge tracked
/// the count only while that view was mounted. `LandedPrints` keeps counting
/// whatever the window is doing, and the app does not terminate when its last
/// window closes, so closing it left the badge frozen at whatever it showed.
/// The Dock is an APPLICATION surface; its observer belongs beside the app
/// delegate, not on a view.
///
/// `LandedPrints` stays AppKit-free on purpose -- it is testable with no app
/// bundle at all -- so this is the one small object that knows about both.
@MainActor
final class DockBadge {
    /// What actually paints the Dock. A parameter so a test reads the label
    /// it was handed instead of an app's real tile.
    private let apply: (String?) -> Void
    private var observation: Task<Void, Never>?

    static let dockTile: (String?) -> Void = { NSApp.dockTile.badgeLabel = $0 }

    init(apply: @escaping (String?) -> Void = DockBadge.dockTile) {
        self.apply = apply
    }

    deinit { observation?.cancel() }

    /// The label for a count. Zero is NO badge, not a badge reading "0".
    static func label(for count: Int) -> String? {
        count > 0 ? "\(count)" : nil
    }

    /// Follows `LandedPrints.count` for the life of the app.
    ///
    /// `withObservationTracking` fires once per change, so it re-arms itself:
    /// the alternative is a timer, and the whole point of `@Observable` is
    /// that there does not have to be one.
    func follow(_ landedPrints: LandedPrints) {
        observation?.cancel()
        paint(landedPrints.count)
        observation = Task { [weak self] in
            while !Task.isCancelled {
                await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                    withObservationTracking {
                        _ = landedPrints.count
                    } onChange: {
                        continuation.resume()
                    }
                }
                guard let self, !Task.isCancelled else { return }
                // `onChange` fires with the OLD value still in place (it is
                // `willSet`, not `didSet`), and resuming a continuation only
                // enqueues this task -- so the read below already happens on
                // a later turn of the main actor. The yield says that out
                // loud rather than relying on it.
                await Task.yield()
                paint(landedPrints.count)
            }
        }
    }

    private func paint(_ count: Int) {
        apply(Self.label(for: count))
    }
}
