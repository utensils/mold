import Foundation

/// The container the app and its two extensions share (DESIGN.md §A): the host
/// list without secrets, the widget's recent-prints snapshot, pending batches,
/// and the Share extension's inbox. Compiled into all three targets.
enum AppGroup {
    static let identifier = "group.io.utensils.mold.companion"

    /// `nil` only when the entitlement is missing -- a signing mistake, never
    /// a state to recover from quietly.
    static var container: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: identifier)
    }
}
