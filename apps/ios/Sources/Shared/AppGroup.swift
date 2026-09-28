import Foundation

/// The container the app and its two extensions share (DESIGN.md §A): the host
/// list without secrets, the widget's recent-prints snapshot, pending batches,
/// and the Share extension's inbox. Compiled into all three targets.
nonisolated enum AppGroup {
    static let identifier = "group.io.utensils.mold.companion"

    /// `nil` only when the entitlement is missing -- a signing mistake, never
    /// a state to recover from quietly.
    static var container: URL? {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: identifier)
    }

    /// One folder inside the group, created on first use. Outside a signed
    /// build (unit tests) the group is a temporary folder instead.
    static func directory(_ name: String) -> URL {
        let root = container ?? FileManager.default.temporaryDirectory.appending(path: "mold-app-group")
        let url = root.appending(path: name, directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Live Activity previews, by batch.
    static var activityPreviews: URL { directory("activity") }
    /// The widgets' snapshot and its thumbnails.
    static var widget: URL { directory("widget") }
    /// Photos the Share extension staged for the app.
    static var shareInbox: URL { directory("share-inbox") }
}
