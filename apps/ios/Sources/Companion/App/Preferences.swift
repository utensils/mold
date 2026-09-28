import Foundation

/// The Settings switches (DESIGN.md §5.6), as `@AppStorage` keys with their
/// defaults: notifications and Live Activities on, auto-save to Photos off.
enum Preference {
    static let notifyFinished = "notify.finished"
    static let notifyFailed = "notify.failed"
    static let notifyHeld = "notify.held"
    static let liveActivities = "liveActivities"
    static let autoSaveToPhotos = "library.autoSave"

    static func isOn(_ key: String, defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: key) as? Bool ?? (key != autoSaveToPhotos)
    }
}
