import Foundation

/// The Settings switches (DESIGN.md §5.6), as `@AppStorage` keys with their
/// defaults: notifications and Live Activities on, auto-save to Photos off.
enum Preference {
    static let appearance = "appearance"
    static let notifyFinished = "notify.finished"
    static let notifyFailed = "notify.failed"
    static let notifyHeld = "notify.held"
    static let liveActivities = "liveActivities"
    static let autoSaveToPhotos = "library.autoSave"
    static let showDateSeparators = "library.showDateSeparators"
    /// Megabytes the offline library may take (`OfflineLimit`).
    static let offlineLimit = "library.offlineLimitMB"

    static func isOn(_ key: String, defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: key) as? Bool ?? (key != autoSaveToPhotos)
    }
}

/// How much of this device the offline library may use: thumbnails and the
/// prints opened in the viewer, together. The listing itself is small and
/// always kept.
enum OfflineLimit: Int, CaseIterable, Identifiable {
    case mb250 = 250, mb500 = 500, gb1 = 1_000, gb2 = 2_000, gb5 = 5_000

    static let standard = OfflineLimit.gb1
    var id: Int { rawValue }
    var bytes: Int { rawValue * 1_000_000 }

    var title: String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }

    /// Thumbnails take 30% (they are what makes the grid work offline),
    /// opened prints the rest.
    var thumbnailBytes: Int { bytes * 3 / 10 }
    var originalBytes: Int { bytes - thumbnailBytes }

    static func current(_ defaults: UserDefaults = .standard) -> OfflineLimit {
        OfflineLimit(rawValue: defaults.integer(forKey: Preference.offlineLimit)) ?? .standard
    }
}
