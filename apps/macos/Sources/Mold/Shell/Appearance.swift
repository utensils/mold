import AppKit
import Foundation

/// Settings ▸ General ▸ Appearance: follow the system, or hold Light or Dark.
///
/// Applied to the APPLICATION (`NSApp.appearance`), not to a window's
/// content: the main window and the Settings window are two scenes, and a
/// `preferredColorScheme` on one would leave the other following the
/// system on its own. Setting it to `nil` hands the choice back to macOS,
/// which is what System means -- there is no third named appearance.
///
/// This is not a brand palette. Every colour in the app is still the
/// system's (`make lint` refuses a literal one); this only picks which of
/// the system's two appearances the app draws in.
enum Appearance: String, CaseIterable, Identifiable {
    case system
    case light
    case dark

    static let key = "appearance"

    var id: String { rawValue }

    var label: String {
        switch self {
        case .system: "System"
        case .light: "Light"
        case .dark: "Dark"
        }
    }

    /// `nil` is "follow the system".
    var nsAppearanceName: NSAppearance.Name? {
        switch self {
        case .system: nil
        case .light: .aqua
        case .dark: .darkAqua
        }
    }

    /// An unknown stored value -- a future choice, a hand-edited plist --
    /// reads as System rather than sticking or crashing.
    static func stored(in defaults: UserDefaults) -> Appearance {
        defaults.string(forKey: key).flatMap(Appearance.init(rawValue:)) ?? .system
    }

    /// Puts this appearance on every window the app has and will open.
    @MainActor
    func apply() {
        NSApp.appearance = nsAppearanceName.flatMap(NSAppearance.init(named:))
    }
}
