import Foundation

/// The Settings window's minimum size, and the rule that earns it.
///
/// Ten tabs (`SettingsView`) must each have room for a `tabItem` label
/// without SwiftUI collapsing the bar into an overflow chevron.
/// `SettingsPanesTests.everyTabFitsTheSettingsWindow` is the plan's own
/// rule that a layout constant gets a test asserting the space it draws into
/// actually fits what it draws. Remote Access adds a tenth destination,
/// so the window reserves room for the longer label too.
enum SettingsLayout {
    // `Double`, not `CGFloat` -- this file stays UI-free so the test can
    // check the arithmetic with no SwiftUI import;
    // `SettingsView` converts at its one `.frame` call.
    static let width: Double = 820
    static let height: Double = 560
    static let tabCount = 10
    static let minTabWidth: Double = 72
}
