import Foundation

/// The Settings window's fixed size, and the rule that earns it.
///
/// Nine tabs (`SettingsView`) must each have room for a `tabItem` label
/// without SwiftUI collapsing the bar into an overflow chevron.
/// `SettingsLayoutTests.everyTabFitsTheSettingsWindow` is the plan's own
/// rule that a layout constant gets a test asserting the space it draws into
/// actually fits what it draws -- it fails red at 560, the width nine tabs
/// inherited from eight, before this file widens it to 700.
enum SettingsLayout {
    // `Double`, not `CGFloat` -- this file stays UI-free so
    // `SettingsLayoutTests` can check the arithmetic with no SwiftUI import;
    // `SettingsView` converts at its one `.frame` call.
    static let width: Double = 700
    static let height: Double = 560
    static let tabCount = 9
    static let minTabWidth: Double = 72
}
