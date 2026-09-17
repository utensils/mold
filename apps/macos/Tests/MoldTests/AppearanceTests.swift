import AppKit
import Foundation
import Testing

@testable import Mold

/// Settings ▸ General ▸ Appearance: System, Light or Dark, applied to the
/// whole application rather than one window, so the Settings window and
/// the main window never disagree. The mapping to AppKit's appearance
/// names is the one fact worth pinning -- `nil` is "follow the system",
/// never a third named appearance.
@MainActor
struct AppearanceTests {
    @Test func systemMeansNoOverride() {
        #expect(Appearance.system.nsAppearanceName == nil)
    }

    @Test func lightAndDarkNameAppKitsOwnAppearances() {
        #expect(Appearance.light.nsAppearanceName == .aqua)
        #expect(Appearance.dark.nsAppearanceName == .darkAqua)
    }

    @Test func theChoicesAreOfferedSystemFirst() {
        #expect(Appearance.allCases == [.system, .light, .dark])
        #expect(Appearance.allCases.map(\.label) == ["System", "Light", "Dark"])
    }

    /// A value this build has never heard of (a future fourth choice, or a
    /// hand-edited plist) reads as System rather than crashing or sticking.
    @Test func anUnknownStoredValueReadsAsSystem() {
        let suite = UserDefaults(suiteName: "io.utensils.mold.tests.appearance")!
        suite.removePersistentDomain(forName: "io.utensils.mold.tests.appearance")
        #expect(Appearance.stored(in: suite) == .system)
        suite.set("sepia", forKey: Appearance.key)
        #expect(Appearance.stored(in: suite) == .system)
        suite.set(Appearance.dark.rawValue, forKey: Appearance.key)
        #expect(Appearance.stored(in: suite) == .dark)
        suite.removePersistentDomain(forName: "io.utensils.mold.tests.appearance")
    }
}
