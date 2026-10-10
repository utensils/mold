import SwiftUI
import Testing
import UIKit
@testable import MoldCompanion

@MainActor struct AppearanceTests {
    @Test func choicesMapToSystemOrExplicitScheme() {
        #expect(AppAppearance.allCases == [.system, .light, .dark])
        #expect(AppAppearance.system.interfaceStyle == .unspecified)
        #expect(AppAppearance.light.interfaceStyle == .light)
        #expect(AppAppearance.dark.interfaceStyle == .dark)
    }

    @Test func changingToSystemClearsTheWindowOverride() {
        let window = UIWindow()
        let probe = AppearanceWindow.Probe(frame: .zero)
        probe.appearance = .dark
        window.addSubview(probe)
        #expect(window.overrideUserInterfaceStyle == .dark)
        probe.appearance = .light
        #expect(window.overrideUserInterfaceStyle == .light)
        probe.appearance = .system
        #expect(window.overrideUserInterfaceStyle == .unspecified)
    }

    @Test func preferencePersistsAndUnknownValuesFollowSystem() throws {
        let name = "AppearanceTests-\(UUID())"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        let preference = AppStorage(wrappedValue: AppAppearance.system,
                                    Preference.appearance, store: defaults)
        #expect(preference.wrappedValue == .system)
        preference.wrappedValue = .dark
        let restored = AppStorage(wrappedValue: AppAppearance.system,
                                  Preference.appearance, store: defaults)
        #expect(restored.wrappedValue == .dark)
        defaults.set("unrecognized", forKey: Preference.appearance)
        let fallback = AppStorage(wrappedValue: AppAppearance.system,
                                 Preference.appearance, store: defaults)
        #expect(fallback.wrappedValue == .system)
    }
}
