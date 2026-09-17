import Foundation
import MoldClient
import Testing

@testable import Mold

/// S4b: Library, Performance, General's Storage-and-Reset absorption, and
/// the nine-tab layout that has to fit them all.
@MainActor
struct SettingsPanesTests {
    // MARK: - The window fits nine tabs

    /// **Fails red** at the old eight-tab width of 560 -- nine tabs at
    /// `minTabWidth` 72 need 648, which 560 does not have room for. Green
    /// once `SettingsLayout.width` is 700.
    @Test func everyTabFitsTheSettingsWindow() {
        #expect(Double(SettingsLayout.tabCount) * SettingsLayout.minTabWidth <= SettingsLayout.width)
        #expect(SettingsLayout.tabCount == SettingsTab.allCases.count)
    }

    // MARK: - A pane draws only what its machine reported

    /// Pure, the same `SettingRow.resolve` idiom `SettingRowTests` already
    /// uses for Generation: a listing missing one of `SettingKeys.library`'s
    /// keys draws nothing for it, and draws every key it does carry.
    @Test func aPaneDrawsOnlyTheKeysItsMachineReported() {
        let entries = SettingKeys.library
            .filter { $0.key != "gallery.authority_log" }
            .map { FakeFixtures.configEntry($0.key, value: .number(30), source: "db") }
        let listing = ConfigListing(entries: entries)

        for setting in SettingKeys.library {
            let entry = listing.entries.first { $0.key == setting.key }
            let plan = SettingRow.resolve(setting, entry: entry)
            if setting.key == "gallery.authority_log" {
                #expect(plan == nil, "an older machine's listing should draw no row for this key")
            } else {
                #expect(plan != nil, "\(setting.key) is in the listing and should draw")
            }
        }
    }

    // MARK: - The media cache control survived the move from Storage

    @Test func theMediaCacheControlSurvivedTheMoveFromStorage() {
        #expect(PrintMaterializer.capKey == "mediaCacheMegabytes")
        #expect(GeneralSettings.mediaCacheChoices == [256, 512, 1_024, 2_048, 4_096, 8_192])
    }

    // MARK: - Resetting preferences leaves the machine list alone

    private func scratch() -> UserDefaults {
        let name = "io.utensils.mold.native.tests.preferencesreset.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    /// The one that matters: `PreferencesReset.keys` never names the
    /// machine-list key (`HostPersistence`'s own `"hosts"`), and a real
    /// round trip through a throwaway suite proves it -- seeding both, then
    /// resetting, leaves `"hosts"` exactly as it was.
    @Test func resettingThisMacsPreferencesLeavesTheMachineListAlone() {
        #expect(!PreferencesReset.keys.contains("hosts"))

        let defaults = scratch()
        defaults.set("kept", forKey: "hosts")
        for key in PreferencesReset.keys { defaults.set("stale", forKey: key) }

        PreferencesReset.reset(in: defaults)

        #expect(defaults.string(forKey: "hosts") == "kept")
        for key in PreferencesReset.keys {
            #expect(defaults.object(forKey: key) == nil, "\(key) should have been cleared")
        }
    }

    // MARK: - An unknown settings tab opens the first one

    @Test func anUnknownSettingsTabOpensTheFirstOne() {
        #expect(SettingsUAT.initialTab(environment: [:]) == SettingsTab.allCases[0])
        #expect(
            SettingsUAT.initialTab(environment: [SettingsUAT.envVar: "not-a-real-tab"])
                == SettingsTab.allCases[0])
    }

    @Test func aKnownSettingsTabOpensItself() {
        #expect(SettingsUAT.initialTab(environment: [SettingsUAT.envVar: "performance"]) == .performance)
    }

    /// Setting the tab alone must OPEN Settings -- the design's "opens the
    /// Settings window on that tab at launch" -- not merely pick the tab if
    /// something else opens the window.
    @Test func namingATabOpensTheSettingsWindow() {
        #expect(SettingsUAT.wantsSettings(environment: ["MOLD_NATIVE_SETTINGS_TAB": "advanced"]))
        #expect(SettingsUAT.wantsSettings(environment: ["MOLD_NATIVE_SETTINGS_TAB": "bogus"]))
        #expect(!SettingsUAT.wantsSettings(environment: [:]))
    }
}
