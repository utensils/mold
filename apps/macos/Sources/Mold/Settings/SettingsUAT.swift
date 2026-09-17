import Foundation

/// `MOLD_NATIVE_SETTINGS_TAB=<tab id>` opens Settings on that tab at launch,
/// mirroring `MachinesSettings.openEditorIfRequested`'s own precedent
/// (`MOLD_NATIVE_DESTINATION`) -- nine deterministic captures instead of
/// nine menu presses.
enum SettingsUAT {
    static let envVar = NativeUAT.settingsTab.rawValue

    /// Pure: whether the hook asks for the Settings window at all -- naming
    /// a tab, even an unknown one, opens the window (`RootView` reads it
    /// beside `MOLD_NATIVE_DESTINATION=settings`); picking the tab alone
    /// would wait on a window nothing opens.
    static func wantsSettings(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        NativeUAT.settingsTab.isSet(in: environment)
    }

    /// Pure: which tab id to open, from the environment and the tabs that
    /// exist. An unknown or absent id opens the first tab, never a crash or
    /// a blank window.
    static func initialTab(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> SettingsTab {
        NativeUAT.settingsTab.value(in: environment)
            .flatMap(SettingsTab.init(rawValue:)) ?? SettingsTab.allCases[0]
    }
}
