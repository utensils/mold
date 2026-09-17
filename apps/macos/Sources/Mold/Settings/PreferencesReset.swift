import Foundation

/// What "Reset these preferences" on Settings ▸ General clears, and what it
/// deliberately leaves alone.
///
/// This is per-Mac UI state -- the remembered destination, the remembered
/// machine, the sidebar and inspector layout, and the Models pane's own
/// sort/scope -- never anything a machine answers for and never the machine
/// list itself (`HostPersistence`'s own `"hosts"` key, plus the API keys it
/// keeps in the Keychain), which is the one thing in this suite somebody
/// would mind losing. The media-cache cap (`PrintMaterializer.capKey`), the
/// appearance (`Appearance.key`) and the two notification toggles
/// (`GeneralSettings`) are also left alone --
/// each already has its own control right on this same page, so a second,
/// wholesale way to change them would only make the page harder to reason
/// about.
enum PreferencesReset {
    static let keys: [String] = [
        "destination",
        "sidebarVisibility",
        "selectedMachine",
        "generateShowsInspector",
        "libraryShowsInspector",
        "inspectorShowsProvenance",
        "createShowsAdapters",
        "createShowsIdentity",
        "createShowsRefine",
        "createShowsClip",
        "createShowsOutput",
        "createShowsFileUnder",
        "createShowsRecent",
        "modelsSortColumn",
        "modelsSortAscending",
        "modelsScope",
    ]

    static func reset(in defaults: UserDefaults) {
        for key in keys { defaults.removeObject(forKey: key) }
    }
}
