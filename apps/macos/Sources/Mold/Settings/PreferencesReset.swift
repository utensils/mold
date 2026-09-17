import Foundation

/// What "Reset These Preferences" on Settings ▸ General clears, and what it
/// deliberately leaves alone.
///
/// This is per-Mac UI state -- the remembered destination, the machine each
/// pane is pointed at and the one new renders go to, the sidebar and inspector
/// layout, the thumbnail size, and every pane's own sort and scope -- never
/// anything a machine answers for and never the machine list itself
/// (`HostPersistence`'s own `"hosts"`, plus the keys `SecretStore` holds),
/// which is the one thing in this suite somebody would mind losing.
///
/// The two lists below must between them name EVERY key this app writes to
/// the suite: `PreferencesResetTests` reads the sources and fails on one that
/// is in neither. Four were in neither before that test existed -- the library
/// scope and thumbnail size, the Generate machine and the default machine --
/// so the button did not do what its own sentence said (review 05-M13).
enum PreferencesReset {
    static let keys: [String] = [
        "destination",
        "sidebarVisibility",
        "selectedMachine",
        "defaultMachine",
        "generateMachine",
        "generateShowsInspector",
        "libraryShowsInspector",
        "libraryScope",
        "libraryEdge",
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

    /// Written to the same suite, and deliberately untouched.
    ///
    /// The media-cache cap, the appearance and the two notification toggles
    /// each already have their own control on this very page, so a second,
    /// wholesale way to change them would only make the page harder to reason
    /// about. The machine list is somebody's setup. `pendingBatches` is
    /// in-flight recovery bookkeeping, not a preference -- clearing it strands
    /// a batch the app is still waiting on. And the Keychain migration flag is
    /// a fact about this install: clearing it would re-read the old Keychain
    /// items and could resurrect a key the person has since removed.
    static let kept: Set<String> = [
        "appearance",
        "badgeLandedPrints",
        "notifyRenders",
        "mediaCacheMegabytes",
        "hosts",
        "hosts.unreadable",
        "pendingBatches",
        "keychainKeysMigrated",
    ]

    static func reset(in defaults: UserDefaults) {
        for key in keys { defaults.removeObject(forKey: key) }
    }
}
