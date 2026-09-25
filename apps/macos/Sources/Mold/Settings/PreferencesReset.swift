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
        "createShowsSampler",
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
    /// The media-cache cap, the appearance, the two notification toggles and
    /// the update channel each already have their own control on this very
    /// page, so a second,
    /// wholesale way to change them would only make the page harder to reason
    /// about -- and a reset that quietly moved somebody from Nightly back to
    /// Stable would be a channel change nobody asked for. The machine list is
    /// somebody's setup. `pendingBatches` and `pendingChainJobs` are in-flight
    /// recovery bookkeeping, not preferences -- clearing them strands a batch
    /// or a chain the app is still waiting on. Library sync copy records and
    /// pending organization are recovery bookkeeping too. The Keychain migration flag is
    /// a fact about this install: clearing it would re-read the old Keychain
    /// items and could resurrect a key the person has since removed.
    ///
    /// SPARKLE'S OWN KEYS are deliberately out of scope and cannot be listed
    /// here: `SUEnableAutomaticChecks`, `SUAutomaticallyUpdate`,
    /// `SUScheduledCheckInterval`, `SUSendProfileInfo` and `SULastCheckTime`
    /// are written by `SPUUpdaterSettings`, not by mold, so no `forKey:` for
    /// them exists in these sources and the test below is blind to them. They
    /// stay for the reason the two toggles above do -- the Updates group on
    /// that same page owns them -- and clearing them behind Sparkle's back
    /// would desynchronise its scheduler from what it believes it agreed with
    /// the user (review F5#6).
    static let kept: Set<String> = [
        "appearance",
        "badgeLandedPrints",
        "notifyRenders",
        "mediaCacheMegabytes",
        "updateChannel",
        "librarySyncCopiesV1",
        "librarySyncPendingOrganizationV1",
        "hosts",
        "hosts.unreadable",
        "pendingBatches",
        "pendingChainJobs",
        "keychainKeysMigrated",
        // Its own toggle on Settings ▸ This Mac; a reset quietly turning the
        // engine back on at launch would be a change nobody asked for.
        "engineStartsAtLaunch",
    ]

    static func reset(in defaults: UserDefaults) {
        for key in keys { defaults.removeObject(forKey: key) }
        // UserDefaults may coalesce its own notification. Reconcile observers
        // with the completed reset now, before a stale in-memory choice can
        // continue routing work to the machine the person just cleared.
        NotificationCenter.default.post(name: UserDefaults.didChangeNotification, object: defaults)
    }
}
