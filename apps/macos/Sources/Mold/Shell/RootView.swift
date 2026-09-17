import SwiftUI

/// The window's one split view.
///
/// A single `Window` rather than a `WindowGroup`: the Library merges every
/// host's index into one in-memory timeline, and a second window would mean a
/// second copy of it arguing with the first.
struct RootView: View {
    @Environment(HostStore.self) private var hosts
    @Environment(DownloadStore.self) private var downloads
    @Environment(\.openSettings) private var openSettings
    /// Reopening where you left off is what every Mac app does. The env
    /// override exists so a UAT run can land on a named destination without
    /// driving the mouse.
    @AppStorage("destination", store: AppStorageSuite.defaults) private var stored = Destination.generate.rawValue
    /// Owned by the scene so a menu command can change it.
    @Binding var destination: Destination

    var body: some View {
        NavigationSplitView {
            Sidebar(destination: $destination)
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 340)
        } detail: {
            DestinationDetail(destination: $destination)
        }
        .navigationTitle("Mold")
        .task { await hosts.refreshAll() }
        .task { openSettingsIfRequested() }
        // `HostStore` cannot reach `DownloadStore` -- it is the root every
        // store is built from, not a peer. So the machine list is watched
        // HERE rather than in the Models pane: a stream for a machine that is
        // gone must stop when it goes, not when somebody next opens Models.
        .onChange(of: hosts.hosts) { _, _ in downloads.reconcile() }
        .onChange(of: destination) { _, new in stored = new.rawValue }
    }
}

extension RootView {
    /// `MOLD_NATIVE_DESTINATION=settings` opens the Settings window on launch,
    /// and `=add-machine` opens it with the host sheet already up.
    ///
    /// Settings is a scene, not a destination, so it cannot be reached by
    /// selecting a sidebar row -- this is what lets a UAT run photograph it
    /// without driving the menu bar.
    func openSettingsIfRequested() {
        let requested = ProcessInfo.processInfo.environment["MOLD_NATIVE_DESTINATION"]
        let sheets = [MachinesSettings.addOnLaunch, MachinesSettings.editOnLaunch]
        guard requested == "settings" || sheets.contains(requested ?? "") else { return }
        openSettings()
    }
}

enum Destination: String, Hashable, CaseIterable, Identifiable {
    case generate, library, queue, models

    var id: Self { self }

    /// Where the window opens: an explicit override, else where you were last.
    static var launch: Destination {
        if let named = ProcessInfo.processInfo.environment["MOLD_NATIVE_DESTINATION"],
           let forced = Destination(rawValue: named) {
            return forced
        }
        let remembered = AppStorageSuite.defaults.string(forKey: "destination")
        return remembered.flatMap(Destination.init(rawValue:)) ?? .generate
    }

    var title: String {
        switch self {
        case .generate: "Generate"
        case .library: "Library"
        case .queue: "Queue"
        // "Models", never "Styles". A person choosing one needs to know what
        // it does, and the manifest already says so in plain language.
        case .models: "Models"
        }
    }

    var symbol: String {
        switch self {
        case .generate: "wand.and.sparkles"
        case .library: "photo.on.rectangle.angled"
        case .queue: "list.bullet.indent"
        case .models: "cube"
        }
    }
}
