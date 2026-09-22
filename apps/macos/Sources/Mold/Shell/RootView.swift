import Foundation
import SwiftUI

/// The window's one split view.
///
/// A single `Window` rather than a `WindowGroup`: the Library merges every
/// host's index into one in-memory timeline, and a second window would mean a
/// second copy of it arguing with the first.
struct RootView: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @Environment(DownloadStore.self) private var downloads
    @Environment(\.openSettings) private var openSettings
    /// Reopening where you left off is what every Mac app does. The env
    /// override exists so a UAT run can land on a named destination without
    /// driving the mouse.
    @AppStorage("destination", store: AppStorageSuite.defaults) private var stored = Destination.generate.rawValue
    /// So does a collapsed sidebar. `@State` initializers cannot read another
    /// property wrapper, so the seed comes straight from the suite -- the same
    /// way `Destination.launch` reads where you were last.
    @AppStorage("sidebarVisibility", store: AppStorageSuite.defaults)
    private var storedVisibility = NavigationSplitViewVisibility.all.stored
    @State private var columnVisibility = NavigationSplitViewVisibility(
        stored: AppStorageSuite.defaults.string(forKey: "sidebarVisibility"))
    /// Owned by the scene so a menu command can change it.
    @Binding var destination: Destination

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            Sidebar(destination: $destination)
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 340)
        } detail: {
            DestinationDetail(destination: $destination)
        }
        .navigationTitle("Mold Studio")
        .task { await hosts.refreshAll() }
        // Shelves and their counts belong to the shell, regardless of which
        // destination opens first. Keep this independent of host probes so
        // an unreachable machine cannot postpone reading reachable ones.
        .task(id: hosts.hosts) { await library.reload() }
        .task { openSettingsIfRequested() }
        // `HostStore` cannot reach `DownloadStore` -- it is the root every
        // store is built from, not a peer. So the machine list is watched
        // HERE rather than in the Models pane: a stream for a machine that is
        // gone must stop when it goes, not when somebody next opens Models.
        .onChange(of: hosts.hosts) { _, _ in downloads.reconcile() }
        .onChange(of: destination) { _, new in stored = new.rawValue }
        .onChange(of: columnVisibility) { _, new in storedVisibility = new.stored }
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
        let requested = NativeUAT.destination.value()
        let sheets = [MachinesSettings.addOnLaunch, MachinesSettings.editOnLaunch]
        guard requested == "settings" || sheets.contains(requested ?? "") || SettingsUAT.wantsSettings()
        else { return }
        openSettings()
    }
}

/// `NavigationSplitViewVisibility` is not `RawRepresentable`, so the two
/// states a person can actually leave the window in are mapped by hand.
/// `.automatic` is the system choosing; remembering it would remember nothing.
extension NavigationSplitViewVisibility {
    init(stored: String?) { self = stored == "detailOnly" ? .detailOnly : .all }

    var stored: String { self == .detailOnly ? "detailOnly" : "all" }
}

enum Destination: String, Hashable, CaseIterable, Identifiable {
    case generate, library, queue, models, machines

    var id: Self { self }

    /// Where the window opens: an explicit override, else where you were last.
    static var launch: Destination { launch(defaults: AppStorageSuite.defaults) }

    /// The same answer with both of its inputs handed in, so a test can ask
    /// what `MOLD_NATIVE_DESTINATION=library` opens without launching a
    /// window -- the hook that lands a UAT run on the library now that no
    /// sidebar row of that name exists to click.
    static func launch(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        defaults: UserDefaults
    ) -> Destination {
        if let named = NativeUAT.destination.value(in: environment),
           let forced = Destination(rawValue: named) {
            return forced
        }
        let remembered = defaults.string(forKey: "destination")
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
        case .machines: "Machines"
        }
    }

    var symbol: String {
        switch self {
        case .generate: "wand.and.sparkles"
        case .library: "photo.on.rectangle.angled"
        case .queue: "list.bullet.indent"
        case .models: "cube"
        // The glyph the app already means "machine" by -- the Settings empty
        // state and the library's machine chip both use it.
        case .machines: "server.rack"
        }
    }
}
