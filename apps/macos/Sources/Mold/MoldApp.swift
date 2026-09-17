import AppKit
import MoldClient
import SwiftUI

/// The composition root.
///
/// This is the only file allowed to know which backend the app is built on.
/// `make lint` fails the build if `MoldClient`'s concrete backends are
/// constructed anywhere else -- that single rule is what keeps the UI honest
/// about its dependencies, and it is why swapping in an in-process Rust engine
/// later is a change to this file rather than to the app.
@main
struct MoldApp: App {
    @State private var hosts: HostStore
    @State private var library: LibraryStore
    @State private var libraryNavigation = LibraryNavigation()
    @State private var thumbnails = ThumbnailCache()
    @State private var materializer = PrintMaterializer()
    @State private var models: ModelStore
    @State private var generate: GenerateController
    @State private var queue: QueueStore
    @State private var upscales: UpscaleStore
    @State private var activity: ActivityStore
    @State private var transfers: TransferStore
    @State private var licenses: LicenseStore
    @State private var downloads: DownloadStore
    @State private var catalog: CatalogStore
    @State private var machines: MachineStore
    @State private var pairing: PairingStore
    @State private var promptHistory: PromptHistoryStore
    @State private var modelDefaults: ConfigStore
    @State private var adapters: LoraStore
    @State private var landedPrints: LandedPrints
    @State private var notifications: MoldNotifications
    @State private var heartbeat: HostHeartbeat
    @State private var engine: MoldEngine
    @State private var destination = Destination.launch
    @NSApplicationDelegateAdaptor(MoldAppDelegate.self) private var delegate

    /// `@State` initializers cannot reference each other, so composition
    /// happens explicitly here: `HostStore` first, since every other store
    /// is built by asking it which machines exist.
    init() {
        // FIRST, before a single store exists and so before URLSession or
        // `HostStore`'s polling has a thread of its own (review 05-M4).
        let engine = MoldEngine.bootstrapped()
        _engine = State(initialValue: engine)
        let hosts = HostStore(hosts: HostStore.seededHosts())
        engine.dropsItsMachine(from: hosts)
        _hosts = State(initialValue: hosts)
        let library = LibraryStore(hosts: hosts)
        _library = State(initialValue: library)
        let models = ModelStore(hosts: hosts)
        _models = State(initialValue: models)
        let queue = QueueStore(hosts: hosts)
        _queue = State(initialValue: queue)
        _upscales = State(initialValue: UpscaleStore(
            hosts: hosts, models: models, library: library))
        _activity = State(initialValue: ActivityStore(hosts: hosts))
        _transfers = State(initialValue: TransferStore(hosts: hosts, queue: queue))
        let licenses = LicenseStore(hosts: hosts)
        _licenses = State(initialValue: licenses)
        _downloads = State(initialValue: DownloadStore(hosts: hosts, licenses: licenses))
        _catalog = State(initialValue: CatalogStore(hosts: hosts))
        let modelDefaults = ConfigStore(hosts: hosts)
        _modelDefaults = State(initialValue: modelDefaults)
        _promptHistory = State(initialValue: PromptHistoryStore(hosts: hosts))
        _generate = State(initialValue: GenerateController(hosts: hosts, defaults: modelDefaults))
        _machines = State(initialValue: MachineStore(hosts: hosts))
        _pairing = State(initialValue: PairingStore(hosts: hosts))
        _adapters = State(initialValue: LoraStore(hosts: hosts))
        let landedPrints = LandedPrints(hosts: hosts)
        _landedPrints = State(initialValue: landedPrints)
        _notifications = State(initialValue: MoldNotifications(
            landedPrints: landedPrints, queue: queue, hosts: hosts, library: library))
        _heartbeat = State(initialValue: HostHeartbeat(hosts: hosts, queue: queue))
    }

    var body: some Scene {
        Window("Mold", id: "main") {
            RootView(destination: $destination)
                .task { ClickModifiers.startObserving() }
                // Whether a caret owns the keyboard, asked once for the whole
                // app -- what stands the Library's bare-space shortcut down.
                .task { TextEditingFocus.shared.startObserving() }
                // The delegate owns quitting, and quitting has to reach the
                // engine and the cache. It is made by SwiftUI, so this is
                // where the two meet.
                .task {
                    delegate.engine = engine
                    delegate.materializer = materializer
                    delegate.thumbnails = thumbnails
                    delegate.landedPrints = landedPrints
                    delegate.dockBadge.follow(landedPrints)
                    // `applicationDidBecomeActive` has already fired by the
                    // time this scene's task runs, so the launch start is
                    // here rather than there; `start()` is idempotent, so the
                    // next activation costs nothing.
                    delegate.heartbeat = heartbeat
                    if NSApp.isActive { heartbeat.start() }
                    // The same signal: `ActivityStore` watches for the two
                    // activation notifications itself, and this is the launch
                    // start that has already fired by the time this runs.
                    if NSApp.isActive { activity.start() }
                    // A notification click reaches the delegate, not a view
                    // -- this is where it meets the destination binding and
                    // the Library's own navigation.
                    delegate.onNotificationRoute = { route in
                        applyNotificationRoute(route, destination: $destination, navigation: libraryNavigation)
                    }
                }
                .environment(hosts)
                .environment(library)
                .environment(libraryNavigation)
                .environment(thumbnails)
                .environment(materializer)
                .environment(models)
                .environment(generate)
                .environment(queue)
                .environment(upscales)
                .environment(activity)
                .environment(transfers)
                .environment(licenses)
                .environment(downloads)
                .environment(catalog)
                .environment(machines)
                .environment(pairing)
                .environment(promptHistory)
                .environment(modelDefaults)
                .environment(adapters)
                .environment(landedPrints)
                .environment(engine)
                // Below this the split view stops being a split view and
                // starts being two cramped columns.
                .frame(minWidth: 880, minHeight: 560)
        }
        .defaultSize(width: 1_280, height: 860)
        .windowToolbarStyle(.unified)
        .commands {
            // SwiftUI's own View ▸ Hide/Show Sidebar (⌃⌘S). Nothing bespoke:
            // the shortcut every Mac app uses belongs to the framework.
            SidebarCommands()
            MoldCommands(destination: $destination)
            LibraryCommands()
            ModelCommands()
            QueueCommands()
            MachineCommands()
        }

        Settings {
            SettingsView()
                .environment(hosts)
                .environment(engine)
                .environment(materializer)
                // Settings ▸ Empty Now empties BOTH caches.
                .environment(thumbnails)
                .environment(catalog)
                .environment(modelDefaults)
                .environment(library)
        }
    }
}
