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
    @State private var downloads: DownloadStore
    @State private var machines: MachineStore
    @State private var engine = MoldEngine()
    @State private var destination = Destination.launch
    @NSApplicationDelegateAdaptor(MoldAppDelegate.self) private var delegate

    /// `@State` initializers cannot reference each other, so composition
    /// happens explicitly here: `HostStore` first, since every other store
    /// is built by asking it which machines exist.
    init() {
        let hosts = HostStore(hosts: HostStore.seededHosts())
        _hosts = State(initialValue: hosts)
        _library = State(initialValue: LibraryStore(hosts: hosts))
        _models = State(initialValue: ModelStore(hosts: hosts))
        _queue = State(initialValue: QueueStore(hosts: hosts))
        _downloads = State(initialValue: DownloadStore(hosts: hosts))
        _generate = State(initialValue: GenerateController(hosts: hosts))
        _machines = State(initialValue: MachineStore(hosts: hosts))
    }

    var body: some Scene {
        Window("Mold", id: "main") {
            RootView(destination: $destination)
                .task { ClickModifiers.startObserving() }
                // The delegate owns quitting, and quitting has to reach the
                // engine and the cache. It is made by SwiftUI, so this is
                // where the two meet.
                .task {
                    delegate.engine = engine
                    delegate.materializer = materializer
                }
                .environment(hosts)
                .environment(library)
                .environment(libraryNavigation)
                .environment(thumbnails)
                .environment(materializer)
                .environment(models)
                .environment(generate)
                .environment(queue)
                .environment(downloads)
                .environment(machines)
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
        }

        Settings {
            SettingsView()
                .environment(hosts)
                .environment(engine)
                .environment(materializer)
        }
    }
}
