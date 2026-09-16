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
    @State private var hosts = HostStore(hosts: HostStore.seededHosts())
    @State private var library = LibraryStore()
    @State private var libraryNavigation = LibraryNavigation()
    @State private var thumbnails = ThumbnailCache()
    @State private var materializer = PrintMaterializer()
    @State private var models = ModelStore()
    @State private var generate = GenerateController()
    @State private var queue = QueueStore()
    @State private var downloads = DownloadStore()
    @State private var engine = MoldEngine()
    @State private var destination = Destination.launch
    @NSApplicationDelegateAdaptor(MoldAppDelegate.self) private var delegate

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
                .environment(engine)
                // Below this the split view stops being a split view and
                // starts being two cramped columns.
                .frame(minWidth: 880, minHeight: 560)
        }
        .defaultSize(width: 1_280, height: 860)
        .windowToolbarStyle(.unified)
        .commands {
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
