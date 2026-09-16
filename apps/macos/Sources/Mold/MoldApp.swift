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
    @State private var thumbnails = ThumbnailCache()
    @State private var models = ModelStore()
    @State private var generate = GenerateController()
    @State private var destination = Destination.launch

    var body: some Scene {
        Window("Mold", id: "main") {
            RootView(destination: $destination)
                .task { ClickModifiers.startObserving() }
                .environment(hosts)
                .environment(library)
                .environment(thumbnails)
                .environment(models)
                .environment(generate)
                // Below this the split view stops being a split view and
                // starts being two cramped columns.
                .frame(minWidth: 880, minHeight: 560)
        }
        .defaultSize(width: 1_280, height: 860)
        .windowToolbarStyle(.unified)
        .commands { MoldCommands(destination: $destination) }

        Settings {
            SettingsView()
                .environment(hosts)
        }
    }
}
