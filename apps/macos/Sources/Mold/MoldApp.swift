import AppKit
import SwiftUI

/// The app: its scenes and its menu bar.
///
/// The store graph it runs on is `AppStores`, composed once and handed to the
/// window through `moldEnvironment`. That split is deliberate -- composition
/// is ORDERED and a page long, and mixing it into the scene body is what took
/// this file past the 150-line rule every other file in the app keeps to.
@main
struct MoldApp: App {
    @State private var stores = AppStores()
    @State private var destination = Destination.launch
    @NSApplicationDelegateAdaptor(MoldAppDelegate.self) private var delegate

    var body: some Scene {
        Window("Mold", id: "main") {
            RootView(destination: $destination)
                .task { ClickModifiers.startObserving() }
                // Whether a caret owns the keyboard, asked once for the whole
                // app -- what stands the Library's bare-space shortcut down.
                .task { TextEditingFocus.shared.startObserving() }
                .task { handOverToTheDelegate() }
                .moldEnvironment(stores)
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
            UpdateCommands()
        }

        Settings {
            SettingsView()
                .moldSettingsEnvironment(stores)
        }
    }

    /// The delegate owns quitting, and quitting has to reach the engine and
    /// the cache. It is made by SwiftUI, so this is where the two meet.
    private func handOverToTheDelegate() {
        delegate.engine = stores.engine
        delegate.materializer = stores.materializer
        delegate.thumbnails = stores.thumbnails
        delegate.landedPrints = stores.landedPrints
        delegate.dockBadge.follow(stores.landedPrints)
        // `applicationDidBecomeActive` has already fired by the time this
        // scene's task runs, so the launch start is here rather than there;
        // `start()` is idempotent, so the next activation costs nothing.
        delegate.heartbeat = stores.heartbeat
        // THE LAUNCH START. A store that watches the activation notifications
        // itself still needs its first one from here, because
        // `applicationDidBecomeActive` has already fired. A NEW STORE THAT
        // STARTS AT LAUNCH GOES ON THIS LINE -- it is the one hunk of the old
        // composition root that did not move into `AppStores`, and the one a
        // lane re-applying its work would otherwise miss (review F5#4).
        if NSApp.isActive { stores.heartbeat.start(); stores.activity.start() }
        // A notification click reaches the delegate, not a view -- this is
        // where it meets the destination binding and the Library's own
        // navigation.
        let navigation = stores.libraryNavigation
        let destination = $destination
        delegate.onNotificationRoute = { route in
            applyNotificationRoute(route, destination: destination, navigation: navigation)
        }
    }
}
