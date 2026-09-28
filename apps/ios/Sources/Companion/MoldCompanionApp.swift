import SwiftUI

/// Mold Studio Companion: the native iPhone and iPad app for mold, remote-only,
/// the Mac app's little sibling (apps/ios/docs/DESIGN.md).
@main
struct MoldCompanionApp: App {
    @State private var stores = CompanionStores()

    var body: some Scene {
        WindowGroup {
            RootView(stores: stores)
                .injecting(stores)
                .supervisesConnections(stores)
        }
        .commands { GoCommands() }
        .backgroundTask(.appRefresh(CompanionStores.refreshTask)) {
            await stores.backgroundRefresh()
        }
    }
}
