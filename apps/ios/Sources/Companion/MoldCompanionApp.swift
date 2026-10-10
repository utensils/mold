import SwiftUI

/// Mold Studio Companion: the native iPhone and iPad app for mold, remote-only,
/// the Mac app's little sibling (apps/ios/docs/DESIGN.md).
@main
struct MoldCompanionApp: App {
    @State private var stores = CompanionStores()
    @AppStorage(Preference.appearance) private var appearance = AppAppearance.system

    var body: some Scene {
        WindowGroup {
            RootView(stores: stores)
                .injecting(stores)
                .supervisesConnections(stores)
                .background(AppearanceWindow(appearance: appearance))
        }
        .commands { GoCommands() }
        .backgroundTask(.appRefresh(CompanionStores.refreshTask)) {
            await stores.backgroundRefresh()
        }
    }
}
