import SwiftUI

/// Mold Studio Companion: the native iPhone and iPad app for mold, remote-only,
/// the Mac app's little sibling (apps/ios/docs/DESIGN.md).
@main
struct MoldCompanionApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
        }
        .commands { GoCommands() }
    }
}
