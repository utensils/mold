import AppKit
import SwiftUI

/// The imperative notification callback meets SwiftUI's singleton Window.
/// Keep the handler installed after the content disappears: a click must be
/// able to recreate a closed window, not just change invisible navigation.
struct NotificationWindowRouting: ViewModifier {
    @Environment(\.openWindow) private var openWindow
    let responses: NotificationResponses
    @Binding var destination: Destination
    let navigation: LibraryNavigation

    func body(content: Content) -> some View {
        content.onAppear {
            responses.install { [destination = $destination, navigation, openWindow] route in
                applyNotificationRoute(route, destination: destination, navigation: navigation)
                openWindow(id: "main")
                NSApp.activate(ignoringOtherApps: true)
            }
        }
    }
}
