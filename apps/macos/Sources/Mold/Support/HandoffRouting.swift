import MoldClient
import SwiftUI

/// Handoff from Mold Studio Companion on iPhone/iPad: the print it was
/// showing opens here, found by the server's run or its address
/// (`PrintHandoff`), the way a notification click opens one.
struct HandoffRouting: ViewModifier {
    @Binding var destination: Destination
    let hosts: HostStore
    let navigation: LibraryNavigation

    func body(content: Content) -> some View {
        content.onContinueUserActivity(PrintHandoff.activityType) { activity in
            guard let id = PrintHandoff.resolve(activity.userInfo ?? [:], hosts: hosts.hosts,
                                                instanceOf: { hosts.instanceIDs[$0] })
            else { return }
            applyNotificationRoute(.openPrint(id), destination: $destination, navigation: navigation)
        }
    }
}
