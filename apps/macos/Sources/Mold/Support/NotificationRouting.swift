import Foundation
import MoldClient
import SwiftUI

/// What a notification click should do, decided purely from the `userInfo`
/// `MoldNotifications.post` attached -- no AppKit or `UserNotifications`
/// type appears here, so a test drives it with a plain dictionary rather
/// than a real `UNNotificationResponse`.
nonisolated enum NotificationRoute: Equatable {
    case openPrint(PrintID)
    case openQueue

    /// `nil` for a payload naming neither a print nor a failure -- an older
    /// or unrecognised notification, which opens nothing rather than
    /// guessing. `nonisolated`: the project defaults to `@MainActor`, but
    /// this runs from `MoldAppDelegate`'s delegate callback, which AppKit
    /// does not promise the main thread for.
    static func route(userInfo: [String: String]) -> NotificationRoute? {
        switch userInfo["kind"] {
        case "print":
            guard let filename = userInfo["filename"], let hostString = userInfo["host"],
                  let host = UUID(uuidString: hostString)
            else { return nil }
            return .openPrint(PrintID(host: host, filename: filename))
        case "failure":
            return .openQueue
        default:
            return nil
        }
    }
}

/// Applies a route to the shell -- the AppKit-touching half `NotificationRoute
/// .route` deliberately stays free of. A print opens on the shelf that can
/// show it, never the collection you were last looking at: All Prints, or for
/// a member of a HIDDEN collection (which All Prints leaves out) that
/// collection. `LibraryPane.revealIfNeeded` decides and sequences it.
@MainActor
func applyNotificationRoute(
    _ route: NotificationRoute, destination: Binding<Destination>, navigation: LibraryNavigation
) {
    switch route {
    case let .openPrint(id):
        // No scope here: the Library's reveal picks the shelf that shows the
        // print (All Prints, or its hidden collection) and changes it itself,
        // so one observer sequences the change and the selection.
        navigation.reveal = id
        destination.wrappedValue = .library
    case .openQueue:
        destination.wrappedValue = .queue
    }
}
