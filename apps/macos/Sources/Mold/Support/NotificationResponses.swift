import UserNotifications

/// Notification Center can deliver a launch response before SwiftUI mounts
/// the main scene. Accept it now and hand it to navigation exactly once.
@MainActor
final class NotificationResponses {
    private var pending: [NotificationRoute] = []
    private var handler: ((NotificationRoute) -> Void)?

    func install(_ handler: @escaping (NotificationRoute) -> Void) {
        self.handler = handler
        let waiting = pending
        pending.removeAll()
        for route in waiting { handler(route) }
    }

    func receive(action: String, userInfo: [String: String], completion: () -> Void) {
        defer { completion() }
        // Mold registers no custom actions. Dismissing an alert must not
        // unexpectedly open a print or bring the app to the foreground.
        guard action == UNNotificationDefaultActionIdentifier,
              let route = NotificationRoute.route(userInfo: userInfo)
        else { return }
        if let handler { handler(route) } else { pending.append(route) }
    }
}
