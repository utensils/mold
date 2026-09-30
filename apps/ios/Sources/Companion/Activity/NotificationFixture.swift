#if DEBUG
import Foundation
import UserNotifications

/// A real system notification for Simulator UAT, without submitting a render.
/// Never included in Release / TestFlight builds.
enum NotificationFixture {
    static func schedule(center: UNUserNotificationCenter?) {
        let arguments = ProcessInfo.processInfo.arguments
        guard let index = arguments.firstIndex(of: "--notification-fixture-link"),
              arguments.indices.contains(index + 1),
              let url = URL(string: arguments[index + 1]), DeepLink(url) != nil else { return }
        Task {
            guard let center, (try? await center.requestAuthorization(options: [.alert, .sound])) == true else { return }
            let content = UNMutableNotificationContent()
            content.title = "Render complete"
            content.body = "Notification tap regression fixture"
            content.categoryIdentifier = Notifier.finishedCategory
            content.userInfo = ["link": url.absoluteString]
            try? await center.add(UNNotificationRequest(identifier: "notification-tap-fixture", content: content,
                trigger: UNTimeIntervalNotificationTrigger(timeInterval: 10, repeats: false)))
        }
    }
}
#endif
