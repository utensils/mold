import Foundation
import MoldClient
import Testing
@preconcurrency import UserNotifications

@testable import MoldCompanion

/// Exercise the Objective-C completion-handler entry point used by the system,
/// on its background queue, rather than calling the Swift routing helper.
@MainActor
struct NotificationDelegateTests {
    @Test(arguments: [UNNotificationDefaultActionIdentifier, "view", "favourite",
                      UNNotificationDismissActionIdentifier, "unknown"])
    func systemResponseCompletesOnMainAfterRouting(action: String) async throws {
        let suite = "notification-delegate-\(UUID())"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let notifier = Notifier(defaults: defaults, center: nil)
        let link = DeepLink.print(host: UUID(), filename: "finished.png")
        var favourite: PrintID?
        notifier.favourite = { favourite = $0 }
        let delivery = try delivery(notifier, action: action, link: link.url.absoluteString)
        let completedOnMain = await delivery.receive()
        #expect(completedOnMain, "UIKit resumes notification activation in this callback")
        if action == "favourite", case let .print(host, filename) = link {
            #expect(favourite == PrintID(host: host, filename: filename))
            #expect(notifier.link == nil)
        } else {
            #expect(favourite == nil)
            #expect(notifier.link == (action == "view" || action == UNNotificationDefaultActionIdentifier ? link : nil))
        }
    }

    @Test(arguments: [DeepLink.queue(job: nil).url.absoluteString, "invalid", ""])
    func queueAndMalformedLinksAlwaysCompleteOnMain(link: String) async throws {
        let notifier = Notifier(center: nil)
        let delivery = try delivery(notifier, action: UNNotificationDefaultActionIdentifier, link: link)
        #expect(await delivery.receive())
        #expect(notifier.link == (link.hasPrefix("moldstudio:") ? .queue(job: nil) : nil))
    }

    @Test func foregroundPresentationIsSuppressedAndCompletesOnMain() async throws {
        let notifier = Notifier(center: nil)
        let delivery = try delivery(notifier, action: "view", link: "invalid")
        let result = await delivery.present()
        #expect(result.0)
        #expect(result.1.isEmpty)
    }

    private func delivery(_ notifier: Notifier, action: String, link: String) throws -> NotificationTestDelivery {
        let content = UNMutableNotificationContent()
        content.userInfo = ["link": link]
        let request = UNNotificationRequest(identifier: "finished", content: content, trigger: nil)
        let notification = try #require(UNNotification(coder: NotificationTestCoder([
            "request": request, "date": Date(),
        ])))
        let response = try #require(UNNotificationResponse(coder: NotificationTestCoder([
            "notification": notification, "actionIdentifier": action,
        ])))
        #expect(response.notification.request.identifier == "finished")
        return NotificationTestDelivery(delegate: notifier, response: response)
    }
}

nonisolated private final class NotificationTestCoder: NSCoder {
    let values: [String: Any]
    init(_ values: [String: Any]) { self.values = values }
    override var allowsKeyedCoding: Bool { true }
    override func decodeObject(forKey key: String) -> Any? { values[key] }
}

/// Immutable Foundation payload delivered in the same way as UserNotifications.
nonisolated private struct NotificationTestDelivery: @unchecked Sendable {
    let delegate: any UNUserNotificationCenterDelegate
    let response: UNNotificationResponse

    func receive() async -> Bool {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                delegate.userNotificationCenter?(UNUserNotificationCenter.current(), didReceive: response,
                    withCompletionHandler: { continuation.resume(returning: Thread.isMainThread) })
            }
        }
    }

    func present() async -> (Bool, UNNotificationPresentationOptions) {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().async {
                delegate.userNotificationCenter?(UNUserNotificationCenter.current(), willPresent: response.notification,
                    withCompletionHandler: { continuation.resume(returning: (Thread.isMainThread, $0)) })
            }
        }
    }
}
