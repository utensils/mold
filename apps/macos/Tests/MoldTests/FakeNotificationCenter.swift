import Foundation
import UserNotifications

@testable import Mold

/// A center that records what it was asked instead of ever touching
/// `UNUserNotificationCenter` -- posting a REAL notification from a test run
/// is exactly what the bundle guard exists to prevent (`MoldNotifications
/// .swift`), so nothing here is allowed to reach the real one.
@MainActor
final class FakeNotificationCenter: NotificationCenterProtocol {
    struct Posted: Equatable {
        let title: String
        let body: String
        let userInfo: [String: String]
    }

    private(set) var authorizationRequests = 0
    private(set) var posted: [Posted] = []

    func requestAuthorization(
        options: UNAuthorizationOptions, completionHandler: @escaping @Sendable (Bool, (any Error)?) -> Void
    ) {
        authorizationRequests += 1
        completionHandler(true, nil)
    }

    func add(
        _ request: UNNotificationRequest,
        withCompletionHandler completionHandler: (@Sendable ((any Error)?) -> Void)?
    ) {
        let content = request.content
        let userInfo = (content.userInfo as? [String: String]) ?? [:]
        posted.append(Posted(title: content.title, body: content.body, userInfo: userInfo))
        completionHandler?(nil)
    }
}
