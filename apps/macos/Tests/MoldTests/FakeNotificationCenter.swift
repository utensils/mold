import Foundation
import UserNotifications

@testable import Mold

/// A center that records what it was asked instead of ever touching
/// `UNUserNotificationCenter` -- posting a REAL notification from a test run
/// is exactly what the bundle guard exists to prevent (`MoldNotifications
/// .swift`), so nothing here is allowed to reach the real one.
///
/// It also models the one thing about the real centre that MATTERS to the
/// order of these calls: a request added while authorization is still
/// `.notDetermined` is dropped, not queued. Answering immediately (the
/// default) hides that entirely, which is how the defect survived.
@MainActor
final class FakeNotificationCenter: NotificationCenterProtocol {
    struct Posted: Equatable {
        let title: String
        let body: String
        let userInfo: [String: String]
    }

    private(set) var authorizationRequests = 0
    private(set) var posted: [Posted] = []
    /// Requests that arrived before the person had answered. The real centre
    /// simply loses these.
    private(set) var dropped = 0

    /// Holds the answer the way a real permission alert does, until
    /// `answerAuthorization(_:)`. Off by default, so every existing test is
    /// unaffected.
    var defersAuthorization = false
    private var isAuthorized = false
    private var waiting: [@Sendable (Bool, (any Error)?) -> Void] = []

    func requestAuthorization(
        options: UNAuthorizationOptions, completionHandler: @escaping @Sendable (Bool, (any Error)?) -> Void
    ) {
        authorizationRequests += 1
        guard defersAuthorization else {
            isAuthorized = true
            completionHandler(true, nil)
            return
        }
        waiting.append(completionHandler)
    }

    /// The person answers the alert.
    func answerAuthorization(_ granted: Bool = true) {
        isAuthorized = granted
        let pending = waiting
        waiting = []
        for completionHandler in pending { completionHandler(granted, nil) }
    }

    func add(
        _ request: UNNotificationRequest,
        withCompletionHandler completionHandler: (@Sendable ((any Error)?) -> Void)?
    ) {
        guard isAuthorized else {
            dropped += 1
            completionHandler?(nil)
            return
        }
        let content = request.content
        let userInfo = (content.userInfo as? [String: String]) ?? [:]
        posted.append(Posted(title: content.title, body: content.body, userInfo: userInfo))
        completionHandler?(nil)
    }
}
