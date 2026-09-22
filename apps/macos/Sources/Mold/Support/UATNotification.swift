#if DEBUG
import Foundation
import UserNotifications

/// Two distinct checks: `notify` posts a REAL OS alert for a human/UI click;
/// `notification-response` injects its payload to exercise window lifecycle
/// without depending on notification authorization. Never in Release.
@MainActor
enum UATNotification {
    static func payload(_ argument: String) -> [String: String]? {
        let words = argument.split(separator: " ").map(String.init)
        if words == ["failure"] { return ["kind": "failure"] }
        guard words.count == 3, words[0] == "print", UUID(uuidString: words[1]) != nil else { return nil }
        return ["kind": "print", "host": words[1], "filename": words[2]]
    }

    static func deliver(_ argument: String, to responses: NotificationResponses?) -> String {
        guard let responses, let payload = payload(argument) else { return "invalid response" }
        responses.receive(action: UNNotificationDefaultActionIdentifier, userInfo: payload, completion: {})
        return "injected"
    }

    static func post(_ argument: String) async -> String {
        guard let payload = payload(argument), MoldNotifications.isInsideBundle() else { return "invalid alert" }
        let center = UNUserNotificationCenter.current()
        do {
            guard try await center.requestAuthorization(options: [.alert, .sound]) else { return "not authorized" }
            let content = UNMutableNotificationContent()
            content.title = "Mold notification verification"
            content.body = "Open the existing Mold Studio window."
            content.userInfo = payload
            // Gives the tester time to put another app in front.
            let request = UNNotificationRequest(identifier: "mold-notification-uat", content: content,
                                                trigger: UNTimeIntervalNotificationTrigger(timeInterval: 5, repeats: false))
            try await center.add(request)
            return "posted"
        } catch { return "notification error: \(error.localizedDescription)" }
    }
}
#endif
