import Foundation
import UserNotifications

// Handing one notification to the centre, and the single authorization
// request every delivery waits behind. Split from
// `MoldNotifications.swift` past the file-size advisory; neither is
// `private`, because `private` does not cross a file boundary even
// within one type.
@MainActor
extension MoldNotifications {
    /// Asks once, and answers only when the person has ANSWERED.
    ///
    /// `requestAuthorization` is asynchronous, and the old code fired it and
    /// called `add` in the same turn -- so the very first notification of a
    /// session was posted while authorization was still `.notDetermined` and
    /// was dropped. The person saw the permission alert and no notification,
    /// which reads as the toggle not working.
    func authorized() async {
        if let authorization { return await authorization.value }
        let task = Task { [weak self] in
            guard let self else { return }
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                center.requestAuthorization(options: [.alert, .sound]) { _, _ in
                    continuation.resume()
                }
            }
        }
        authorization = task
        await task.value
    }

    /// Chained rather than fired: each delivery waits for the one before it,
    /// so they arrive in the order they were decided AND only the first pays
    /// for authorization.
    func post(title: String, body: String, userInfo: [String: String]) {
        let previous = deliveries
        deliveries = Task { [weak self] in
            await previous?.value
            guard let self else { return }
            await authorized()
            let content = UNMutableNotificationContent()
            content.title = title
            content.body = body
            content.userInfo = userInfo
            center.add(
                UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil),
                withCompletionHandler: nil)
        }
    }
}
