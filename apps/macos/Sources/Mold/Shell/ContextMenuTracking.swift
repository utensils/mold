import AppKit

/// Notification tracking is intentionally not observable: starting or ending
/// a native menu session must not itself invalidate SwiftUI menu contents.
@MainActor
final class ContextMenuTracking: NSObject {
    static let shared = ContextMenuTracking()
    private let notifications: NotificationCenter
    private var menus: Set<ObjectIdentifier> = []
    var isTracking: Bool { !menus.isEmpty }

    init(notifications: NotificationCenter = .default) {
        self.notifications = notifications
        super.init()
        notifications.addObserver(self, selector: #selector(begin(_:)),
                                  name: NSMenu.didBeginTrackingNotification, object: nil)
        notifications.addObserver(self, selector: #selector(end(_:)),
                                  name: NSMenu.didEndTrackingNotification, object: nil)
    }

    @objc private func begin(_ notification: Notification) {
        guard let menu = notification.object as? NSMenu else { return }
        menus.insert(ObjectIdentifier(menu))
    }

    @objc private func end(_ notification: Notification) {
        guard let menu = notification.object as? NSMenu else { return }
        menus.remove(ObjectIdentifier(menu))
    }
}
