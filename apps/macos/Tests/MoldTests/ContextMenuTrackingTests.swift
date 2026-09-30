import AppKit
import MoldClient
import SwiftUI
import Testing

@testable import Mold

@MainActor
struct ContextMenuTrackingTests {
    @Test func statusChangesPreserveTheOpenMenuAndNextOpeningRefreshes() {
        let notifications = NotificationCenter()
        let tracking = ContextMenuTracking(notifications: notifications)
        let nativeMenu = NSMenu()
        let before = ContextualRowActionMenu(
            actions: [RowAction(kind: "save", title: "Save to This Mac’s Library")],
            perform: { _ in }, tracking: tracking)
        let after = ContextualRowActionMenu(
            actions: [RowAction(kind: "save", title: "Saving Locally…", isDisabled: true)],
            perform: { _ in }, tracking: tracking)

        #expect(before != after)
        notifications.post(name: NSMenu.didBeginTrackingNotification, object: nativeMenu)
        #expect(before == after)
        notifications.post(name: NSMenu.didEndTrackingNotification, object: nativeMenu)
        #expect(before != after)
    }

    @Test func submenusAndRepeatedNotificationsDoNotUnfreezeTheParent() {
        let notifications = NotificationCenter()
        let tracking = ContextMenuTracking(notifications: notifications)
        let parent = NSMenu()
        let submenu = NSMenu()
        notifications.post(name: NSMenu.didBeginTrackingNotification, object: parent)
        notifications.post(name: NSMenu.didBeginTrackingNotification, object: parent)
        notifications.post(name: NSMenu.didBeginTrackingNotification, object: submenu)
        notifications.post(name: NSMenu.didEndTrackingNotification, object: submenu)
        #expect(tracking.isTracking)
        notifications.post(name: NSMenu.didEndTrackingNotification, object: parent)
        #expect(!tracking.isTracking)
    }
}
