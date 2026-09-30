import AppKit
import MoldClient
import Observation
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

    @Test func statusChangesPreserveTheMenuOwnerUntilDismissal() {
        let notifications = NotificationCenter()
        let tracking = ContextMenuTracking(notifications: notifications)
        let nativeMenu = NSMenu()
        let before = ContextualRowActionOwner(
            content: Text("Print"),
            actions: { [RowAction(kind: "save", title: "Save Locally")] },
            perform: { _ in }, tracking: tracking)
        let after = ContextualRowActionOwner(
            content: Text("Updated Print"),
            actions: { [RowAction(kind: "save", title: "Saving…", isDisabled: true)] },
            perform: { _ in }, tracking: tracking)
        #expect(before != after)
        notifications.post(name: NSMenu.didBeginTrackingNotification, object: nativeMenu)
        #expect(before == after)
        notifications.post(name: NSMenu.didEndTrackingNotification, object: nativeMenu)
        #expect(before != after)
    }

    @Test func ownerInputsAreFreshNormallyAndForwardedWhileTracking() {
        let notifications = NotificationCenter()
        let tracking = ContextMenuTracking(notifications: notifications)
        var performed: [String] = []
        let before = ContextualRowActionOwner(
            content: Text("First"),
            actions: { [RowAction(kind: "first", title: "First")] },
            perform: { performed.append($0) }, tracking: tracking)
        let after = ContextualRowActionOwner(
            content: Text("Latest"),
            actions: { [RowAction(kind: "latest", title: "Latest")] },
            perform: { performed.append("updated:\($0)") }, tracking: tracking)
        #expect(before.latest !== after.latest)
        #expect(before.latest.actions().first?.kind == "first")
        #expect(after.latest.actions().first?.kind == "latest")
        let menu = NSMenu()
        notifications.post(name: NSMenu.didBeginTrackingNotification, object: menu)
        #expect(before == after)
        #expect(before.latest.actions().first?.kind == "latest")
        before.latest.perform("latest")
        #expect(performed == ["updated:latest"])
        notifications.post(name: NSMenu.didEndTrackingNotification, object: menu)
        #expect(before != after)
        #expect(after.latest.actions().first?.kind == "latest")
    }

    @Test func purePlanRemainsLazyAndNeverObservesNativeBuilderDependencies() {
        let probe = MenuObservationProbe()
        let plan = LibraryMenuPlan(scope: .prints, count: 1, name: probe.title)
        let owner = ContextualRowActionOwner(
            content: Text("Print"), actions: {
                probe.reads += 1
                return plan.items
            }, perform: { _ in })
        withObservationTracking {
            let _ = owner.body
        } onChange: {
            MainActor.assumeIsolated { probe.invalidated = true }
        }
        #expect(probe.reads == 0)
        withObservationTracking {
            let _ = owner.actions()
        } onChange: {
            MainActor.assumeIsolated { probe.invalidated = true }
        }
        #expect(probe.reads == 1)
        probe.progress = "Copying 2…"
        probe.title = "Updated selection"
        #expect(!probe.invalidated)
    }

    @Test func completionReportWaitsForMenuDismissalWithoutClearingTheRequest() {
        let notifications = NotificationCenter()
        let tracking = ContextMenuTracking(notifications: notifications)
        let menu = NSMenu()
        #expect(tracking.shouldPresentSheet(true))
        notifications.post(name: NSMenu.didBeginTrackingNotification, object: menu)
        #expect(!tracking.shouldPresentSheet(true))
        #expect(!tracking.shouldPresentSheet(false))
        notifications.post(name: NSMenu.didEndTrackingNotification, object: menu)
        #expect(tracking.shouldPresentSheet(true))
        #expect(!tracking.shouldPresentSheet(false))
    }

    @Test func completedActivityKeepsItsFooterUntilMenuDismissal() {
        let presentation = ContextMenuPresentation<[String]>()
        #expect(presentation.resolve(["Copying 2 of 8…"], isTracking: false, isEmpty: false)
                == ["Copying 2 of 8…"])
        #expect(presentation.resolve(["Copying 5 of 8…"], isTracking: true, isEmpty: false)
                == ["Copying 5 of 8…"])
        #expect(presentation.resolve([], isTracking: true, isEmpty: true) == ["Copying 5 of 8…"])
        #expect(presentation.resolve([], isTracking: false, isEmpty: true).isEmpty)
        #expect(presentation.resolve([], isTracking: true, isEmpty: true).isEmpty)
    }

    @Test func finishingOnlyTheFinalMenuRefreshesFrozenOwnersOnce() {
        let notifications = NotificationCenter()
        let tracking = ContextMenuTracking(notifications: notifications)
        var refreshed: [Int] = []
        let subscription = tracking.didFinishTracking.sink { refreshed.append($0) }
        defer { subscription.cancel() }
        let parent = NSMenu()
        let submenu = NSMenu()
        notifications.post(name: NSMenu.didBeginTrackingNotification, object: parent)
        notifications.post(name: NSMenu.didBeginTrackingNotification, object: submenu)
        #expect(refreshed.isEmpty)
        notifications.post(name: NSMenu.didEndTrackingNotification, object: submenu)
        #expect(refreshed.isEmpty)
        notifications.post(name: NSMenu.didEndTrackingNotification, object: parent)
        #expect(refreshed == [1])
        notifications.post(name: NSMenu.didEndTrackingNotification, object: parent)
        #expect(refreshed == [1])
    }

    @Test func pendingOwnerInputsKeepTheCurrentMenuSnapshotAndRefreshNextActions() {
        var performed: [String] = []
        let initial = ContextMenuOwnerInputs(
            content: Text("First selection"),
            actions: { [RowAction(kind: "first", title: "First action")] },
            perform: { performed.append($0) }, extra: { AnyView(EmptyView()) })
        let openedActions = initial.actions
        initial.update(content: Text("Latest selection"),
                       actions: { [RowAction(kind: "latest", title: "Latest action")] },
                       perform: { performed.append("updated:\($0)") },
                       extra: { AnyView(EmptyView()) })
        #expect(openedActions().first?.kind == "first")
        #expect(initial.actions().first?.kind == "latest")
        initial.perform("latest")
        #expect(performed == ["updated:latest"])
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

@MainActor
@Observable
private final class MenuObservationProbe {
    var title = "Save Locally"
    var progress = "Copying 1…"
    @ObservationIgnored var reads = 0
    @ObservationIgnored var invalidated = false
}
