import AppKit
import Combine
import MoldClient
import SwiftUI

/// Starting tracking never redraws menu contents. Finishing the final tracked
/// menu publishes once so frozen owners apply their latest pending inputs.
@MainActor
final class ContextMenuTracking: NSObject {
    static let shared = ContextMenuTracking()
    private let notifications: NotificationCenter
    private var menus: Set<ObjectIdentifier> = []
    var isTracking: Bool { !menus.isEmpty }
    func shouldPresentSheet(_ requested: Bool) -> Bool { requested && !isTracking }
    private(set) var completionRevision = 0
    let didFinishTracking = PassthroughSubject<Int, Never>()

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
        guard menus.remove(ObjectIdentifier(menu)) != nil, menus.isEmpty else { return }
        completionRevision &+= 1
        didFinishTracking.send(completionRevision)
    }

}

/// AppKit tracks an open contextual menu in its own run-loop mode. SwiftUI
/// otherwise replaces that menu's hierarchy when the source row redraws,
/// losing the highlighted submenu during background save/status updates.
/// Keep the content boundary equal while tracking; outside that session every
/// evaluation refreshes closures as well as titles and enabled states.
struct ContextualRowActionMenu<Kind: Hashable>: View, Equatable {
    let actions: [RowAction<Kind>]
    let perform: (Kind) -> Void
    var tracking: ContextMenuTracking = .shared
    var extra = AnyView(EmptyView())

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        MainActor.assumeIsolated { lhs.tracking === rhs.tracking && lhs.tracking.isTracking }
    }

    var body: some View {
        if RowAction.offersMenu(actions) {
            RowActionMenu(actions: actions, perform: perform, extra: { extra })
        }
    }
}

/// Pending source and action inputs survive an equal owner comparison without
/// notifying SwiftUI while a native menu is open.
@MainActor
final class ContextMenuOwnerInputs<Content: View, Kind: Hashable> {
    var content: Content
    var actions: () -> [RowAction<Kind>]
    var perform: (Kind) -> Void
    var extra: () -> AnyView

    init(content: Content, actions: @escaping () -> [RowAction<Kind>],
         perform: @escaping (Kind) -> Void, extra: @escaping () -> AnyView) {
        self.content = content
        self.actions = actions
        self.perform = perform
        self.extra = extra
    }

    func update(content: Content, actions: @escaping () -> [RowAction<Kind>],
                perform: @escaping (Kind) -> Void, extra: @escaping () -> AnyView) {
        self.content = content
        self.actions = actions
        self.perform = perform
        self.extra = extra
    }
}

/// A transient presentation keeps its last nonempty footprint when completion
/// would remove it under an open menu. Ongoing progress still updates normally.
@MainActor
final class ContextMenuPresentation<Value> {
    private var displayed: Value?

    func resolve(_ current: Value, isTracking: Bool, isEmpty: Bool) -> Value {
        if isTracking, isEmpty, let displayed { return displayed }
        displayed = current
        return current
    }
}

/// The context-menu builder constructs this value without resolving the plan.
/// Native menu rendering asks its body, so grid redraws do not build every
/// tile's submenu rows or its selection's share objects.
struct ContextualRowActionMenuProvider<Kind: Hashable>: View {
    let actions: () -> [RowAction<Kind>]
    let perform: (Kind) -> Void
    let tracking: ContextMenuTracking
    let extra: () -> AnyView

    var body: some View {
        ContextualRowActionMenu(actions: actions(), perform: perform,
                               tracking: tracking, extra: extra())
            .equatable()
    }
}
