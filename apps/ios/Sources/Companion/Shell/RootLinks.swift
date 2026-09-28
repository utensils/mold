import MoldClient
import SwiftUI

/// Everything that arrives from outside the window and says where to go:
/// `moldstudio://` links (widgets, the Live Activity), a tapped notification,
/// and Handoff from Mold Studio on the Mac. A print opens full screen over
/// whatever is up.
struct RootLinks: ViewModifier {
    @Environment(Notifier.self) private var notifier
    @Environment(HostStore.self) private var hosts
    @Environment(\.scenePhase) private var phase
    @Bindable var router: AppRouter

    func body(content: Content) -> some View {
        content
            .fullScreenCover(item: $router.openedPrint) { opened in LinkedPrint(id: opened.id) }
            .onOpenURL { url in DeepLink(url).map(router.open) }
            .onContinueUserActivity(PrintHandoff.activityType) { activity in
                let info = activity.userInfo ?? [:]
                Task {
                    // From a cold launch no machine has answered yet, and the
                    // match is by the server's run: ask first.
                    if hosts.upHosts.isEmpty { await hosts.refreshAll() }
                    if let id = PrintHandoff.resolve(info, hosts: hosts.hosts, instanceOf: hosts.instanceID(of:)) {
                        router.openedPrint = AppRouter.OpenedPrint(id: id)
                    }
                }
            }
            // One window takes it: the first active one to see it clears it,
            // and every other window then finds nothing. A tap that launched
            // the app waits for the window to become active.
            .onChange(of: notifier.link) { takeLink() }
            .onChange(of: phase) { takeLink() }
    }

    private func takeLink() {
        guard phase == .active, let link = notifier.link else { return }
        notifier.link = nil
        router.open(link)
    }
}

/// Hands the window's `UndoManager` to the Library, so ⌘Z on iPad and
/// shake-to-undo on iPhone put back the last change to prints.
struct UndoBridge: ViewModifier {
    @Environment(\.undoManager) private var undoManager
    @Environment(LibraryStore.self) private var library

    func body(content: Content) -> some View {
        content
            .onAppear { library.undoManager = undoManager }
            .onChange(of: undoManager) { _, new in library.undoManager = new }
    }
}
