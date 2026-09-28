import MoldClient
import SwiftUI

/// Everything that arrives from outside the window and says where to go:
/// `moldstudio://` links (widgets, the Live Activity), a tapped notification,
/// and Handoff from Mold Studio on the Mac. A print opens full screen over
/// whatever is up.
struct RootLinks: ViewModifier {
    @Environment(Notifier.self) private var notifier
    @Environment(HostStore.self) private var hosts
    @Bindable var router: AppRouter

    func body(content: Content) -> some View {
        content
            .fullScreenCover(item: $router.openedPrint) { opened in LinkedPrint(id: opened.id) }
            .onOpenURL { url in DeepLink(url).map(router.open) }
            .onContinueUserActivity(PrintHandoff.activityType) { activity in
                if let id = PrintHandoff.resolve(activity.userInfo ?? [:], hosts: hosts.hosts,
                                                 instanceOf: hosts.instanceID(of:)) {
                    router.openedPrint = AppRouter.OpenedPrint(id: id)
                }
            }
            .onChange(of: notifier.link) { _, link in
                guard let link else { return }
                router.open(link)
                notifier.link = nil
            }
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
