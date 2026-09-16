import os
import QuickLookUI

/// Quick Look over the selection.
///
/// The real `QLPreviewPanel`, not a viewer of our own: it brings zoom, full
/// screen, rotation, the share menu, "Open with", and arrowing between several
/// items, all of which people already know and none of which is worth
/// rebuilding. Space is its key everywhere else on the Mac, so it is its key
/// here too.
///
/// **Deliberately not `@MainActor`.** QuickLookUI reads an item's URL from its
/// own queues -- `-[QLPreviewView shouldUseAsyncLoading]` does it from a file
/// coordinator's block -- and this target defaults every type to the main
/// actor, so the first preview trapped in `_swift_task_checkIsolatedSwift`
/// before anything was drawn. The state is a list that is replaced whole, so a
/// lock is the honest answer rather than an isolation it cannot honour.
final class QuickLook: NSObject, @unchecked Sendable {
    static let shared = QuickLook()

    private let items = OSAllocatedUnfairLock<[QuickLookItem]>(initialState: [])

    /// Shows the panel over the given files, titled by print.
    @MainActor
    func show(_ files: [(url: URL, title: String)]) {
        guard !files.isEmpty, let panel = QLPreviewPanel.shared() else { return }
        items.withLock { $0 = files.map { QuickLookItem(url: $0.url, title: $0.title) } }
        panel.dataSource = self
        panel.reloadData()
        // Space toggles, the way it does in the Finder.
        if panel.isVisible { panel.orderOut(nil) } else { panel.makeKeyAndOrderFront(nil) }
    }
}

extension QuickLook: QLPreviewPanelDataSource {
    func numberOfPreviewItems(in panel: QLPreviewPanel!) -> Int {
        items.withLock(\.count)
    }

    func previewPanel(_ panel: QLPreviewPanel!, previewItemAt index: Int) -> (any QLPreviewItem)! {
        // `QuickLookItem` is Sendable; `any QLPreviewItem` is not, so the
        // widening happens after the lock rather than inside it.
        let item: QuickLookItem? = items.withLock { index < $0.count ? $0[index] : nil }
        return item
    }
}

/// One print, as Quick Look wants to see it.
///
/// Immutable and nonisolated, because its getters are called from whichever
/// queue QuickLookUI happens to be on. The title is the print's own name
/// rather than the cache's path, which is why the materializer keeps the real
/// filename inside a keyed directory instead of hashing it into the file.
nonisolated final class QuickLookItem: NSObject, QLPreviewItem, @unchecked Sendable {
    let previewItemURL: URL?
    let previewItemTitle: String?

    init(url: URL, title: String) {
        previewItemURL = url
        previewItemTitle = title
    }
}
