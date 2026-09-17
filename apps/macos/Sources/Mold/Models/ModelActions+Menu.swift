import MoldClient
import SwiftUI

// The pure menu the contextual menu and the Model menu both draw from, and
// the one dispatcher that turns a picked item back into a call on
// `ModelActions` -- split from the actions themselves for size.
extension ModelActions {
    /// What can be done to one model. Both the contextual menu and the Model
    /// menu draw exactly this list -- nothing here is ever disabled, the way
    /// `ModelStateCell` already refuses a disabled control.
    enum Kind: Hashable {
        case install, repair, cancelDownload, load, unload, components, licence, delete
    }

    /// One row of that list, in the app's one menu model -- which is also
    /// what puts Delete last and behind a divider wherever it is drawn, so a
    /// right click cannot land the destructive item under the cursor.
    typealias Item = RowAction<Kind>

    /// The exact items that apply to one model right now -- pure, so the
    /// four load/unload cases and the licence gate are tested without a
    /// view (design S5 tests 3-4).
    static func menu(
        for model: Model, installState: ModelInstallState, isBusy: Bool, isDownloading: Bool, licensed: Bool
    ) -> [Item] {
        guard !isBusy else { return [] }
        var items: [Item] = []
        if isDownloading { items.append(Item(kind: .cancelDownload, title: "Cancel Download")) }
        switch installState {
        case .available:
            if !isDownloading { items.append(Item(kind: .install, title: "Install")) }
            return items
        case .needsRepair:
            if !isDownloading { items.append(Item(kind: .repair, title: "Repair")) }
        case .installed:
            items.append(Item(kind: .load, title: "Load"))
        case .loaded:
            items.append(Item(kind: .unload, title: "Unload"))
        }
        items.append(Item(kind: .components, title: "Components…"))
        if licensed { items.append(Item(kind: .licence, title: "Show Licence…")) }
        items.append(Item(kind: .delete, title: "Delete…", isDestructive: true))
        return items
    }

    /// One dispatcher both the contextual menu and the Model menu call, so
    /// there is exactly one place that turns a picked `Item` into the store
    /// call it means (design S5 test: the menu and the contextual menu call
    /// the same thing).
    func perform(_ kind: Kind, on model: Model, host: MoldHost) {
        switch kind {
        case .install: install(model, on: host)
        case .repair: repair(model, on: host)
        case .cancelDownload: cancelHandler(for: model, on: host)?()
        case .load: load(model, on: host.id)
        case .unload: unload(model, on: host.id)
        case .components: showComponents(model)
        case .licence: showLicence(model, on: host.id)
        case .delete: delete(model, on: host.id)
        }
    }
}
