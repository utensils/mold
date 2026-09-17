import MoldClient
import SwiftUI

// The pure menu the contextual menu and the Model menu both draw from, and
// the one dispatcher that turns a picked item back into a call on
// `ModelActions` -- split from the actions themselves for size.
extension ModelActions {
    /// One offered action, in menu order. Both the contextual menu and the
    /// Model menu draw exactly this list -- nothing here is ever disabled,
    /// the way `ModelStateCell` already refuses a disabled control.
    struct Item: Identifiable, Equatable {
        enum Kind: Equatable {
            case install, repair, cancelDownload, load, unload, components, licence, delete
        }

        let kind: Kind
        let title: String
        let systemImage: String

        var id: Kind { kind }
        var role: ButtonRole? { kind == .delete ? .destructive : nil }
    }

    /// The exact items that apply to one model right now -- pure, so the
    /// four load/unload cases and the licence gate are tested without a
    /// view (design S5 tests 3-4).
    static func menu(
        for model: Model, installState: ModelInstallState, isBusy: Bool, isDownloading: Bool, licensed: Bool
    ) -> [Item] {
        guard !isBusy else { return [] }
        var items: [Item] = []
        if isDownloading {
            items.append(Item(kind: .cancelDownload, title: "Cancel Download", systemImage: "xmark.circle"))
        }
        switch installState {
        case .available:
            if !isDownloading {
                items.append(Item(kind: .install, title: "Install", systemImage: "arrow.down.circle"))
            }
            return items
        case .needsRepair:
            if !isDownloading {
                items.append(Item(kind: .repair, title: "Repair", systemImage: "wrench.and.screwdriver"))
            }
        case .installed:
            items.append(Item(kind: .load, title: "Load", systemImage: "bolt"))
        case .loaded:
            items.append(Item(kind: .unload, title: "Unload", systemImage: "bolt.slash"))
        }
        items.append(Item(kind: .components, title: "Components…", systemImage: "square.stack.3d.up"))
        if licensed {
            items.append(Item(kind: .licence, title: "Show Licence…", systemImage: "doc.text"))
        }
        items.append(Item(kind: .delete, title: "Delete…", systemImage: "trash"))
        return items
    }

    /// One dispatcher both the contextual menu and the Model menu call, so
    /// there is exactly one place that turns a picked `Item` into the store
    /// call it means (design S5 test: the menu and the contextual menu call
    /// the same thing).
    func perform(_ kind: Item.Kind, on model: Model, host: MoldHost) {
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
