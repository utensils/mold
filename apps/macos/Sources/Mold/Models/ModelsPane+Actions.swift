import MoldClient
import SwiftUI

// `ModelActions`, the row menus that read from it, and the transient removal
// caption -- split from the pane itself for size. `private` does not cross a
// file boundary, so what this reads on the pane is declared plain `@State`
// there.
extension ModelsPane {
    /// The one door for what a row's controls, its contextual menu and the
    /// Model menu all do -- `ModelActions.swift` (M5 S5).
    var actions: ModelActions {
        ModelActions(
            hosts: hosts, models: models, downloads: downloads, licenses: licenses,
            confirmDestruction: { pendingDestruction = $0 },
            presentComponents: { componentsModel = $0 },
            presentLicense: { licenseInfo = $0 },
            onRemoved: { report($0) }
        )
    }

    func install(_ model: Model) {
        guard let host else { return }
        actions.install(model, on: host)
    }

    /// The Cancel action for a row mid-download, or `nil` off it -- forwards
    /// to `ModelActions.cancelHandler`, the one place the job id and its
    /// progress are read from the same dictionary together.
    func cancel(_ model: Model) -> (() -> Void)? {
        guard let host else { return nil }
        return actions.cancelHandler(for: model, on: host)
    }

    /// The current row's offered actions, for the row's own contextual menu
    /// and for the Model menu this pane publishes below.
    func menuItems(for model: Model) -> [ModelActions.Item] {
        guard let host else { return [] }
        return ModelActions.menu(
            for: model, installState: model.installState,
            isBusy: models.isBusy(with: model, on: host.id),
            isDownloading: actions.isDownloading(model, on: host.id),
            licensed: licenses.licence(gating: model.name, on: host.id) != nil)
    }

    /// What the Model menu publishes for the row currently selected --
    /// `nil` off no selection, which is an empty menu (`ModelCommands`).
    var modelSelection: ModelSelection? {
        guard let host, let selectedModel = candidates.first(where: { $0.id == selection }) else { return nil }
        return ModelSelection(
            target: .init(host: host.id, model: selectedModel.id),
            items: menuItems(for: selectedModel),
            perform: { actions.perform($0, on: selectedModel, host: host) }
        )
    }

    /// Shows what a delete just did for ~8 seconds -- there is no persistent
    /// place for a one-off SUCCESS sentence, and nobody reads a footnote
    /// nobody's looking at (design S5).
    func report(_ summary: String) {
        removalSummary = summary
        Task {
            try? await Task.sleep(for: .seconds(8))
            if removalSummary == summary { removalSummary = nil }
        }
    }
}
