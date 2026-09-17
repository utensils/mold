import MoldClient
import SwiftUI

// Choosing what to render with: loading the fleet's models and adopting one.
// Split from the pane's own shape purely for size.
extension GeneratePane {
    func loadModels() async {
        await models.refresh()
        adoptFirstReadyModel()
    }

    /// Nothing chosen yet: put back what was last being authored, or start on
    /// something the machine can actually run.
    ///
    /// Also on reachability, because which machine `host` resolves to is an
    /// answer this pane no longer probes for itself -- the root does the one
    /// automatic check, and this adopts when it lands.
    ///
    /// A restored draft goes through the SAME adopt path as any other choice,
    /// so the recipe reconciles the draft that was put back rather than a
    /// default one -- and `keepingDraft` is what stops the recipe's own
    /// numbers overwriting the ones somebody chose last session. A model that
    /// is gone from this machine simply does not match, and the first ready
    /// one is adopted instead.
    func adoptFirstReadyModel() {
        guard controller.modelName == nil, let host else { return }
        let ready = models.ready(on: host.id)
        if let restored = drafts.restoredModel,
           let model = ready.first(where: { $0.name == restored }) {
            drafts.adoptedRestoredModel()
            controller.adopt(model: model, on: host.id, keepingDraft: true)
            return
        }
        guard let first = ready.first else { return }
        drafts.adoptedRestoredModel()
        controller.select(model: first, on: host.id)
    }
}
