import MoldClient
import SwiftUI

// Choosing what to render with: loading the fleet's models and adopting one.
// Split from the pane's own shape purely for size.
extension GeneratePane {
    func loadModels() async {
        await models.refresh()
        adoptFirstReadyModel()
    }

    /// Nothing chosen yet: start on something the machine can actually run.
    ///
    /// Also on reachability, because which machine `host` resolves to is an
    /// answer this pane no longer probes for itself -- the root does the one
    /// automatic check, and this adopts when it lands.
    func adoptFirstReadyModel() {
        guard controller.modelName == nil, let host,
              let first = models.ready(on: host.id).first
        else { return }
        controller.select(model: first, on: host.id)
    }
}
