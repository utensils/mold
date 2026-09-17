import MoldClient
import SwiftUI

// What this machine says it will chain, for the model that is chosen. Split
// from the pane's own shape purely for size.
extension GeneratePane {
    /// Everything the answer depends on. A `.task(id:)` over this asks once
    /// per machine, model and rate, and never otherwise.
    struct ChainLimitsKey: Equatable {
        let host: MoldHost.ID?
        let model: String?
        let fps: Int?
    }

    var chainLimitsKey: ChainLimitsKey {
        ChainLimitsKey(host: host?.id,
                       model: recipe?.temporal == nil ? nil : selectedModel?.name,
                       fps: controller.draft.fps)
    }

    /// This machine's answer for the chosen model. `nil` while it is in the
    /// air, and on a host too OLD to publish the route -- both of which mean
    /// this app's ported constants stand, which is what absence is for.
    var advertisedChainLimits: ChainLimits? {
        guard let host, let model = chainLimitsKey.model else { return nil }
        return chainLimits.limits(host: host.id, model: model, fps: controller.draft.fps)
    }

    func refreshChainLimits() {
        guard let host, let model = chainLimitsKey.model,
              let backend = hosts.backend(for: host.id) else { return }
        chainLimits.refresh(host: host.id, model: model,
                            fps: controller.draft.fps, backend: backend)
    }
}
