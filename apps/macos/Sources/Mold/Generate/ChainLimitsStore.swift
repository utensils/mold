import MoldClient
import SwiftUI

/// What each machine says it will chain, per model and rate.
///
/// Read ONCE per `(machine, model, fps)` and then held: the answer is a
/// property of the checkpoint and the host's own configuration, neither of
/// which moves while the pane is open. A host that does not publish the route
/// -- an older mold -- answers nothing at all, and `ChainRouting`'s ported
/// constants stand, which is exactly what absence is supposed to mean.
@MainActor
@Observable
final class ChainLimitsStore {
    private struct Key: Hashable {
        let host: MoldHost.ID
        let model: String
        let fps: Int?
    }

    private var answers: [Key: ChainLimits] = [:]
    /// Keys already asked, answered or not -- so an older host is asked once
    /// rather than on every keystroke that moves the Length slider.
    private var asked: Set<Key> = []
    private var epochs: [MoldHost.ID: Int] = [:]

    /// This machine's limits for that model, or `nil` while the answer is
    /// still in the air or the host has none.
    func limits(host: MoldHost.ID, model: String, fps: Int?) -> ChainLimits? {
        answers[Key(host: host, model: model, fps: fps)]
    }

    func invalidate(host: MoldHost.ID) {
        epochs[host, default: 0] += 1
        answers = answers.filter { $0.key.host != host }
        asked = asked.filter { $0.host != host }
    }

    /// Asks, once. Deliberately silent on failure: an unpublished route and an
    /// unreachable machine both mean "no advertised limits", and a banner
    /// about a capability probe would be noise over a control that still works.
    func refresh(host: MoldHost.ID, model: String, fps: Int?, backend: any MoldChainBackend) {
        let key = Key(host: host, model: model, fps: fps)
        guard asked.insert(key).inserted else { return }
        let epoch = epochs[host, default: 0]
        Task { [weak self] in
            guard let limits = try? await backend.chainLimits(model: model, fps: fps)
            else { return }
            guard let self, self.epochs[host, default: 0] == epoch else { return }
            self.answers[key] = limits
        }
    }
}
