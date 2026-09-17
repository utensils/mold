import Foundation
import MoldClient

/// What a rewrite was asked FOR, frozen so a landed answer can be refused by
/// name when the box moved while it was in the air.
///
/// `expand` and `remix` installed whatever came back, even if the model, the
/// family, the prompt or the machine had changed in the meantime -- and
/// `accept` then wrote a `sourcePrompt` that was never in the box (finding
/// 02#13). Desktop and web refuse a landed rewrite by name for exactly these
/// facts; the sentences are studio's own
/// (`studio/lib/preparedExpansion.ts:171-195`).
struct ExpansionSnapshot: Equatable {
    let prompt: String
    let model: String?
    let family: String?
    let task: ExpandTask
    let host: MoldHost.ID?

    /// Why the answer that just landed no longer belongs to what is on screen.
    /// Empty means it does.
    func staleReasons(against current: ExpansionSnapshot) -> [String] {
        var reasons: [String] = []
        if current.prompt != prompt {
            reasons.append("The prompt changed after the rewrite was asked for.")
        }
        if current.model != model {
            reasons.append("Style changed from \"\(model ?? "none")\" to \"\(current.model ?? "none")\".")
        }
        if current.family != family {
            reasons.append(
                "Style family changed from \"\(family ?? "none")\" to \"\(current.family ?? "none")\".")
        }
        if current.task != task {
            reasons.append(
                "Conditioning changed from \(task.rawValue) to \(current.task.rawValue).")
        }
        if current.host != host {
            reasons.append("The machine changed while the rewrite was in flight.")
        }
        return reasons
    }

    /// The one sentence to show instead of installing a landed rewrite, or
    /// nil when it still belongs to what is on screen.
    func refusalIfStale(against current: ExpansionSnapshot) -> String? {
        guard let first = staleReasons(against: current).first else { return nil }
        return "\(first) Ask again to rewrite what is in the box now."
    }
}
