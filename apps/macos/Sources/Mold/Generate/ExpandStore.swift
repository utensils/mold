import Foundation
import MoldClient

/// The prompt wand: rewriting a prompt in place, and what an accepted rewrite
/// replaced.
///
/// Its own object rather than more of `GenerateController`: the controller is
/// what is being authored and what the host says about it, and a rewrite is a
/// round trip of its own -- its own backend calls, its own staleness fence,
/// its own refusals and its own result, none of which a submit ever reads. It
/// takes the controller as a parameter because the prompt it rewrites lives in
/// that draft, the arrangement `LibraryMutations` already has with
/// `LibraryStore`.
@MainActor
@Observable
final class ExpandStore {
    /// Where a prompt rewrite stands. The wand reads this and nothing else to
    /// decide what its popover shows.
    var expansion: Expansion = .idle

    /// What `revert(_:)` puts back, and until when -- see `canRevert(_:)`.
    var lastAcceptedPrompt: LastAcceptedPrompt?

    /// How many rewrites to ask for. Expand's server default is 1 and remix's
    /// is 3; the app asks for three either way, because a popover you arrow
    /// through with one row in it is a popover that should have been a
    /// button.
    static let expansionChoices = 3

    /// Rewrites the prompt into a generation-aware one.
    ///
    /// `modelFamily` is what `/api/models` reported for the chosen model --
    /// this never maps a model name to a family itself, `ignored_prompt_advice`
    /// resolves the family through the prompting registry on the server.
    func expand(_ controller: GenerateController, on host: MoldHost,
                backend: any MoldBackend) async {
        guard let modelFamily = controller.modelFamily else { return }
        let asked = snapshot(of: controller)
        // Decided BEFORE any call: a 422 naming the model to pull is never
        // parsed, because it hard-codes `qwen3-expand` whatever this host
        // actually configured.
        if let model = controller.hosts.capabilities(of: host)?.expanderModelToPull {
            expansion = .needsModel(model)
            return
        }
        expansion = .working(.expand)
        do {
            let response = try await backend.expand(ExpandRequest(
                prompt: controller.draft.prompt, modelFamily: modelFamily,
                variations: Self.expansionChoices, task: asked.task))
            if let stale = asked.refusalIfStale(against: snapshot(of: controller)) {
                expansion = .refused(stale)
                return
            }
            // A family that reads no prompt is answered with exactly one
            // entry whatever `variations` asked for
            // (`crates/mold-server/src/routes.rs:4064-4073`); a real
            // expansion honours `variations` and returns more.
            if response.expanded.count == 1, Self.expansionChoices > 1 {
                expansion = .advised(response.expanded[0])
            } else {
                expansion = .offering(Expansion.Offer(
                    kind: .expand, original: response.original, task: asked.task,
                    choices: response.expanded.map { Expansion.Choice(prompt: $0, dimensions: []) }))
            }
        } catch is CancellationError {
            expansion = .idle
        } catch {
            // A failed rewrite is about the prompt in front of you, not the
            // machine's health -- the answer belongs in the popover the wand
            // opened, never in the machine banner.
            expansion = .refused(error.failureSentence)
        }
    }

    /// Subject-preserving alternatives. `sourceKind` records whether this
    /// remixed the text as first typed or an already-rewritten current
    /// prompt, which is how a second rewrite keeps the chain straight.
    func remix(_ controller: GenerateController, on host: MoldHost,
               backend: any MoldBackend) async {
        guard let modelFamily = controller.modelFamily else { return }
        let asked = snapshot(of: controller)
        if let model = controller.hosts.capabilities(of: host)?.expanderModelToPull {
            expansion = .needsModel(model)
            return
        }
        expansion = .working(.remix)
        do {
            let response = try await backend.remix(RemixRequest(
                sourcePrompt: controller.draft.prompt,
                rootPrompt: controller.draft.originalPrompt,
                sourceKind: controller.draft.originalPrompt == nil ? .direct : .current,
                modelFamily: modelFamily,
                variations: Self.expansionChoices, task: asked.task))
            if let stale = asked.refusalIfStale(against: snapshot(of: controller)) {
                expansion = .refused(stale)
                return
            }
            expansion = .offering(Expansion.Offer(
                kind: .remix, original: response.sourcePrompt, task: response.task,
                choices: response.variants.map { Expansion.Choice(prompt: $0.prompt, dimensions: $0.dimensions) }))
        } catch is CancellationError {
            expansion = .idle
        } catch {
            expansion = .refused(error.failureSentence)
        }
    }
}
