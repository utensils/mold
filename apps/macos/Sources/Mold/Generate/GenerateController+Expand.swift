import Foundation
import MoldClient

// The prompt wand: rewriting a prompt in place, and the rule that decides
// whether there is anything to offer at all.
@MainActor
extension GenerateController {
    /// How many rewrites to ask for. Expand's server default is 1 and remix's
    /// is 3; the app asks for three either way, because a popover you arrow
    /// through with one row in it is a popover that should have been a
    /// button.
    static let expansionChoices = 3

    /// Whether the wand is offered at all, and why not.
    ///
    /// Three questions, and only the first two are about the machine: no
    /// model chosen is nothing to expand, absence of the whole `expand` block
    /// is UNKNOWN and still offers the control (`mayExpandPrompts` is
    /// `configured ?? true`), and a configured-but-uninstalled local model
    /// names itself rather than being offered. `remix` absent narrows the
    /// wand rather than hiding it. A recipe whose prompt mode is `.ignored`
    /// is deliberately NOT a fourth question here -- that is refused by the
    /// SERVER, at 200, with the family guide's own words
    /// (`crates/mold-server/src/routes.rs:4064-4073`), and hiding the wand
    /// first would mean the guide's sentences are never asked for.
    func expansionOffer(for recipe: GenerationRecipe?, on host: MoldHost) -> ExpansionOffer {
        guard recipe != nil else { return .hidden }
        let capabilities = hosts.capabilities(of: host)
        guard capabilities?.mayExpandPrompts ?? true else { return .hidden }
        if let model = capabilities?.expanderModelToPull { return .needsModel(model) }
        return .wand(canRemix: capabilities?.canRemixPrompts ?? false)
    }

    /// Rewrites the prompt into a generation-aware one.
    ///
    /// `modelFamily` is what `/api/models` reported for the chosen model --
    /// this never maps a model name to a family itself, `ignored_prompt_advice`
    /// resolves the family through the prompting registry on the server.
    func expand(on host: MoldHost, backend: any MoldBackend) async {
        guard let modelFamily else { return }
        // Decided BEFORE any call: a 422 naming the model to pull is never
        // parsed, because it hard-codes `qwen3-expand` whatever this host
        // actually configured.
        if let model = hosts.capabilities(of: host)?.expanderModelToPull {
            expansion = .needsModel(model)
            return
        }
        expansion = .working(.expand)
        do {
            let response = try await backend.expand(ExpandRequest(
                prompt: draft.prompt, modelFamily: modelFamily,
                variations: Self.expansionChoices))
            // A family that reads no prompt is answered with exactly one
            // entry whatever `variations` asked for
            // (`crates/mold-server/src/routes.rs:4064-4073`); a real
            // expansion honours `variations` and returns more.
            if response.expanded.count == 1, Self.expansionChoices > 1 {
                expansion = .advised(response.expanded[0])
            } else {
                expansion = .offering(Expansion.Offer(
                    kind: .expand, original: response.original, task: .textToImage,
                    choices: response.expanded.map { Expansion.Choice(prompt: $0, dimensions: []) }))
            }
        } catch is CancellationError {
            expansion = .idle
        } catch {
            // A failed rewrite is about the prompt in front of you, not the
            // machine's health -- the answer belongs in the popover the wand
            // opened, never in the machine banner.
            expansion = .refused(error.reason)
        }
    }

    /// Subject-preserving alternatives. `sourceKind` records whether this
    /// remixed the text as first typed or an already-rewritten current
    /// prompt, which is how a second rewrite keeps the chain straight.
    func remix(on host: MoldHost, backend: any MoldBackend) async {
        guard let modelFamily else { return }
        if let model = hosts.capabilities(of: host)?.expanderModelToPull {
            expansion = .needsModel(model)
            return
        }
        expansion = .working(.remix)
        do {
            let response = try await backend.remix(RemixRequest(
                sourcePrompt: draft.prompt,
                rootPrompt: draft.originalPrompt,
                sourceKind: draft.originalPrompt == nil ? .direct : .current,
                modelFamily: modelFamily,
                variations: Self.expansionChoices))
            expansion = .offering(Expansion.Offer(
                kind: .remix, original: response.sourcePrompt, task: response.task,
                choices: response.variants.map { Expansion.Choice(prompt: $0.prompt, dimensions: $0.dimensions) }))
        } catch is CancellationError {
            expansion = .idle
        } catch {
            expansion = .refused(error.reason)
        }
    }

    /// Takes a choice: the prompt is REWRITTEN IN PLACE and the original is
    /// kept, so what is submitted is what is on screen. `GenerateRequest` has
    /// no server-side expand field to arm -- a generate-time rewrite would
    /// make this invisible until after the render.
    func accept(_ choice: Expansion.Choice) {
        guard case let .offering(offer) = expansion else { return }
        let previous = LastAcceptedPrompt(
            prompt: choice.prompt, previousPrompt: draft.prompt,
            previousOriginalPrompt: draft.originalPrompt, previousTransform: draft.promptTransform)
        // The root is the earliest idea and survives a second rewrite; the
        // source is always the text that actually went in to THIS rewrite.
        let sourceKind: RemixSourceKind = draft.originalPrompt == nil ? .direct : .current
        draft.originalPrompt = draft.originalPrompt ?? offer.original
        draft.promptTransform = PromptTransformProvenance(
            operation: offer.kind == .expand ? .expand : .remix,
            rootPrompt: draft.originalPrompt,
            sourcePrompt: offer.original,
            sourceKind: sourceKind,
            task: offer.task,
            dimensions: choice.dimensions)
        draft.prompt = choice.prompt
        lastAcceptedPrompt = previous
        expansion = .idle
    }

    /// Whether `revertExpansion()` would do anything. Editing the prompt by
    /// hand after an accept is a different intent from "undo the wand", so
    /// the affordance disappears the moment the draft's prompt no longer
    /// matches what was just accepted.
    var canRevertExpansion: Bool {
        guard let lastAcceptedPrompt else { return false }
        return draft.prompt == lastAcceptedPrompt.prompt
    }

    /// Puts the prompt -- and its provenance -- back to what they were before
    /// the last accept.
    func revertExpansion() {
        guard canRevertExpansion, let lastAcceptedPrompt else { return }
        draft.prompt = lastAcceptedPrompt.previousPrompt
        draft.originalPrompt = lastAcceptedPrompt.previousOriginalPrompt
        draft.promptTransform = lastAcceptedPrompt.previousTransform
        self.lastAcceptedPrompt = nil
    }

    func dismissExpansion() {
        expansion = .idle
    }
}
