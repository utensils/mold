import Foundation
import MoldClient

// Taking a rewrite, putting one back, and the facts one is asked against.
@MainActor
extension ExpandStore {
    /// Takes a choice: the prompt is REWRITTEN IN PLACE and the original is
    /// kept, so what is submitted is what is on screen. `GenerateRequest` has
    /// no server-side expand field to arm -- a generate-time rewrite would
    /// make this invisible until after the render.
    func accept(_ choice: Expansion.Choice, into controller: GenerateController) {
        guard case let .offering(offer) = expansion else { return }
        let previous = LastAcceptedPrompt(
            prompt: choice.prompt, previousPrompt: controller.draft.prompt,
            previousOriginalPrompt: controller.draft.originalPrompt,
            previousTransform: controller.draft.promptTransform)
        // The root is the earliest idea and survives a second rewrite; the
        // source is always the text that actually went in to THIS rewrite.
        let sourceKind: RemixSourceKind = controller.draft.originalPrompt == nil ? .direct : .current
        controller.draft.originalPrompt = controller.draft.originalPrompt ?? offer.original
        controller.draft.promptTransform = PromptTransformProvenance(
            operation: offer.kind == .expand ? .expand : .remix,
            rootPrompt: controller.draft.originalPrompt,
            sourcePrompt: offer.original,
            sourceKind: sourceKind,
            task: offer.task,
            dimensions: choice.dimensions)
        controller.draft.prompt = choice.prompt
        lastAcceptedPrompt = previous
        expansion = .idle
    }

    /// Whether `revert(_:)` would do anything. Editing the prompt by hand
    /// after an accept is a different intent from "undo the wand", so the
    /// affordance disappears the moment the draft's prompt no longer matches
    /// what was just accepted.
    func canRevert(_ controller: GenerateController) -> Bool {
        guard let lastAcceptedPrompt else { return false }
        return controller.draft.prompt == lastAcceptedPrompt.prompt
    }

    /// Puts the prompt -- and its provenance -- back to what they were before
    /// the last accept.
    func revert(_ controller: GenerateController) {
        guard canRevert(controller), let lastAcceptedPrompt else { return }
        controller.draft.prompt = lastAcceptedPrompt.previousPrompt
        controller.draft.originalPrompt = lastAcceptedPrompt.previousOriginalPrompt
        controller.draft.promptTransform = lastAcceptedPrompt.previousTransform
        self.lastAcceptedPrompt = nil
    }

    func dismiss() {
        expansion = .idle
    }

    /// The facts a rewrite is asked against. The task comes from the REQUEST
    /// this draft would submit, not from the family alone (findings 01#13,
    /// 02#12) -- `ExpandTask.forRequest`.
    func snapshot(of controller: GenerateController) -> ExpansionSnapshot {
        ExpansionSnapshot(
            prompt: controller.draft.prompt, model: controller.modelName,
            family: controller.modelFamily,
            task: ExpandTask.forRequest(
                family: controller.modelFamily,
                request: controller.draft.request(model: controller.modelName ?? "")),
            host: controller.machineChoice ?? controller.hostID)
    }
}
