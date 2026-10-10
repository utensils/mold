import Foundation

/// The explicit Clear recovery keeps provenance with the text it belongs to.
public struct PromptEditingSession {
    public init() {}
    private struct Cleared {
        let prompt: String
        let original: String?
        let transform: PromptTransformProvenance?
    }
    private var cleared: Cleared?

    public mutating func clear(_ draft: inout RenderDraft) {
        guard !draft.prompt.isEmpty else { return }
        cleared = Cleared(prompt: draft.prompt, original: draft.originalPrompt, transform: draft.promptTransform)
        draft.prompt = ""
        draft.originalPrompt = nil
        draft.promptTransform = nil
    }

    public func canUndoClear(prompt: String) -> Bool { cleared != nil && prompt.isEmpty }

    public mutating func observe(prompt: String) {
        if !prompt.isEmpty { cleared = nil }
    }

    public mutating func replaced() { cleared = nil }

    public mutating func undoClear(_ draft: inout RenderDraft) {
        guard canUndoClear(prompt: draft.prompt), let cleared else { return }
        draft.prompt = cleared.prompt
        draft.originalPrompt = cleared.original
        draft.promptTransform = cleared.transform
        self.cleared = nil
    }
}
