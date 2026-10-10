import Testing
@testable import MoldClient

struct PromptEditingSessionTests {
    @Test func undoClearRestoresProvenanceWithoutRevertingOtherEdits() {
        var draft = RenderDraft()
        draft.prompt = "A long prompt\nwith a second paragraph."
        draft.originalPrompt = "Original"
        draft.promptTransform = PromptTransformProvenance(
            operation: .expand, sourcePrompt: "Original", task: .textToImage)
        let provenance = draft.promptTransform
        var session = PromptEditingSession()
        session.clear(&draft)
        #expect(draft.prompt.isEmpty)
        #expect(draft.promptTransform == nil)
        // Changes outside prompt editing must not be rolled back by Undo clear.
        draft.negativePrompt = "Keep this"
        draft.width = 768
        draft.seed = 42
        session.undoClear(&draft)
        #expect(draft.prompt == "A long prompt\nwith a second paragraph.")
        #expect(draft.originalPrompt == "Original")
        #expect(draft.promptTransform == provenance)
        #expect(draft.negativePrompt == "Keep this")
        #expect(draft.width == 768)
        #expect(draft.seed == 42)
        #expect(!session.canUndoClear(prompt: ""))
    }

    @Test func emptyClearDoesNotDestroyRecoveryAndLaterTypingRetiresIt() {
        var draft = RenderDraft()
        draft.prompt = "Draft"
        var session = PromptEditingSession()
        session.clear(&draft)
        session.clear(&draft)
        #expect(session.canUndoClear(prompt: draft.prompt))
        session.observe(prompt: "New text")
        session.observe(prompt: "")
        session.undoClear(&draft)
        #expect(draft.prompt.isEmpty)
        #expect(!session.canUndoClear(prompt: draft.prompt))
    }

    @Test func historyReplacementRetiresClearRecovery() {
        var draft = RenderDraft()
        draft.prompt = "Draft"
        var session = PromptEditingSession()
        session.clear(&draft)
        session.replaced()
        session.undoClear(&draft)
        #expect(draft.prompt.isEmpty)
        #expect(!session.canUndoClear(prompt: draft.prompt))
    }
}
