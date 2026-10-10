import AppKit
import MoldClient
import Testing

@testable import Mold

struct PromptEditingTests {
    @Test func clearRestoresTextAndProvenanceUntilAnotherEdit() {
        var draft = RenderDraft()
        draft.prompt = "First line\nSecond line"
        draft.originalPrompt = "Original"
        var editing = PromptEditingSession()
        editing.clear(&draft)
        #expect(draft.prompt.isEmpty)
        #expect(draft.originalPrompt == nil)
        #expect(editing.canUndoClear(prompt: draft.prompt))
        editing.undoClear(&draft)
        #expect(draft.prompt == "First line\nSecond line")
        #expect(draft.originalPrompt == "Original")
        editing.clear(&draft)
        editing.observe(prompt: "Later edit")
        #expect(!editing.canUndoClear(prompt: ""))
    }

    @Test func replacementOnlyChangesPromptAndItsProvenance() {
        var draft = RenderDraft()
        draft.prompt = "Current"
        draft.originalPrompt = "Old root"
        draft.negativePrompt = "Keep negative"
        draft.width = 1024
        PromptHistoryRecall.apply("Chosen\nHistory", to: &draft)
        #expect(draft.prompt == "Chosen\nHistory")
        #expect(draft.originalPrompt == nil)
        #expect(draft.negativePrompt == "Keep negative")
        #expect(draft.width == 1024)
    }

    @Test func largeEditorNeverRecallsHistoryAndLeavesKeysToNativeTextEditing() {
        #expect(!PromptEditingKeyboard.allowsHistory(editorOpen: true, promptFocused: true))
        #expect(!PromptEditingKeyboard.allowsHistory(editorOpen: true, promptFocused: false))
        #expect(PromptEditingKeyboard.allowsHistory(editorOpen: false, promptFocused: true))
        #expect(!PromptEditingKeyboard.allowsHistory(editorOpen: false, promptFocused: false))
    }

    @Test func compactPromptStaysBounded() {
        #expect(PromptEditingLayout.compactHeight(preferred: 560, available: 800) == 144)
        #expect(PromptEditingLayout.compactHeight(preferred: 72, available: 24) == 24)
    }
    @MainActor @Test func backgroundWandNeverPresentsOrDismissesEditorRewrite() {
        let expansions = ExpandStore()
        expansions.expansion = .advised("Advice")
        let background = PromptWand.popoverBinding(expansions: expansions, enabled: false)
        #expect(!background.wrappedValue)
        background.wrappedValue = false
        if case .advised = expansions.expansion {} else { Issue.record("Background dismissed editor rewrite") }
        let editor = PromptWand.popoverBinding(expansions: expansions, enabled: true)
        #expect(editor.wrappedValue)
        editor.wrappedValue = false
        if case .idle = expansions.expansion {} else { Issue.record("Editor dismissal was ignored") }
    }

}
