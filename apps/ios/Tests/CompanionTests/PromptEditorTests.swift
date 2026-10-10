import Foundation
import MoldClient
import Testing
@testable import MoldCompanion

@MainActor struct PromptEditorTests {
    @Test func clearUndoRestoresOnlyPromptFields() {
        var draft = RenderDraft()
        draft.prompt = "First line\nSecond line"
        draft.originalPrompt = "Before rewrite"
        draft.negativePrompt = "noise"
        var editing = PromptEditingSession()
        editing.clear(&draft)
        #expect(draft.prompt.isEmpty)
        #expect(editing.canUndoClear(prompt: draft.prompt))
        draft.negativePrompt = "later settings"
        editing.undoClear(&draft)
        #expect(draft.prompt == "First line\nSecond line")
        #expect(draft.originalPrompt == "Before rewrite")
        #expect(draft.negativePrompt == "later settings")
    }

    @Test func replacementEndsClearUndo() {
        var draft = RenderDraft()
        draft.prompt = "old"
        var editing = PromptEditingSession()
        editing.clear(&draft)
        editing.replaced()
        PromptHistoryStore.recall("history\ntext", into: &draft)
        editing.undoClear(&draft)
        #expect(draft.prompt == "history\ntext")
        #expect(!editing.canUndoClear(prompt: draft.prompt))
    }

    @Test func dateSeparatorsDefaultOnAndPersistOff() throws {
        let name = "PromptEditorTests-\(UUID())"
        let defaults = try #require(UserDefaults(suiteName: name))
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(Preference.isOn(Preference.showDateSeparators, defaults: defaults))
        defaults.set(false, forKey: Preference.showDateSeparators)
        #expect(!Preference.isOn(Preference.showDateSeparators, defaults: defaults))
    }
    @Test func rewriteUndoSurvivesReopenButNotLaterDraftChanges() {
        var draft = RenderDraft()
        draft.prompt = "original"
        var undo = PromptExpansionUndo()
        undo.record(original: draft, expanded: "expanded")
        draft.prompt = "expanded"
        undo.observe(draft)
        #expect(undo.original(for: draft) == "original")
        draft.negativePrompt = "later setting"
        undo.observe(draft)
        draft.negativePrompt = ""
        #expect(undo.original(for: draft) == nil)
    }

    @Test func clearUndoRestoresRewriteReceiptOnlyForMatchingSettings() {
        var draft = RenderDraft()
        draft.prompt = "original"
        var undo = PromptExpansionUndo()
        undo.record(original: draft, expanded: "expanded")
        draft.prompt = "expanded"
        var editing = PromptEditingSession()
        editing.clear(&draft)
        let receipt = ClearedPromptExpansion(clearedDraft: draft, undo: undo)
        undo.observe(draft)
        #expect(undo.original(for: draft) == nil)
        editing.undoClear(&draft)
        #expect(receipt.restored(for: draft)?.original(for: draft) == "original")
        draft.negativePrompt = "changed elsewhere"
        #expect(receipt.restored(for: draft) == nil)
    }

}
