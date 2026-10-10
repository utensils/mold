import MoldClient

/// Window-local rewrite undo survives opening and closing the prompt sheet.
/// Later authoring or settings changes permanently supersede that rewrite.
struct PromptExpansionUndo {
    private var before: String?
    private var after: RenderDraft?

    mutating func record(original: RenderDraft, expanded: String) {
        before = original.prompt
        var updated = original
        updated.prompt = expanded
        after = updated
    }

    mutating func observe(_ draft: RenderDraft) {
        if let after, draft != after { self = PromptExpansionUndo() }
    }

    func original(for draft: RenderDraft) -> String? {
        draft == after ? before : nil
    }
}

/// Clear can temporarily retire the live rewrite, then Undo clear restores it
/// only while the rest of the draft still agrees with the cleared snapshot.
struct ClearedPromptExpansion {
    let clearedDraft: RenderDraft
    let undo: PromptExpansionUndo

    func matches(_ draft: RenderDraft) -> Bool { draft == clearedDraft }

    func restored(for draft: RenderDraft) -> PromptExpansionUndo? {
        var cleared = draft
        cleared.prompt = ""
        cleared.originalPrompt = nil
        cleared.promptTransform = nil
        return matches(cleared) ? undo : nil
    }
}
