import AppKit
import Foundation
import MoldClient

enum PromptHistoryRecall {
    static func apply(_ prompt: String, to draft: inout RenderDraft) {
        draft.prompt = prompt
        draft.originalPrompt = nil
        draft.promptTransform = nil
    }
}

/// Shell-style history navigation. Entries arrive newest first from the host.
struct PromptHistoryCycler {
    private var entries: [String] = []
    private var cursor: Int?
    private var draft = ""

    mutating func setEntries(_ prompts: [String]) {
        entries = prompts
        cursor = nil
    }

    mutating func previous(from current: String) -> String? {
        if let cursor, entries.indices.contains(cursor), entries[cursor] != current {
            self.cursor = nil // the person edited a recalled prompt
        }
        guard !entries.isEmpty else { return nil }
        if let cursor {
            guard cursor + 1 < entries.count else { return nil }
            self.cursor = cursor + 1
        } else {
            draft = current
            cursor = 0
        }
        return entries[cursor!]
    }

    mutating func next(from current: String) -> String? {
        guard let cursor, entries.indices.contains(cursor), entries[cursor] == current else {
            self.cursor = nil
            return nil
        }
        if cursor == 0 {
            self.cursor = nil
            return draft
        }
        self.cursor = cursor - 1
        return entries[cursor - 1]
    }
}

enum PromptHistoryCaret {
    static func isOnFirstLine(_ text: String, selection: NSRange) -> Bool {
        let ns = text as NSString
        guard selection.location <= ns.length else { return false }
        return !ns.substring(to: selection.location).contains("\n")
    }

    static func isOnLastLine(_ text: String, selection: NSRange) -> Bool {
        let ns = text as NSString
        guard NSMaxRange(selection) <= ns.length else { return false }
        return !ns.substring(from: NSMaxRange(selection)).contains("\n")
    }
}
