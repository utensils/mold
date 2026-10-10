import AppKit
import Foundation
import MoldClient

enum PromptEditingLayout {
    static func compactHeight(preferred: CGFloat, available: CGFloat) -> CGFloat {
        PromptEditorHeight.resolve(preferred, available: min(144, available))
    }
}

enum PromptEditingKeyboard {
    static func allowsHistory(editorOpen: Bool, promptFocused: Bool) -> Bool {
        !editorOpen && promptFocused
    }
}
