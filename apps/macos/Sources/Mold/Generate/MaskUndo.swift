import Foundation

/// Which `UndoManager` the mask sheet's ⌘Z registers a stroke's inverse into.
///
/// A sheet is not guaranteed a first responder SwiftUI is willing to hand an
/// `@Environment(\.undoManager)` for, so the sheet always keeps one of its
/// own as a compiled-in fallback. `MoldUndo` is not that fallback: it holds
/// the Library window's manager, and its registrations are `PrintEdit`
/// inverses -- interleaving "Undo Stroke" into that stack would make a
/// favourite and a brush dab share one Edit menu entry.
enum MaskUndo {
    static func resolve(environment: UndoManager?, own: UndoManager) -> UndoManager {
        environment ?? own
    }
}
