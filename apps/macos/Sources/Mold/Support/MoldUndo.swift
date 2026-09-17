import Foundation
import MoldClient

/// The Edit menu's Undo, for library edits.
///
/// Thin on purpose. Everything interesting about undoing a library edit --
/// what a change actually altered, and what reverses it -- is `PrintEdit`, a
/// value in `MoldClient` with tests. This is only the bridge to AppKit.
@MainActor
final class MoldUndo {
    /// The WINDOW's undo manager, handed over by the view that owns the
    /// library, never one of our own making.
    ///
    /// SwiftUI's `.undoRedo` command group targets whatever the responder
    /// chain answers with, so a private manager would give us a working stack
    /// and a dead menu item. Taking the window's is also what lets ⌘Z reach a
    /// text field's own manager while one is first responder: the field is
    /// nearer in the chain, and our registrations simply are not what ⌘Z finds.
    var manager: UndoManager?

    /// Registers the inverse of a change that just happened.
    ///
    /// The caller hands over the edit it APPLIED; what is registered is that
    /// edit's inverse. An empty edit registers nothing, which is what keeps
    /// favouriting an already-favourite print out of the Edit menu entirely.
    ///
    /// `apply` must be the same funnel the original change went through. That
    /// is what makes redo free: undoing calls it with the inverse, which
    /// registers the inverse's inverse -- the original change again.
    func register(_ edit: PrintEdit, apply: @escaping (PrintEdit) -> Void) {
        guard !edit.isEmpty else { return }
        let inverse = edit.inverse
        register(edit.actionName) { apply(inverse) }
    }

    /// Registers an inverse for a change that is not about a set of prints --
    /// renaming a tag across the whole library, say.
    ///
    /// `inverse` must go through the same path the original change took. That
    /// is what makes redo free: undoing calls it, and it registers its own
    /// inverse in turn, which is the original change again.
    func register(_ actionName: String, inverse: @escaping () -> Void) {
        guard let manager else { return }
        manager.registerUndo(withTarget: self) { _ in
            // `UndoManager` calls back on whichever thread invoked undo, and
            // for a menu item that is the main one. The store it is about to
            // touch is `@MainActor`, so state the fact rather than hopping --
            // a hop would let a second undo start before the first finished.
            MainActor.assumeIsolated { inverse() }
        }
        // While undoing, the registration above IS the redo entry, and naming
        // it after the inverse would put "Redo Unfavorite" in the menu of
        // someone who asked to undo a favourite.
        if !manager.isUndoing {
            manager.setActionName(actionName)
        }
    }

    /// Forgets OUR entries. For a change that makes them meaningless -- a tag
    /// deleted everywhere, or an edit a machine refused.
    ///
    /// Scoped to this target, because `manager` is the WINDOW's and is shared
    /// with every `NSTextField` field editor in it: `removeAllActions()` threw
    /// away the rename you were about to undo and whatever a focused field had
    /// recorded. `register` already registers `withTarget: self`, so the
    /// targeted form removes exactly what this object put there.
    func forget() {
        manager?.removeAllActions(withTarget: self)
    }
}
