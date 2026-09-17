import MoldClient
import SwiftUI

/// What one key press means to the grid.
///
/// A map, and testable as one, because both bugs here were about a MODIFIER
/// the handler never looked at. The arrows took `ClickModifiers.current` --
/// deliberately a record of the last left-mouse-DOWN, which is the right
/// answer for a deferred tap and the wrong one for a key press -- so ⇧←
/// never extended a selection and, after any shift-click, every bare arrow
/// extended one until the next unmodified click. And `.onKeyPress(.delete)`
/// matches its key whatever is held, so a bare Backspace moved the selection
/// to the trash: the menu and the README both promise ⌘⌫, and in the Finder
/// a bare Delete does nothing at all.
enum LibraryGridAction: Hashable {
    case move(LibraryCursor.Move, LibraryCursor.Modifier)
    case open
    case quickLook
    case trash
}

enum LibraryGridKeys {
    /// Every key the grid answers, so the binding and the map cannot drift.
    static let keys: Set<KeyEquivalent> = [
        .leftArrow, .rightArrow, .upArrow, .downArrow, .return, .space, .delete,
    ]

    static func action(for key: KeyEquivalent,
                       modifiers: EventModifiers) -> LibraryGridAction? {
        // Caps lock and the numeric-pad flag say nothing about intent.
        let held = modifiers.intersection([.shift, .command, .option, .control])
        switch key {
        case .leftArrow: return arrow(.left, held)
        case .rightArrow: return arrow(.right, held)
        case .upArrow: return arrow(.up, held)
        case .downArrow: return arrow(.down, held)
        case .return: return held.isEmpty ? .open : nil
        case .space: return held.isEmpty ? .quickLook : nil
        // ⌘⌫, and only ⌘⌫. The menu item owns the same chord and a main-menu
        // key equivalent is consumed first, so this is what answers when the
        // menu is not in play -- never a bare Backspace.
        case .delete: return held == .command ? .trash : nil
        default: return nil
        }
    }

    /// Shift extends; nothing else means anything to an arrow.
    ///
    /// Command is deliberately NOT `.toggle` the way a click's is: toggling
    /// needs something you pointed at, and ⌘← on a Mac is Back. An arrow with
    /// anything but shift is left for whoever else wants it.
    private static func arrow(_ move: LibraryCursor.Move,
                              _ held: EventModifiers) -> LibraryGridAction? {
        guard held.subtracting(.shift).isEmpty else { return nil }
        return .move(move, held.contains(.shift) ? .extend : .none)
    }
}
