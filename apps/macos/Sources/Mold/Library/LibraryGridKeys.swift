import MoldClient
import SwiftUI

/// Keys are answered only by the focused Library grid or viewer.
enum LibraryGridAction: Hashable {
    case move(LibraryCursor.Move, LibraryCursor.Modifier)
    case open
    case quickLook
    case trash
    case favorite
    case emptyTrash
    case clearSelection
}

enum LibraryGridKeys {
    /// AppKit sends U+007F for the physical backward-delete key, while
    /// SwiftUI's canonical `.delete` is U+0008.
    static let physicalDelete = KeyEquivalent("\u{7f}")
    static let viewerKeys: Set<KeyEquivalent> = [.delete, .deleteForward, physicalDelete, "f"]

    /// Every key the grid answers, so the binding and the map cannot drift.
    static let keys: Set<KeyEquivalent> = [
        .leftArrow, .rightArrow, .upArrow, .downArrow, .return, .space, .delete, .deleteForward, physicalDelete, .escape, "f",
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
        case .escape: return held.isEmpty ? .clearSelection : nil
        case "f": return held.isEmpty ? .favorite : nil
        case .delete, .deleteForward, physicalDelete:
            if held == [.command, .shift] { return .emptyTrash }
            return held.isEmpty || held == .command ? .trash : nil
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
