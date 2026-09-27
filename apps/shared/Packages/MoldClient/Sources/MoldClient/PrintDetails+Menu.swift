import Foundation

/// What a detail row offers when you right-click it.
public enum PrintDetailAction: Hashable, Sendable {
    /// Put this one value on the pasteboard.
    case copy
    /// Something the Library already offers about the whole print, performed
    /// through the Library's own door.
    case library(LibraryAction)
}

public extension PrintDetails {
    /// Every row is copyable -- the point of showing a seed is that somebody
    /// copies it -- and a row on a print the Library would let you reuse also
    /// offers that.
    ///
    /// The reuse item is NOT written here: it is filtered out of the plan the
    /// tile and the menu bar already draw, so it keeps that one wording and
    /// that one gate. Passing `offering:` a plan that does not carry it (a
    /// trashed print, a multiple selection) leaves the row with copying alone.
    static func menu(for row: PrintDetailRow,
                     offering plan: [RowAction<LibraryAction>]) -> [RowAction<PrintDetailAction>] {
        let copy = RowAction<PrintDetailAction>(kind: .copy, title: row.copyTitle)
        let reuse = plan
            .filter { $0.kind == .reuse }
            .map { $0.mapKind(PrintDetailAction.library) }
        guard !reuse.isEmpty else { return [copy] }
        return [copy, .separator] + reuse
    }
}
