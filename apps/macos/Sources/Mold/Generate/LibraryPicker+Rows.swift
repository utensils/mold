import Foundation
import MoldClient

/// The grid behind "From Library…" (M8 design, decision 5): the fleet's
/// pictures, never a clip or a mesh, newest first, optionally narrowed by a
/// search query -- pure, so the sheet needs no store to test.
enum LibraryPicker {}

extension LibraryPicker {
    static func rows(_ entries: [LibraryEntry], query: String) -> [LibraryEntry] {
        let isBlank = query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let folded = LibraryEntry.fold(query)
        // The store's `items` already come out newest first -- `filter`
        // preserves that order, so this must never sort.
        return entries.filter { entry in
            guard entry.print.kind == .picture, entry.print.trashedAt == nil else { return false }
            return isBlank || entry.searchKey.contains(folded)
        }
    }
}
