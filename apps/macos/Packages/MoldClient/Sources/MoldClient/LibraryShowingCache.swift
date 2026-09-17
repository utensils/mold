import Foundation

/// The filtered, sorted and grouped library, derived ONCE per data or query
/// change.
///
/// `LibraryShowing` exists so that a pane's body computes this once per PASS
/// rather than eight times, and it does. The missing half was that a pass
/// happens on every selection change: an arrow key, a click, an inspector
/// edit, a character typed into search -- each one re-filtered and re-sorted
/// the whole merged library and rebuilt a dictionary of day buckets, on the
/// main actor, at ten thousand prints. The desktop's stated invariant is the
/// opposite: `organizationIndex` and `bucketIndex` compute once per DATA
/// change (`.claude/rules/desktop.md`), and the tiles are immutable snapshots.
///
/// So the expensive half is keyed on what it actually depends on -- the rows
/// and the query -- and the cheap half, which rows are selected, is answered
/// from a position map instead of another scan.
///
/// `revision` is the rows' own counter (`LibraryStore.revision`) rather than a
/// comparison: two libraries of the same size differ by one print's favourite
/// star, and comparing them would cost exactly what this avoids.
@MainActor
public final class LibraryShowingCache {
    private struct Key: Hashable {
        let revision: Int
        let count: Int
        /// The ends of the pool, so switching between two lists of the same
        /// size at the same revision -- the timeline and the trash -- is a
        /// different key.
        let first: PrintID?
        let last: PrintID?
        let query: LibraryQuery
    }

    private var key: Key?
    /// The derivation itself, with no selection applied -- `LibraryShowing` is
    /// still the one definition of what filtering, sorting and grouping mean;
    /// this only remembers the answer.
    private var derived = LibraryShowing(pool: [], visible: [], sections: [], selected: [])
    /// Where each visible row sits, so a selection resolves in the order the
    /// grid draws without walking the whole list.
    private var positions: [PrintID: Int] = [:]

    public init() {}

    /// How many times the expensive derivation has actually run. The budget a
    /// test asserts on.
    public private(set) var derivations = 0

    public func showing(pool: [LibraryEntry], revision: Int, query: LibraryQuery,
                        selection: Set<PrintID>) -> LibraryShowing {
        let key = Key(revision: revision, count: pool.count, first: pool.first?.id,
                      last: pool.last?.id, query: query)
        if key != self.key {
            self.key = key
            derived = LibraryShowing(pool: pool, query: query, selection: [])
            positions = Dictionary(uniqueKeysWithValues: derived.visible.enumerated()
                .map { ($0.element.id, $0.offset) })
            derivations += 1
        }
        return LibraryShowing(pool: derived.pool, visible: derived.visible,
                              sections: derived.sections, selected: selected(selection))
    }

    private func selected(_ selection: Set<PrintID>) -> [LibraryEntry] {
        guard !selection.isEmpty else { return [] }
        return selection.compactMap { positions[$0] }.sorted().map { derived.visible[$0] }
    }
}
