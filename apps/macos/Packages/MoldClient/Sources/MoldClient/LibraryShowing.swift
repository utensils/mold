import Foundation

/// The three lists the grid draws, derived ONCE.
///
/// `visible` is a full filter and sort of the whole library; `LibraryPane`'s
/// body was asking for it seven or eight times per pass, and `sections` and
/// `selected` each asked again.
public struct LibraryShowing: Sendable {
    public let pool: [LibraryEntry]
    public let visible: [LibraryEntry]
    public let sections: [LibrarySection]
    public let selected: [LibraryEntry]

    /// The three lists as a cache already has them, with only the selection
    /// applied. See `LibraryShowingCache`: a pass that moved the cursor must
    /// not re-filter, re-sort and re-group the whole library.
    public init(pool: [LibraryEntry], visible: [LibraryEntry],
                sections: [LibrarySection], selected: [LibraryEntry]) {
        self.pool = pool
        self.visible = visible
        self.sections = sections
        self.selected = selected
    }

    public init(pool: [LibraryEntry], query: LibraryQuery, selection: Set<PrintID>) {
        self.pool = pool
        self.visible = query.apply(to: pool)
        // The sections ARE `visible`, cut -- never re-ordered. The grid draws
        // these and the viewer's ← → walk `visible`, so anything else is two
        // orders for one list.
        self.sections = query.sort.groupsByDay
            ? LibraryGrouping.byDay(visible)
            : LibraryGrouping.ungrouped(visible)
        self.selected = visible.filter { selection.contains($0.id) }
    }
}
