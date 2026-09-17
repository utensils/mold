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

    public init(pool: [LibraryEntry], query: LibraryQuery, selection: Set<PrintID>) {
        self.pool = pool
        self.visible = query.apply(to: pool)
        self.sections = LibraryGrouping.byDay(visible)
        self.selected = visible.filter { selection.contains($0.id) }
    }
}
