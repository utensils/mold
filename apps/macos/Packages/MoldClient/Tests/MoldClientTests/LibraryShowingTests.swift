import Foundation
import Testing

@testable import MoldClient

// `LibraryShowing` derives `visible`, `sections` and `selected` once from a
// pool and a query, so a pane asking for all three in one body pass gets one
// filter-and-sort rather than three.

private let plato = UUID()

private func entry(_ name: String, at seconds: UInt64 = 1_000,
                   favorite: Bool = false) -> LibraryEntry {
    PrintFixtures.entry(name, host: plato, timestamp: seconds, favorite: favorite)
}

@Test func theThreeListsAgreeWithApplyingTheQueryByHand() {
    let pool = [
        entry("robot.png", at: 300),
        entry("turtle.png", at: 200, favorite: true),
        entry("chair.png", at: 100),
    ]
    var query = LibraryQuery()
    query.tokens = [.favorite]

    let showing = LibraryShowing(pool: pool, query: query, selection: [])

    #expect(showing.pool.map(\.print.filename) == pool.map(\.print.filename))
    #expect(showing.visible.map(\.print.filename) == query.apply(to: pool).map(\.print.filename))
    #expect(showing.sections.flatMap(\.items).map(\.print.filename)
        == LibraryGrouping.byDay(query.apply(to: pool)).flatMap(\.items).map(\.print.filename))
}

@Test func aSelectionNamingARowTheQueryExcludesIsNotShownAsSelected() {
    let visible = entry("robot.png", at: 300)
    let excluded = entry("turtle.png", at: 200, favorite: true)
    let pool = [visible, excluded]
    var query = LibraryQuery()
    query.tokens = [.favorite] // only turtle.png matches, so robot.png never becomes "visible"

    // Select BOTH rows -- one the query keeps, one it drops.
    let showing = LibraryShowing(pool: pool, query: query, selection: [visible.id, excluded.id])

    #expect(showing.visible.map(\.print.filename) == ["turtle.png"])
    #expect(showing.selected.map(\.print.filename) == ["turtle.png"])
}
