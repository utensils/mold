import Foundation
import Testing

@testable import MoldClient

// `LibraryShowing` derives `visible`, `sections` and `selected` once from a
// pool and a query, so a pane asking for all three in one body pass gets one
// filter-and-sort rather than three.

private let workstation = UUID()

private func entry(_ name: String, at seconds: UInt64 = 1_000,
                   favorite: Bool = false) -> LibraryEntry {
    PrintFixtures.entry(name, host: workstation, timestamp: seconds, favorite: favorite)
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

/// The grid draws `sections` and the viewer's ← → walk `visible`, so they have
/// to BE the same list -- and both have to be what Sort By asked for.
///
/// **Fails today**: `LibraryGrouping.byDay` re-sorts every bucket newest-first
/// and the days descending (`LibrarySection.swift:24-28`), so only `.newest`
/// ever reaches the grid and under Oldest First the two walks move in opposite
/// directions. The old assertion compared `sections` against `byDay(...)` --
/// it re-derived the implementation and so could never fail.
@Test(arguments: [LibrarySort.newest, .oldest, .largest, .name])
func theGridDrawsTheOrderTheQueryAskedFor(_ sort: LibrarySort) {
    let pool = [
        PrintFixtures.entry("robot.png", host: workstation, timestamp: 300, bytes: 10),
        PrintFixtures.entry("turtle.png", host: workstation, timestamp: 200, bytes: 900),
        PrintFixtures.entry("chair.png", host: workstation, timestamp: 100_000, bytes: 50),
        PrintFixtures.entry("anvil.png", host: workstation, timestamp: 100, bytes: 1),
    ]
    var query = LibraryQuery()
    query.sort = sort

    let showing = LibraryShowing(pool: pool, query: query, selection: [])

    #expect(showing.sections.flatMap(\.items).map(\.id) == showing.visible.map(\.id))
    #expect(showing.visible.map(\.id) == query.apply(to: pool).map(\.id))
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
