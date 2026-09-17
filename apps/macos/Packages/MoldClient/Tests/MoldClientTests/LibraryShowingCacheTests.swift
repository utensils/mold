import Foundation
import Testing

@testable import MoldClient

/// How often the library is actually filtered, sorted and grouped.
///
/// **Fails today**: `LibraryPane.swift:43` builds a `LibraryShowing` in its
/// body, and a body pass happens on every arrow key, click and keystroke --
/// so the whole derivation ran per PASS, not per data change. There is no
/// cache to count.
@MainActor
struct LibraryShowingCacheTests {
    private let plato = UUID()

    private func pool(_ count: Int) -> [LibraryEntry] {
        (0 ..< count).map {
            PrintFixtures.entry("print-\($0).png", host: plato, timestamp: UInt64(1_000 + $0))
        }
    }

    @Test func movingTheSelectionDoesNotReDeriveTheLibrary() {
        let cache = LibraryShowingCache()
        let rows = pool(50)
        let query = LibraryQuery()

        let first = cache.showing(pool: rows, revision: 1, query: query, selection: [])
        for row in rows.prefix(10) {
            _ = cache.showing(pool: rows, revision: 1, query: query, selection: [row.id])
        }

        #expect(cache.derivations == 1)
        #expect(first.visible.count == 50)
    }

    @Test func aChangeToTheRowsReDerivesEvenAtTheSameSize() {
        let cache = LibraryShowingCache()
        let rows = pool(3)

        _ = cache.showing(pool: rows, revision: 1, query: LibraryQuery(), selection: [])
        // Same count, same ends -- one print's star changed, and the store
        // said so by bumping its revision.
        _ = cache.showing(pool: rows, revision: 2, query: LibraryQuery(), selection: [])

        #expect(cache.derivations == 2)
    }

    /// **Fails today**: the calendar is not in the key, so an app left open
    /// across midnight keeps yesterday's cut -- and the headings, which ARE
    /// recomputed every pass, then read "Today" over yesterday's prints.
    @Test func aNewDayReDerivesTheCut() {
        let cache = LibraryShowingCache()
        let rows = pool(3)
        let today = Date(timeIntervalSince1970: 1_789_560_000)

        _ = cache.showing(pool: rows, revision: 1, query: LibraryQuery(), selection: [],
                          now: today)
        _ = cache.showing(pool: rows, revision: 1, query: LibraryQuery(), selection: [],
                          now: today.addingTimeInterval(60))
        #expect(cache.derivations == 1)

        _ = cache.showing(pool: rows, revision: 1, query: LibraryQuery(), selection: [],
                          now: today.addingTimeInterval(86_400))
        #expect(cache.derivations == 2)
    }

    @Test func aChangeToTheQueryReDerives() {
        let cache = LibraryShowingCache()
        let rows = pool(3)
        var narrowed = LibraryQuery()
        narrowed.text = "print-1"

        _ = cache.showing(pool: rows, revision: 1, query: LibraryQuery(), selection: [])
        let showing = cache.showing(pool: rows, revision: 1, query: narrowed, selection: [])

        #expect(cache.derivations == 2)
        #expect(showing.visible.map(\.print.filename) == ["print-1.png"])
    }

    /// Two lists of the same size at the same revision -- the timeline and the
    /// trash -- are not the same list.
    @Test func adifferentPoolOfTheSameSizeIsADifferentKey() {
        let cache = LibraryShowingCache()
        let timeline = pool(3)
        let trash = (0 ..< 3).map {
            PrintFixtures.entry("trashed-\($0).png", host: plato, timestamp: UInt64(9_000 + $0))
        }

        _ = cache.showing(pool: timeline, revision: 1, query: LibraryQuery(), selection: [])
        let showing = cache.showing(pool: trash, revision: 1, query: LibraryQuery(), selection: [])

        #expect(cache.derivations == 2)
        #expect(showing.visible.first?.print.filename == "trashed-2.png")
    }

    /// What is selected comes back in the order the grid draws it, whatever
    /// order the set iterates in.
    @Test func theSelectionComesBackInTheOrderTheGridDrawsIt() {
        let cache = LibraryShowingCache()
        let rows = pool(5)
        var query = LibraryQuery()
        query.sort = .oldest

        let showing = cache.showing(pool: rows, revision: 1, query: query,
                                    selection: [rows[3].id, rows[0].id, rows[4].id])

        #expect(showing.selected.map(\.print.filename)
            == ["print-0.png", "print-3.png", "print-4.png"])
        #expect(showing.selected.map(\.id)
            == showing.visible.filter { showing.selected.map(\.id).contains($0.id) }.map(\.id))
    }
}
