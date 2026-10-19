import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

@MainActor struct LibraryQueryCountCacheTests {
    @Test func progressAndSelectionRedrawsDoNotRefilterTenThousandPrints() {
        let host = UUID()
        let entries = (0..<10_000).map { PrintFixtures.entry("print-\($0).png", host: host) }
        let cache = LibraryQueryCountCache()
        for _ in 0..<100 {
            #expect(cache.count(in: entries, query: LibraryQuery(), revision: 1) == 10_000)
        }
        #expect(cache.derivations == 1)
        #expect(cache.count(in: Array(entries.dropLast()), query: LibraryQuery(), revision: 2) == 9_999)
        #expect(cache.derivations == 2)
    }

    @Test func queryMachineAndReceiptChangesCannotReuseStaleCounts() {
        let first = UUID(), second = UUID()
        let entries = [PrintFixtures.entry("a.png", host: first), PrintFixtures.entry("b.png", host: second)]
        let cache = LibraryQueryCountCache()
        var query = LibraryQuery()
        #expect(cache.count(in: entries, query: query, revision: 1, privacyRevision: 0) == 2)
        query.tokens = [.machine(id: first, name: "First")]
        #expect(cache.count(in: entries, query: query, revision: 1, privacyRevision: 0) == 1)
        #expect(cache.count(in: entries, query: query, revision: 1, privacyRevision: 1) == 1)
        #expect(cache.derivations == 3)
    }
    @Test func liveAndTrashPoolsNeverAliasWithIdenticalFilters() {
        let host = UUID(), cache = LibraryQueryCountCache()
        let live = [PrintFixtures.entry("live.png", host: host)]
        #expect(cache.count(in: live, query: LibraryQuery(), revision: 1) == 1)
        #expect(cache.count(in: [], query: LibraryQuery(), partition: .trash, revision: 1) == 0)
    }

}
