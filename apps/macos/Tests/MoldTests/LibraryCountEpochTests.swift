import MoldClient
import Testing

@testable import Mold

/// Library counts follow `countEpoch`, which a sync may hold for a second;
/// every copied print used to re-filter the whole library once per count.
@MainActor
struct LibraryCountEpochTests {
    @Test func aSyncHoldsCountsAndStoppingFlushesThem() {
        let library = LibraryStore(hosts: HostStore(hosts: []))
        library.localSaveRunning = true
        let held = library.countEpoch
        let derivations = library.countCache.derivations
        for _ in 0..<50 {
            library.items = []
            let epoch = library.countEpoch
            #expect(epoch == held)
            _ = library.countCache.count(in: library.items, query: LibraryQuery(),
                                         revision: epoch.rows, privacyRevision: epoch.receipts)
        }
        #expect(library.countCache.derivations == derivations + 1, "one derivation, not one per copied print")
        library.localSaveRunning = false
        #expect(library.countEpoch == library.sourcePrivacyRevision)
        #expect(library.countEpoch != held)
    }

    @Test func outsideASyncCountsAreExact() {
        let library = LibraryStore(hosts: HostStore(hosts: []))
        _ = library.countEpoch
        library.items = []
        #expect(library.countEpoch == library.sourcePrivacyRevision)
    }
}
