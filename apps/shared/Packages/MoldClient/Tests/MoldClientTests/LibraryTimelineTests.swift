import Foundation
import Testing

@testable import MoldClient

/// A patched timeline must be exactly what a full merge would produce, and
/// must refuse -- untouched -- every change a merge could answer differently.
struct LibraryTimelineTests {
    let here = UUID()
    let there = UUID()

    func entry(_ name: String, on host: UUID, at timestamp: UInt64, favorite: Bool? = nil,
               bytes: Int? = 100) -> LibraryEntry {
        PrintFixtures.entry(name, host: host, hostName: host == here ? "This Mac" : "workstation",
                            timestamp: timestamp, favorite: favorite, bytes: bytes)
    }

    func rebuilt(_ entries: [LibraryEntry]) -> LibraryTimeline {
        var timeline = LibraryTimeline()
        timeline.rebuild(entries, localHost: here)
        return timeline
    }

    var library: [LibraryEntry] {
        [entry("a", on: here, at: 30), entry("b", on: here, at: 20), entry("b", on: there, at: 20),
         entry("c", on: there, at: 10)]
    }

    @Test func anOrganizationEditPatchesInPlace() {
        var timeline = rebuilt(library)
        let starred = entry("c", on: there, at: 10, favorite: true)
        let patched1 = timeline.patch(upserted: [starred], removed: [])
        #expect(patched1)
        var expected = library
        expected[3] = starred
        #expect(timeline.tiles == rebuilt(expected).tiles)
        #expect(timeline.tile(containing: starred.id)?.isFavorite == true)
    }

    @Test func editingAMergedCopyKeepsItsTile() {
        var timeline = rebuilt(library)
        let copy = entry("b", on: there, at: 20, favorite: true)
        let patched2 = timeline.patch(upserted: [copy], removed: [])
        #expect(patched2)
        let tile = timeline.tile(containing: copy.id)
        #expect(tile?.hostID == here)
        #expect(tile?.isFavorite == true)
    }

    @Test func aNewUnrelatedPrintIsInsertedInOrder() {
        var timeline = rebuilt(library)
        let fresh = entry("d", on: there, at: 25)
        let patched3 = timeline.patch(upserted: [fresh], removed: [])
        #expect(patched3)
        #expect(timeline.tiles == rebuilt(library + [fresh]).tiles)
        #expect(timeline.tiles.map(\.print.filename) == ["a", "d", "b", "c"])
        #expect(timeline.tile(containing: entry("c", on: there, at: 10).id)?.print.filename == "c")
    }

    @Test func aSoleCopyIsRemovedInPlace() {
        var timeline = rebuilt(library)
        let gone = entry("a", on: here, at: 30)
        let patched4 = timeline.patch(upserted: [], removed: [gone.id])
        #expect(patched4)
        #expect(timeline.tiles == rebuilt(Array(library.dropFirst())).tiles)
        #expect(timeline.tile(containing: gone.id) == nil)
        #expect(timeline.copy(entry("b", on: there, at: 20).id)?.hostID == there)
    }

    @Test func changesAMergeCouldAnswerDifferentlyAreRefused() {
        let original = rebuilt(library)
        var timeline = original
        // A copy of an existing print: it would join a tile.
        let patched5 = timeline.patch(upserted: [entry("a", on: there, at: 30)], removed: [])
        #expect(!patched5)
        // Removing one copy of a merged print can change the lead.
        let patched6 = timeline.patch(upserted: [], removed: [entry("b", on: here, at: 20).id])
        #expect(!patched6)
        // A re-dated print moves.
        let patched7 = timeline.patch(upserted: [entry("c", on: there, at: 99)], removed: [])
        #expect(!patched7)
        // A sync link appeared.
        let link = [entry("c", on: there, at: 10).id: entry("a", on: here, at: 30).id]
        let patched8 = timeline.patch(upserted: [], removed: [], links: link)
        #expect(!patched8)
        #expect(timeline.tiles == original.tiles)
    }

    @Test func aTieIsLeftToTheFullMerge() {
        var timeline = rebuilt(library)
        let tied = timeline.patch(upserted: [entry("d", on: there, at: 20)], removed: [])
        #expect(!tied)
        #expect(timeline.tiles == rebuilt(library).tiles)
    }

    @MainActor @Test func theShowingCacheNamesEachDerivation() {
        let cache = LibraryShowingCache()
        let first = cache.showing(pool: library, revision: 1, query: LibraryQuery(), selection: [])
        let again = cache.showing(pool: library, revision: 1, query: LibraryQuery(), selection: [])
        let next = cache.showing(pool: library, revision: 2, query: LibraryQuery(), selection: [])
        #expect(first.derivation == again.derivation)
        #expect(next.derivation != first.derivation)
    }

    @MainActor @Test func layoutIsKeyedOnTheSectionsRevision() {
        let sections = LibraryGrouping.ungrouped(library)
        let layout = JustifiedLibraryLayout()
        _ = layout.resolve(sections, revision: 1, width: 800, targetHeight: 180)
        _ = layout.resolve(sections, revision: 1, width: 800, targetHeight: 180)
        #expect(layout.layouts == 1)
        _ = layout.resolve(sections, revision: 2, width: 800, targetHeight: 180)
        #expect(layout.layouts == 2)
        _ = layout.resolve(sections, revision: 2, width: 600, targetHeight: 180)
        #expect(layout.layouts == 3)
    }
}
