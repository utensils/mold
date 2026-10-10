import Foundation
import MoldClient
import Testing

@testable import MoldCompanion

@MainActor
struct LibraryScrollPositionTests {
    @Test func resizeCapturesThePrintBeforeNewGeometryReportsVisibility() {
        let host = MoldHost.ID()
        let before = PrintID(host: host, filename: "before.png")
        let after = PrintID(host: host, filename: "after.png")
        var position = LibraryScrollPosition()
        position.report(before)
        position.prepareReflow()
        position.report(after)
        position.prepareReflow() // Coalesced size changes keep the first anchor.
        #expect(position.takeReflowAnchor() == before)
        #expect(position.takeReflowAnchor() == nil)
        position.prepareReflow()
        position.reset()
        #expect(position.takeReflowAnchor() == nil)
    }

    @Test func viewerReturnRestoresViewportRatherThanOpenedTile() {
        let viewport = LibraryViewport()
        viewport.report(offset: 1234.5)
        viewport.cover()
        viewport.report(offset: 0) // navigation hides or rebuilds the grid
        #expect(viewport.uncover() == 1234.5)
        viewport.report(offset: 1567)
        viewport.cover()
        #expect(viewport.uncover() == 1567)
    }

    @Test func largeGalleryScrollingVisitsOnlyReportedTargetsAndDerivesOnce() throws {
        let host = MoldHost(name: "Fixture", baseURL: URL(string: "http://localhost:7680")!)
        let prints = try (0..<10_000).map { index in
            try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
                "{\"filename\":\"\(index).png\",\"timestamp\":1790000000,\"metadata\":{\"prompt\":\"fixture\",\"seed\":1,\"model\":\"flux-dev:q4\"}}".utf8))
        }
        let entries = prints.map { LibraryEntry(host: host, print: $0) }
        let cache = LibraryGridProjectionCache()
        let query = LibraryQuery()
        for _ in 0..<100 {
            let projection = cache.project(entries: entries, revision: 1, scope: .all, query: query)
            #expect(projection.firstVisible([entries[9999].id, entries[9000].id]) == entries[9000].id)
        }
        #expect(cache.derivations == 1)
        #expect(cache.targetLookups == 200)
        let projection = cache.project(entries: entries, revision: 1, scope: .all, query: query)
        #expect(projection.pages(around: entries[9000].id).map(\.id) == entries[8998...9002].map(\.id))
        #expect(!projection.shouldRecenter(selected: entries[9001].id, anchor: entries[9000].id))
        #expect(projection.shouldRecenter(selected: entries[9002].id, anchor: entries[9000].id))
        #expect(projection.shouldRecenter(selected: entries[9010].id, anchor: entries[9000].id))
        #expect(projection.pages(around: entries[0].id).count == 3)
        #expect(projection.pages(around: entries[9999].id).count == 3)
        #expect(projection.step(1, from: entries[9000].id) == entries[9001].id)
        #expect(projection.step(-1, from: entries[0].id) == nil)
        #expect(projection.entry(entries[9000].id)?.hostID == host.id)
        _ = cache.project(entries: entries, revision: 2, scope: .all, query: query)
        #expect(cache.derivations == 2)
        let refreshed = cache.project(entries: Array(entries.prefix(50)), revision: 3, scope: .all, query: query)
        #expect(refreshed.entry(entries[9000].id) == nil)
        #expect(refreshed.pages(around: entries[9000].id).isEmpty)

        let selected = entries[9001].id
        let removedAnchor = cache.project(entries: entries.filter { $0.id != entries[9000].id },
                                          revision: 4, scope: .all, query: query)
        let repaired = removedAnchor.anchor(for: selected, preferred: entries[9000].id)
        #expect(repaired == selected)
        #expect(removedAnchor.pages(around: repaired).contains { $0.id == selected })
        let reordered = cache.project(entries: [entries[9001]] + entries.filter { $0.id != selected },
                                      revision: 5, scope: .all, query: query)
        let moved = reordered.anchor(for: selected, preferred: entries[9000].id)
        #expect(moved == selected)
        #expect(reordered.pages(around: moved).contains { $0.id == selected })
        #expect(reordered.step(1, from: selected) == entries[0].id)
    }

    @Test func leavingTheViewerDoesNotEraseTheLastVisiblePrint() {
        let host = MoldHost.ID()
        let first = PrintID(host: host, filename: "first.png")
        let later = PrintID(host: host, filename: "later.png")
        var position = LibraryScrollPosition()

        position.report(first)
        position.report(later)
        position.report(nil) // SwiftUI reports no visible grid target while it is covered.
        #expect(position.id == later)

        position.reset() // A different shelf or search should begin at the top.
        #expect(position.id == nil)
    }
}
