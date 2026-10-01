import MoldClient
import Testing
@testable import MoldCompanion

@MainActor struct LibraryDragSelectionTests {
    private let host = MoldHost.ID()
    private var ids: [PrintID] { (0..<8).map { PrintID(host: host, filename: "\($0).png") } }

    @Test func sweepSelectsSkippedTilesAndRestoresBaselineOnReversal() throws {
        let ids = ids
        let drag = try #require(LibraryDragSelection(ids: ids, start: ids[2], selection: [ids[7]]))
        #expect(drag.selection(through: ids[5]) == Set(ids[2...5]).union([ids[7]]))
        #expect(drag.selection(through: ids[3]) == [ids[2], ids[3], ids[7]])
        #expect(drag.selection(through: ids[0]) == [ids[0], ids[1], ids[2], ids[7]])
    }

    @Test func sweepFromSelectedTileDeselectsWithoutTogglingRevisitedTiles() throws {
        let ids = ids
        let baseline = Set(ids)
        let drag = try #require(LibraryDragSelection(ids: ids, start: ids[2], selection: baseline))
        #expect(drag.selection(through: ids[5]) == [ids[0], ids[1], ids[6], ids[7]])
        #expect(drag.selection(through: ids[3]) == baseline.subtracting([ids[2], ids[3]]))
        #expect(drag.selection(through: ids[5]) == [ids[0], ids[1], ids[6], ids[7]])
    }

    @Test func unknownStartDoesNotBeginSelection() {
        #expect(LibraryDragSelection(ids: ids, start: PrintID(host: host, filename: "unknown"), selection: []) == nil)
    }
}
