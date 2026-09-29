import MoldClient
import Testing

@testable import MoldCompanion

@MainActor
struct LibraryScrollPositionTests {
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
