import Foundation
import MoldClient
import Testing
@testable import MoldCompanion

struct ViewerReadBoundaryTests {
    @Test func viewerReadBoundaryFencesNeighborsHostsAndCancelledLoads() {
        let one = PrintID(host: UUID(), filename: "same.png")
        let two = PrintID(host: UUID(), filename: "same.png")
        #expect(!ViewerReadBoundary.allows(displayed: nil, current: one, selected: true, trashed: false))
        #expect(!ViewerReadBoundary.allows(displayed: one, current: one, selected: false, trashed: false))
        #expect(ViewerReadBoundary.allows(displayed: one, current: one, selected: true, trashed: false))
        #expect(!ViewerReadBoundary.allows(displayed: one, current: two, selected: true, trashed: false))
        #expect(!ViewerReadBoundary.allows(displayed: one, current: one, selected: true, trashed: true))
        #expect(!ViewerReadBoundary.allows(displayed: one, current: one, selected: true, trashed: false, cancelled: true))
    }
    @Test func preloadedNeighborBecomingVisibleRechecksWhenItsImageArrives() {
        let id = PrintID(host: UUID(), filename: "neighbor.png")
        let preloading = ViewerReadBoundary(displayed: nil, current: id, selected: false, trashed: false)
        let selectedBeforeLoad = ViewerReadBoundary(displayed: nil, current: id, selected: true, trashed: false)
        let displayedAfterSelection = ViewerReadBoundary(displayed: id, current: id, selected: true, trashed: false)
        #expect(!preloading.shouldMarkViewed)
        #expect(!selectedBeforeLoad.shouldMarkViewed)
        #expect(displayedAfterSelection != selectedBeforeLoad)
        #expect(displayedAfterSelection.shouldMarkViewed)
        let swipedAwayBeforeLoad = ViewerReadBoundary(displayed: id, current: id, selected: false, trashed: false)
        #expect(!swipedAwayBeforeLoad.shouldMarkViewed)
    }

    @Test func selectedPreviewIsReadWhileOriginalLoadIsPending() {
        let id = PrintID(host: UUID(), filename: "preview.png")
        let preview = ViewerReadBoundary(displayed: id, current: id, selected: true, trashed: false)
        #expect(preview.shouldMarkViewed, "A displayed preview does not wait for the original download")
    }
}
