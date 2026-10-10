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
    @MainActor @Test func selectedPreviewIsReadBeforeOriginalLoadBegins() async {
        let id = PrintID(host: UUID(), filename: "preview.png")
        var viewed = false
        let original: String? = await ViewerReadBoundary.originalAfterPreview(
            displayed: id, current: id, selected: true, trashed: false,
            markViewed: { viewed = true }, load: {
                #expect(viewed, "A displayed preview must be read while original loading is pending")
                await Task.yield()
                return nil
            })
        #expect(original == nil)
        #expect(viewed)
    }
}
