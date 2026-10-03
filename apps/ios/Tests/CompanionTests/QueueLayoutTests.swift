import Foundation
import Testing
@testable import MoldCompanion

struct QueueLayoutTests {
    @Test func wideQueueKeepsWordsNearTheirImages() {
        #expect(QueueLayout.readableWidth <= 900)
        #expect(QueueLayout.readableWidth >= 600)
    }
}
