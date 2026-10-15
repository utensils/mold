import MoldClient
import MoldClientTesting
import MoldStyle
import SwiftUI
import Testing
@testable import MoldCompanion

@MainActor struct QueuePlaceholderTests {
    @Test func emptyInventoryIsDistinctFromLoadingOrUnavailable() async throws {
        let (queue, hosts, fake) = try await QueueStoreTests.setUp()
        let id = hosts.hosts[0].id
        let entry = try #require(queue.listings[id]?.first)
        #expect(!queue.hasLoadedInputPreviews(for: entry, on: id))
        fake.stub("queueInputs(id:)", returning: [QueueInput]())
        await queue.loadSourceThumbnail(for: entry, on: id)
        #expect(queue.hasLoadedInputPreviews(for: entry, on: id))
        #expect(queue.inputPreviews(for: entry, on: id).isEmpty)
        #expect(!queue.inputLoadFailed(for: entry, on: id))
    }

    @Test func everyPlaceholderKeepsTheThumbnailDimensions() throws {
        for state in MediaPreviewPlaceholder.State.allCases {
            let renderer = ImageRenderer(content: MediaPreviewPlaceholder(state: state).frame(width: 52, height: 52))
            renderer.scale = 1
            let image = try #require(renderer.cgImage)
            #expect(image.width == 52 && image.height == 52)
        }
    }
}
