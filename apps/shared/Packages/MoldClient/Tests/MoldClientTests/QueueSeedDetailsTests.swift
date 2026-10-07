import Foundation
import Testing
@testable import MoldClient

struct QueueSeedDetailsTests {
    @Test func queuedRandomPlaceholderIsNotAFixedZero() throws {
        let entry = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"job","state":"queued","seed_pinned":false,"metadata":{"seed":0}}"#.utf8))
        let seed = PrintDetails.groups(for: entry).flatMap(\.rows).first { $0.label == "Seed" }
        #expect(seed?.value == "Random")
    }
    @Test func explicitFixedZeroRemainsZero() throws {
        let entry = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"job","state":"queued","seed_pinned":true,"metadata":{"seed":0}}"#.utf8))
        #expect(PrintDetails.groups(for: entry).flatMap(\.rows).first { $0.label == "Seed" }?.value == "0")
    }
    @Test func freshNativeRequestsAreRandomAndFixedZeroIsExplicit() {
        var draft = RenderDraft()
        #expect(RenderRequest.one(draft, model: "flux-dev:q4").seed == nil)
        draft.seed = 0
        #expect(RenderRequest.one(draft, model: "flux-dev:q4").seed == nil)
        draft.locksSeed = true
        #expect(RenderRequest.one(draft, model: "flux-dev:q4").seed == 0)
    }
}
