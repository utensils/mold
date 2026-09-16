import Foundation
import Testing

@testable import MoldClient

private func listing() throws -> QueueListing {
    try MoldJSON.decoder.decode(QueueListing.self, from: RepoFixtures.fixture("queue.json"))
}

@Test func decodesAQueueCapturedFromALiveHost() throws {
    let queue = try listing()
    #expect(queue.entries.count == 3)
    #expect(queue.entries.allSatisfy { $0.state == .held })
    #expect(queue.entries.first?.heldReason != nil)
}

@Test func liveOnlyRowsMergeByIdRatherThanAppending() {
    func entry(_ id: String, position: Int, state: QueueState) -> QueueEntry {
        QueueEntry(id: id, model: "m", state: state, position: position,
                   startedAtUnixMs: nil, heldReason: nil, error: nil, retryable: nil,
                   durable: nil, batchId: nil, clientBatchId: nil, dispatchAttempts: nil)
    }
    let listing = QueueListing(
        entries: [entry("a", position: 0, state: .accepted)],
        liveOnlyEntries: [entry("a", position: 0, state: .running),
                          entry("b", position: 1, state: .accepted)]
    )
    // "a" appears once, and the live view of it wins.
    #expect(listing.merged.count == 2)
    #expect(listing.merged.first { $0.id == "a" }?.state == .running)
}

@Test func aHeldJobSaysWhatIsActuallyWrong() throws {
    let held = try #require(listing().entries.first)
    // The host's own sentence names a file to restore. Replacing it with
    // "Held" would throw away the only actionable thing on the row.
    #expect(held.waitDescription == held.heldReason)
    #expect(held.waitDescription.count > 10)
}

@Test func queuePositionReadsAsPlaceInLine() {
    func accepted(position: Int?) -> QueueEntry {
        QueueEntry(id: "x", model: nil, state: .accepted, position: position,
                   startedAtUnixMs: nil, heldReason: nil, error: nil, retryable: nil,
                   durable: nil, batchId: nil, clientBatchId: nil, dispatchAttempts: nil)
    }
    #expect(accepted(position: 0).waitDescription == "Next up")
    #expect(accepted(position: 3).waitDescription == "#4 in line")
    // An unknown position says the host is working, not a made-up cause.
    #expect(accepted(position: nil).waitDescription == "Waiting on the host")
}

@Test func anUnknownStateFromANewerHostDoesNotFailTheDecode() throws {
    let json = Data("""
    {"entries":[{"id":"x","state":"some_future_state"}]}
    """.utf8)
    let queue = try MoldJSON.decoder.decode(QueueListing.self, from: json)
    #expect(queue.entries.first?.state == .unknown)
}
