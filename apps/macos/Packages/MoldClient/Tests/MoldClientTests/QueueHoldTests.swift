import Foundation
import Testing

@testable import MoldClient

private func heldEntry(
    model: String? = "flux2-dev:q8", heldReason: String? = "The host said this.",
    error: String? = nil, retryable: Bool? = true
) -> QueueEntry {
    QueueEntry(id: "j", model: model, state: .held, position: nil, startedAtUnixMs: nil,
               heldReason: heldReason, error: error, retryable: retryable, durable: nil,
               batchId: "batch", clientBatchId: "client", dispatchAttempts: nil, gpu: nil,
               targetGpu: nil, batchIndex: nil, explicitlyPaused: nil, replayed: nil)
}

private func child(errorCode: String?, retryable: Bool? = nil) -> BatchChild {
    BatchChild(index: 0, jobId: "j", state: .held, error: nil, errorCode: errorCode,
               retryable: retryable, revision: 1, updatedAtMs: nil, result: nil)
}

/// The typed code names two things, and the answer takes the model from the
/// ROW's own field -- the sentence is never parsed for it.
@Test func aMissingModelHoldNamesTheModelFromTheRow() {
    let entry = heldEntry(
        model: "flux2-dev:q8", heldReason: "That checkpoint isn't installed on this machine.")
    let hold = QueueHold.resolve(entry: entry, child: child(errorCode: "MODEL_NOT_FOUND"))
    #expect(hold == .missingModel(
        "flux2-dev:q8", sentence: "That checkpoint isn't installed on this machine."))
}

@Test func anUntypedHoldIsTheMachinesOwnSentenceAndARetry() throws {
    let listing = try MoldJSON.decoder.decode(
        QueueListing.self, from: RepoFixtures.fixture("queue.json"))
    let row = try #require(listing.entries.first)
    let hold = QueueHold.resolve(entry: row, child: nil)
    #expect(hold == .prose(row.heldReason ?? "", retryable: true))
}

@Test func anUnretryableHoldOffersNoRetry() {
    let entry = heldEntry(retryable: false)
    let hold = QueueHold.resolve(entry: entry, child: nil)
    #expect(hold == .prose("The host said this.", retryable: false))
}

/// A row with no batch status answer at all still has the machine's own
/// sentence -- the state this pane has shipped in since M1.
@Test func aHostThatAnsweredNoBatchStatusStillShowsTheRowsSentence() {
    let entry = heldEntry(heldReason: "GPU ran out of memory.", retryable: true)
    let hold = QueueHold.resolve(entry: entry, child: nil)
    #expect(hold == .prose("GPU ran out of memory.", retryable: true))
}

@Test func aLiveRowHasNoHoldAtAll() {
    let entry = heldEntry()
    let queued = QueueEntry(
        id: entry.id, model: entry.model, state: .queued, position: nil, startedAtUnixMs: nil,
        heldReason: entry.heldReason, error: entry.error, retryable: entry.retryable, durable: nil,
        batchId: entry.batchId, clientBatchId: entry.clientBatchId, dispatchAttempts: nil,
        gpu: nil, targetGpu: nil, batchIndex: nil, explicitlyPaused: nil, replayed: nil)
    #expect(QueueHold.resolve(entry: queued, child: nil) == nil)
}
