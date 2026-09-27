import Foundation
import Testing

@testable import MoldClient

private func entry(
    _ id: String, state: QueueState = .queued, batchId: String? = nil,
    clientBatchId: String? = nil, batchIndex: Int? = nil
) -> QueueEntry {
    QueueEntry(id: id, model: "m", state: state, position: nil, startedAtUnixMs: nil,
               heldReason: nil, error: nil, retryable: nil, durable: nil, batchId: batchId,
               clientBatchId: clientBatchId, dispatchAttempts: nil, gpu: nil, targetGpu: nil,
               batchIndex: batchIndex, explicitlyPaused: nil, replayed: nil)
}

private func child(
    _ jobId: String, index: Int, revision: UInt64?, updatedAtMs: Int64?,
    state: BatchChildState = .accepted
) -> BatchChild {
    BatchChild(index: index, jobId: jobId, state: state, error: nil, errorCode: nil,
               retryable: nil, revision: revision, updatedAtMs: updatedAtMs, result: nil)
}

@Test func aBatchIsOneGroupAndASingletonIsAPlainRow() {
    let entries = [
        entry("standalone"),
        entry("b1", batchId: "batch"), entry("b2", batchId: "batch"),
    ]
    let groups = QueueGroup.build(entries, children: [:])
    #expect(groups.count == 2)
    #expect(groups[0].isExpandable == false)
    #expect(groups[0].id == "standalone")
    #expect(groups[1].isExpandable)
    #expect(groups[1].rows.map(\.id) == ["b1", "b2"])
}

/// A group sits where its FIRST row sat -- a later member of a batch showing
/// up further down the listing (interleaved with an unrelated row) must not
/// drag the whole group down to there.
@Test func groupingNeverReordersTheMachinesQueue() {
    let entries = [
        entry("a"), entry("b1", batchId: "batch", batchIndex: 1), entry("c"),
        entry("b2", batchId: "batch", batchIndex: 2),
    ]
    let groups = QueueGroup.build(entries, children: [:])
    #expect(groups.map(\.id) == ["a", "batch", "c"])
}

/// Pins the master plan's own correction: a retry moves a child BACKWARD, and
/// same-millisecond commits are routine, so `revision` -- not the timestamp
/// -- decides which of two views of one job id is trusted for order.
@Test func aRetriedChildIsTheNewerViewEvenAMillisecondEarlier() {
    let entries = [
        entry("r1", batchId: "batch", batchIndex: 2), entry("r2", batchId: "batch", batchIndex: 1),
    ]
    let children: [String: [BatchChild]] = [
        "batch": [
            // Stale: higher revision-less timestamp, wrong index.
            child("r1", index: 5, revision: 4, updatedAtMs: 9000),
            // The retry's own newer view -- lower revision timestamp, correct index.
            child("r1", index: 0, revision: 5, updatedAtMs: 1000),
            child("r2", index: 1, revision: 1, updatedAtMs: 500),
        ],
    ]
    let groups = QueueGroup.build(entries, children: children)
    #expect(groups[0].rows.map(\.id) == ["r1", "r2"])
}

/// Two views of the same pre-migration child both carry `revision == 0`, so
/// `supersedes` falls back to comparing `updatedAtMs` alone.
@Test func twoPreMigrationChildrenFallBackToTheirTimestamps() {
    let entries = [
        entry("r1", batchId: "batch", batchIndex: 5), entry("r2", batchId: "batch", batchIndex: 1),
    ]
    let children: [String: [BatchChild]] = [
        "batch": [
            child("r1", index: 5, revision: 0, updatedAtMs: 1000),
            // The later timestamp wins and carries the corrected index.
            child("r1", index: 0, revision: 0, updatedAtMs: 2000),
            child("r2", index: 1, revision: 0, updatedAtMs: 1500),
        ],
    ]
    let groups = QueueGroup.build(entries, children: children)
    #expect(groups[0].rows.map(\.id) == ["r1", "r2"])
}
