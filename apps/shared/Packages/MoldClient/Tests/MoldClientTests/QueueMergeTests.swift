import Foundation
import Testing

@testable import MoldClient

// Merging the two lists `GET /api/queue` answers with, and the order the
// result comes out in.

private func entry(_ id: String, position: Int?, state: QueueState = .queued) -> QueueEntry {
    QueueEntry(id: id, model: "m", state: state, position: position,
               startedAtUnixMs: nil, heldReason: nil, error: nil, retryable: nil,
               durable: nil, batchId: nil, clientBatchId: nil, dispatchAttempts: nil,
               gpu: nil, targetGpu: nil, batchIndex: nil, explicitlyPaused: nil, replayed: nil)
}

/// Determinism, which is what the rows swapping places is ABOUT. This one
/// cannot be made to fail on demand -- a Dictionary's iteration order is
/// unspecified but does not change within one process for one key set, and
/// the hash seed is re-randomised per LAUNCH -- so it is here as the pin, and
/// `theMergedOrderIsTheOrderTheHostSentAmongTies` below is the red.
@Test func anUnchangedQueueMergesToTheSameOrderEveryTime() {
    let entries = [
        entry("running", position: 0, state: .running),
        entry("held-a", position: 1, state: .held),
        entry("held-b", position: 1, state: .held),
        entry("held-c", position: 1, state: .held),
        entry("queued", position: 1),
        entry("nowhere-a", position: nil),
        entry("nowhere-b", position: nil),
    ]
    let expected = QueueListing(entries: entries, liveOnlyEntries: nil).merged.map(\.id)
    for _ in 0..<50 {
        #expect(QueueListing(entries: entries, liveOnlyEntries: nil).merged.map(\.id) == expected)
    }
}

/// **Fails today**: `merged` sorts a Dictionary's `values` with
/// `sorted(by:)`, which is an introsort and not stable, over a comparator
/// that routinely ties. `assign_positions` (`job_registry.rs:57-64`) gives a
/// held row the position of the next row that CAN run, so ties are the
/// design, not a corner case: every held row ties with the queued row behind
/// it, and every row with no position ties with all the others. The
/// intermediate dictionary also throws away the HOST's own order -- `entries`
/// arrives in the durable page's `(created_at, rowid)` traversal order, which
/// IS dispatch order (`routes.rs:7141-7168`). Between two refreshes of an
/// unchanged queue the rows come back differently, `List` identifies by id,
/// and they visibly swap places while nothing happened.
@Test func theMergedOrderIsTheOrderTheHostSentAmongTies() {
    let listing = QueueListing(
        entries: [
            entry("held-a", position: 1, state: .held),
            entry("held-b", position: 1, state: .held),
            entry("queued", position: 1),
        ],
        liveOnlyEntries: nil)
    #expect(listing.merged.map(\.id) == ["held-a", "held-b", "queued"])
}

/// Position still leads: a row the host put later in the line sorts later
/// whatever order the two lists arrived in.
@Test func positionStillDecidesWhereARowSits() {
    let listing = QueueListing(
        entries: [entry("third", position: 2), entry("first", position: 0)],
        liveOnlyEntries: [entry("second", position: 1)])
    #expect(listing.merged.map(\.id) == ["first", "second", "third"])
}

/// A row with no position at all goes last, and the ones that have none keep
/// the order they arrived in rather than shuffling among themselves.
@Test func rowsWithNoPositionGoLastInTheOrderTheyArrived() {
    let listing = QueueListing(
        entries: [entry("no-b", position: nil), entry("placed", position: 3),
                  entry("no-a", position: nil)],
        liveOnlyEntries: nil)
    #expect(listing.merged.map(\.id) == ["placed", "no-b", "no-a"])
}

/// The live view of a row still wins, and it wins IN PLACE -- a job does not
/// jump to the end of the queue because the registry answered about it.
@Test func aLiveOverlayWinsWithoutMovingTheRow() {
    let listing = QueueListing(
        entries: [entry("a", position: 0), entry("b", position: 1), entry("c", position: 2)],
        liveOnlyEntries: [entry("b", position: 1, state: .running), entry("d", position: 3)])
    #expect(listing.merged.map(\.id) == ["a", "b", "c", "d"])
    #expect(listing.merged.first { $0.id == "b" }?.state == .running)
}
