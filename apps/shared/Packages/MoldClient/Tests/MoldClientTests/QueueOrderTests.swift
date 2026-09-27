import Foundation
import Testing

@testable import MoldClient

private func entry(
    _ id: String, state: QueueState, position: Int? = nil, batchId: String? = nil
) -> QueueEntry {
    QueueEntry(id: id, model: "m", state: state, position: position, startedAtUnixMs: nil,
               heldReason: nil, error: nil, retryable: nil, durable: nil, batchId: batchId,
               clientBatchId: nil, dispatchAttempts: nil, gpu: nil, targetGpu: nil,
               batchIndex: nil, explicitlyPaused: nil, replayed: nil)
}

/// **Fails today**: there is no such function, and the obvious implementation
/// -- the row's index on screen -- would send `4`, not `0` (design M6 fact 2).
@Test func aReorderIndexCountsOnlyQueuedRows() {
    let entries = [
        entry("running", state: .running),
        entry("held", state: .held),
        entry("queuedA", state: .queued),
        entry("paused", state: .paused),
        // Its own `position` field is a decoy: the server assigns it over
        // queued AND running rows, which is not the PATCH's candidate set.
        entry("queuedB", state: .queued, position: 10),
    ]
    let move = QueueOrder.move("queuedB", after: nil, in: entries)
    #expect(move?.position == 0)
    #expect(move?.id == "queuedB")
}

@Test func aHeldRowIsNeverReorderable() {
    let entries = [entry("held", state: .held), entry("queued", state: .queued)]
    #expect(QueueOrder.move("held", after: nil, in: entries) == nil)
}

@Test func aRunningRowIsNeverReorderable() {
    let entries = [entry("running", state: .running), entry("queued", state: .queued)]
    #expect(QueueOrder.move("running", after: nil, in: entries) == nil)
}

/// There is no multi-row reorder route, so a batch of four moving to the
/// front is four calls -- and only ASCENDING target order lands them
/// contiguous, because each call sees the previous call's result.
@Test func aBatchMovesAsAscendingCallsSoItsChildrenLandTogether() {
    let entries = [
        entry("o1", state: .queued, batchId: nil), entry("c1", state: .queued, batchId: "batch"),
        entry("o2", state: .queued, batchId: nil), entry("c2", state: .queued, batchId: "batch"),
        entry("o3", state: .queued, batchId: nil), entry("c3", state: .queued, batchId: "batch"),
        entry("o4", state: .queued, batchId: nil), entry("c4", state: .queued, batchId: "batch"),
    ]
    let moves = QueueOrder.moves(["c1", "c2", "c3", "c4"], after: nil, in: entries)
    let pairs = moves.map { (id: $0.id, position: $0.position) }
    #expect(pairs.map(\.id) == ["c1", "c2", "c3", "c4"])
    #expect(pairs.map(\.position) == [0, 1, 2, 3])

    // Replay against a model of the server's own insert (`generation_queue.rs
    // :1815-1863`): remove the row, then re-insert clamped to the current
    // length. Applying the calls in ASCENDING order lands the four children
    // contiguous at the front.
    func replay(_ calls: [(id: String, position: Int)], startingFrom initial: [String]) -> [String] {
        var working = initial
        for call in calls {
            if let index = working.firstIndex(of: call.id) { working.remove(at: index) }
            working.insert(call.id, at: min(call.position, working.count))
        }
        return working
    }
    let initial = entries.map(\.id)
    let ascending = replay(pairs, startingFrom: initial)
    #expect(Array(ascending.prefix(4)) == ["c1", "c2", "c3", "c4"])

    // The negative: the SAME target positions applied DESCENDING interleave
    // the children instead of landing them together -- ascending is not a
    // style choice.
    let descending = replay(pairs.reversed(), startingFrom: initial)
    #expect(Array(descending.prefix(4)) != ["c1", "c2", "c3", "c4"])
}

/// Mirrors `requested_position.min(order.len())`: moving a row to sit after
/// the LAST candidate clamps to the candidate count, never past it.
@Test func aPositionPastTheEndClamps() {
    let entries = [
        entry("q1", state: .queued), entry("q2", state: .queued), entry("q3", state: .queued),
    ]
    let move = QueueOrder.move("q1", after: "q3", in: entries)
    #expect(move?.position == 2)
}
