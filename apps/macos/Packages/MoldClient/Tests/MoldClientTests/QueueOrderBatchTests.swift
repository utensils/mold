import Foundation
import Testing

@testable import MoldClient

// A batch moved as a unit, replayed against the server's own arithmetic.
// `QueueOrderTests` covers the single-row calls and the `after: nil` batch;
// this file is the multi-row plan against a NON-nil neighbour, which is the
// case where the two index spaces stop coinciding.

private func row(_ id: String, _ state: QueueState = .queued) -> QueueEntry {
    QueueEntry(id: id, model: "m", state: state, position: nil, startedAtUnixMs: nil,
               heldReason: nil, error: nil, retryable: nil, durable: nil, batchId: nil,
               clientBatchId: nil, dispatchAttempts: nil, gpu: nil, targetGpu: nil,
               batchIndex: nil, explicitlyPaused: nil, replayed: nil)
}

/// `generation_queue.rs:1815-1836`: ONE `PATCH /api/queue/:id` removes only
/// the row it names and re-inserts it at `requested_position.min(order.len())`.
/// Every later call therefore resolves against the previous call's result --
/// which is the whole reason a multi-row plan cannot be computed once against
/// a list with all the movers taken out.
private func replay(_ calls: [(id: String, position: Int)], from initial: [String]) -> [String] {
    var order = initial
    for call in calls {
        guard let index = order.firstIndex(of: call.id) else { continue }
        order.remove(at: index)
        order.insert(call.id, at: min(call.position, order.count))
    }
    return order
}

private struct Case {
    let name: String
    let queue: [String]
    let moving: [String]
    let after: String?
    let expected: [String]
}

/// **Fails today**: `moves` takes ONE base index from a candidate list with
/// every mover already removed, so `[A, c1, c2, N, B]` dropped after `N`
/// plans `(c1, 2), (c2, 3)` and the server lands `[A, c1, N, c2, B]` -- the
/// children on either side of the row they were dropped behind. Only the
/// `after: nil` case, where the two index spaces coincide, passes.
@Test func aBatchLandsContiguousWhereItWasDropped() {
    let cases: [Case] = [
        Case(name: "one row down across a neighbour", queue: ["A", "B", "C"],
             moving: ["B"], after: "C", expected: ["A", "C", "B"]),
        Case(name: "one row up", queue: ["A", "B", "C"],
             moving: ["C"], after: "A", expected: ["A", "C", "B"]),
        Case(name: "a pair down across one neighbour", queue: ["A", "c1", "c2", "N", "B"],
             moving: ["c1", "c2"], after: "N", expected: ["A", "N", "c1", "c2", "B"]),
        Case(name: "a pair down across several", queue: ["c1", "c2", "A", "B", "C"],
             moving: ["c1", "c2"], after: "C", expected: ["A", "B", "C", "c1", "c2"]),
        Case(name: "a pair up", queue: ["A", "B", "c1", "c2"],
             moving: ["c1", "c2"], after: "A", expected: ["A", "c1", "c2", "B"]),
        Case(name: "a three-child batch down", queue: ["c1", "c2", "c3", "A", "N", "B"],
             moving: ["c1", "c2", "c3"], after: "N", expected: ["A", "N", "c1", "c2", "c3", "B"]),
        Case(name: "to the front", queue: ["A", "c1", "c2", "N", "B"],
             moving: ["c1", "c2"], after: nil, expected: ["c1", "c2", "A", "N", "B"]),
        Case(name: "to the end", queue: ["c1", "c2", "A", "N", "B"],
             moving: ["c1", "c2"], after: "B", expected: ["A", "N", "B", "c1", "c2"]),
    ]
    for test in cases {
        let entries = test.queue.map { row($0) }
        let plan = QueueOrder.moves(test.moving, after: test.after, in: entries)
        #expect(plan.map(\.id) == test.moving, "\(test.name): every mover is a call")
        #expect(replay(plan, from: test.queue) == test.expected, "\(test.name)")
    }
}

/// The candidate space is `state = 'queued'` alone, so a held or running row
/// between two children is not a position the plan may count -- and the rows
/// that cannot move must still be where they were when the calls land.
@Test func aBatchMovePlansOverQueuedRowsOnly() {
    let entries = [
        row("running", .running), row("c1", .queued), row("held", .held),
        row("c2", .queued), row("A", .queued), row("N", .queued),
    ]
    let plan = QueueOrder.moves(["c1", "c2"], after: "N", in: entries)
    // The replay set is the server's own: queued rows, in order.
    #expect(replay(plan, from: ["c1", "c2", "A", "N"]) == ["A", "N", "c1", "c2"])
}

/// A row that cannot be reordered is not a call, and it does not consume a
/// slot the rows behind it were planned into.
@Test func aBatchMoveSkipsRowsThatCannotMove() {
    let entries = [row("A"), row("held", .held), row("c1"), row("N")]
    let plan = QueueOrder.moves(["held", "c1"], after: "N", in: entries)
    #expect(plan.map(\.id) == ["c1"])
    #expect(replay(plan, from: ["A", "c1", "N"]) == ["A", "N", "c1"])
}
