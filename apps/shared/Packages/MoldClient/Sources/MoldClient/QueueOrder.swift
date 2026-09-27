import Foundation

/// Where a row has to be told to go.
///
/// Three different indices exist and only one of them is the answer:
///  - the row's place ON SCREEN, which includes held, paused and running rows
///    and, once batches are grouped, is not even flat;
///  - `QueueEntry.position`, which the server assigns over queued AND running
///    rows and which a held row inherits from the next runnable one
///    (`job_registry.rs:49-65`);
///  - the index `PATCH /api/queue/:id` wants, which is into the machine's
///    `state = 'queued'` rows in dispatch order and nothing else
///    (`generation_queue.rs:1815-1824`).
/// Handing the server either of the first two moves the job somewhere nobody
/// asked for, and the PATCH answers 200 either way.
public enum QueueOrder {
    /// The one PATCH for a single row moved to sit after `neighbour` (`nil`
    /// means the front). `nil` for a row that is not `isReorderable`.
    public static func move(
        _ id: String, after neighbour: String?, in entries: [QueueEntry]
    ) -> (id: String, position: Int)? {
        moves([id], after: neighbour, in: entries).first
    }

    /// A whole batch moved as a unit, as the calls to issue IN ORDER.
    ///
    /// One PATCH moves one row (there is no multi-row route), and the server
    /// resolves each one against the queue as it stands THEN: it removes only
    /// the row that call names and re-inserts it
    /// (`generation_queue.rs:1815-1836`). So the plan is computed the same
    /// way -- against a working copy that every planned call is applied to
    /// before the next is planned, each child aimed at sitting behind the one
    /// ahead of it.
    ///
    /// Taking ONE base index from a list with every mover removed up front is
    /// the wrong index space and splits the batch: `[A, c1, c2, N, B]` dropped
    /// after `N` planned `(c1, 2), (c2, 3)`, which the server lands as
    /// `[A, c1, N, c2, B]` -- the children on either side of the row they were
    /// dropped behind. `after: nil` was the one case where the two spaces
    /// coincide, which is why it looked right.
    public static func moves(
        _ ids: [String], after neighbour: String?, in entries: [QueueEntry]
    ) -> [(id: String, position: Int)] {
        // The server's candidate set is `state = 'queued'` alone -- not
        // paused, not held (`generation_queue.rs:1815-1824`).
        var order = entries.filter { $0.state.isReorderable }.map(\.id)
        var anchor = neighbour
        var results: [(id: String, position: Int)] = []
        for id in ids {
            guard let current = order.firstIndex(of: id) else { continue }
            order.remove(at: current)
            let target = position(after: anchor, in: order)
            order.insert(id, at: target)
            results.append((id, target))
            // The next child goes behind THIS one, wherever it landed.
            anchor = id
        }
        return results
    }

    /// `requested_position.min(order.len())` (`generation_queue.rs:1815-1836`):
    /// the neighbour's index in `order`, one past it -- or the front when
    /// there is no neighbour, or none was found among the candidates --
    /// clamped to `order.count` rather than trusted from the caller. `order`
    /// is the candidate list with the moving row ALREADY removed, exactly as
    /// the server has it when it reads the requested position.
    private static func position(after neighbour: String?, in order: [String]) -> Int {
        guard let neighbour, let index = order.firstIndex(of: neighbour) else { return 0 }
        return min(index + 1, order.count)
    }
}
