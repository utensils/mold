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
    ///
    /// The candidate list is `entries` filtered to reorderable rows with `id`
    /// itself removed, mirroring the server's own "remove, then re-insert"
    /// (`generation_queue.rs:1815-1863`) -- so the position this computes is
    /// exactly the slot the row will land in, not one off because the row was
    /// still occupying it.
    public static func move(
        _ id: String, after neighbour: String?, in entries: [QueueEntry]
    ) -> (id: String, position: Int)? {
        guard let entry = entries.first(where: { $0.id == id }), entry.state.isReorderable
        else { return nil }
        let candidates = entries.filter { $0.state.isReorderable && $0.id != id }
        return (id, position(after: neighbour, in: candidates))
    }

    /// A whole batch moved as a unit, as the calls to issue IN ORDER.
    ///
    /// One PATCH moves one row (there is no multi-row route), and each call
    /// sees the previous call's result -- so ascending target indices land
    /// the children contiguous: after `(c1, k)` the candidate list has `c1`
    /// at `k`, and `(c2, k+1)` inserts immediately behind it. Ascending is not
    /// a style choice; descending interleaves them. Every id in `ids` is
    /// removed from the candidate list up front, since all of them are about
    /// to move regardless of order.
    public static func moves(
        _ ids: [String], after neighbour: String?, in entries: [QueueEntry]
    ) -> [(id: String, position: Int)] {
        let moving = Set(ids)
        let candidates = entries.filter { $0.state.isReorderable && !moving.contains($0.id) }
        let base = position(after: neighbour, in: candidates)
        var results: [(id: String, position: Int)] = []
        for id in ids {
            guard let entry = entries.first(where: { $0.id == id }), entry.state.isReorderable
            else { continue }
            results.append((id, base + results.count))
        }
        return results
    }

    /// `requested_position.min(order.len())` (`generation_queue.rs:1815-1863`):
    /// the neighbour's index in `candidates`, one past it -- or the front
    /// when there is no neighbour, or none was found among the candidates --
    /// clamped to `candidates.count` rather than trusted from the caller.
    private static func position(after neighbour: String?, in candidates: [QueueEntry]) -> Int {
        guard let neighbour, let index = candidates.firstIndex(where: { $0.id == neighbour })
        else { return 0 }
        return min(index + 1, candidates.count)
    }
}
