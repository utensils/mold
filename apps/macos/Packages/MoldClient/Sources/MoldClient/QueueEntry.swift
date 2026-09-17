import Foundation

/// A job on a host's queue.
public struct QueueEntry: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let model: String?
    public let state: QueueState
    public let position: Int?
    public let startedAtUnixMs: Int?
    /// Why the host parked this job. It names something actionable -- a file
    /// to restore, a licence to accept -- and is shown as written.
    public let heldReason: String?
    public let error: String?
    public let retryable: Bool?
    public let durable: Bool?
    public let batchId: String?
    public let clientBatchId: String?
    public let dispatchAttempts: Int?
    /// GPU ordinal actually running this row. Absent on every row that is not
    /// running (`types.rs:4402-4403`) -- never `null`.
    public let gpu: Int?
    /// The lane a queued row prefers. Absent means Auto, which is the
    /// ordinary case and is not a missing answer (`types.rs:4405-4406`).
    public let targetGpu: Int?
    /// 1-based position within its batch. This is what a child row shows;
    /// `position` is about the machine's queue, not about the batch.
    public let batchIndex: Int?
    /// For a `paused` row: whether SOMEBODY paused this one, as opposed to
    /// the restart sweep parking the whole queue (`types.rs:4452-4460`).
    /// Absent means the host does not distinguish them, which is true of
    /// every server built before per-job pause -- so absence reads as
    /// "paused", never as "paused by someone".
    public let explicitlyPaused: Bool?
    /// Whether the row was resumed from the journal rather than submitted by
    /// a live client.
    public let replayed: Bool?

    public var startedAt: Date? {
        startedAtUnixMs.map { Date(timeIntervalSince1970: TimeInterval($0) / 1000) }
    }
}

public enum QueueState: String, OpenWireEnum {
    /// Admitted and waiting. THE ordinary state of a queue, and the one this
    /// enum could not spell until M6: `accepted` is
    /// `GenerationBatchChildState`'s word for the same idea on a DIFFERENT
    /// endpoint, and having both here meant `"queued"` fell to `.unknown`,
    /// `isLive` answered false, and no row could be cancelled.
    case queued
    case running, paused, cancelling, complete, failed, cancelled, held
    case unknown

    /// Still going to happen, or happening.
    public var isLive: Bool {
        switch self {
        case .queued, .running, .paused, .cancelling, .held: true
        case .complete, .failed, .cancelled, .unknown: false
        }
    }

    /// Whether a machine could still reorder this row.
    /// `PATCH /api/queue/:id` refuses anything else (`routes.rs:7322-7326`),
    /// and the reorder candidate set is `state = 'queued'` alone
    /// (`generation_queue.rs:1815-1824`) -- not paused, not held.
    public var isReorderable: Bool { self == .queued }
}

/// `GET /api/queue`.
public struct QueueListing: Codable, Sendable {
    public let entries: [QueueEntry]
    /// Rows the durable store does not know about yet. They are NOT a separate
    /// list to show -- merge them in by id, or a job appears twice or not at
    /// all depending on which store answered first.
    public let liveOnlyEntries: [QueueEntry]?

    /// The two lists as one, in an order that does not change unless the
    /// queue does.
    ///
    /// `entries` arrives in the durable page's own `(created_at, rowid)`
    /// traversal order -- which IS dispatch order (`routes.rs:7141-7168`) --
    /// and a live row overlays the durable one IN PLACE rather than moving it.
    ///
    /// The comparator has to be TOTAL, because ties are the design rather
    /// than a corner case: `assign_positions` (`job_registry.rs:57-64`) gives
    /// a held row the position of the next row that can RUN, so every held
    /// row ties with the queued one behind it, and every row with no position
    /// ties with all the others. Sorting a Dictionary's values -- unspecified
    /// order, re-randomised per launch -- with a non-stable `sorted(by:)`
    /// meant two refreshes of an unchanged queue came back differently, and
    /// since `List` identifies by id the rows visibly swapped places while
    /// nothing had happened. Position, then the host's own index.
    public var merged: [QueueEntry] {
        var ordered: [QueueEntry] = []
        var indexByID: [String: Int] = [:]
        for entry in entries + (liveOnlyEntries ?? []) {
            if let index = indexByID[entry.id] {
                ordered[index] = entry
            } else {
                indexByID[entry.id] = ordered.count
                ordered.append(entry)
            }
        }
        return ordered.enumerated()
            .sorted { ($0.element.position ?? .max, $0.offset)
                      < ($1.element.position ?? .max, $1.offset) }
            .map(\.element)
    }
}

public extension QueueEntry {
    /// What to say about where this job stands.
    ///
    /// Mirrors mold's own queue vocabulary: a blocked job says what is wrong
    /// ONLY when that is something a person can act on, and an unknown reason
    /// says the host is working rather than inventing a cause.
    var waitDescription: String {
        switch state {
        case .running: "Rendering"
        case .held: heldReason ?? "Waiting on the host"
        case .failed: error ?? "Failed"
        case .paused: explicitlyPaused == true ? "Paused" : "Paused after restart"
        case .cancelling: "Stopping"
        case .cancelled: "Cancelled"
        case .complete: "Done"
        case .queued, .unknown:
            switch position {
            case .some(0): "Next up"
            case let .some(n): "#\(n + 1) in line"
            case nil: "Waiting on the host"
            }
        }
    }
}
