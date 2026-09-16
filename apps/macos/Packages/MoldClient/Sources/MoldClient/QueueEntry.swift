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

    public var startedAt: Date? {
        startedAtUnixMs.map { Date(timeIntervalSince1970: TimeInterval($0) / 1000) }
    }
}

public enum QueueState: String, OpenWireEnum {
    case accepted, running, paused, cancelling, complete, failed, cancelled, held
    case unknown

    /// Still going to happen, or happening.
    public var isLive: Bool {
        switch self {
        case .accepted, .running, .paused, .cancelling, .held: true
        case .complete, .failed, .cancelled, .unknown: false
        }
    }
}

/// `GET /api/queue`.
public struct QueueListing: Codable, Sendable {
    public let entries: [QueueEntry]
    /// Rows the durable store does not know about yet. They are NOT a separate
    /// list to show -- merge them in by id, or a job appears twice or not at
    /// all depending on which store answered first.
    public let liveOnlyEntries: [QueueEntry]?

    public var merged: [QueueEntry] {
        var byID: [String: QueueEntry] = [:]
        for entry in entries { byID[entry.id] = entry }
        for entry in liveOnlyEntries ?? [] { byID[entry.id] = entry }
        return byID.values.sorted { ($0.position ?? .max) < ($1.position ?? .max) }
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
        case .paused: "Paused"
        case .cancelling: "Stopping"
        case .cancelled: "Cancelled"
        case .complete: "Done"
        case .accepted, .unknown:
            switch position {
            case .some(0): "Next up"
            case let .some(n): "#\(n + 1) in line"
            case nil: "Waiting on the host"
            }
        }
    }
}
