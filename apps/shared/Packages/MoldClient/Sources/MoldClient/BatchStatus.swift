import Foundation

/// Where a submitted batch stands. The events stream sends this WHOLE
/// snapshot every time rather than a delta, so reconnecting anywhere is safe.
public struct BatchStatus: Codable, Hashable, Sendable {
    public let id: String
    public let clientBatchId: String
    public let instanceId: String?
    public let durable: Bool?
    public let children: [BatchChild]

    /// Nothing left that could still change.
    public var isSettled: Bool { children.allSatisfy { !$0.state.isLive } }

    /// Nothing left that will change WITHOUT A PERSON: every child is either
    /// settled or held. A hold is live on the wire -- Retry can still move
    /// it -- but the machine has parked it until someone decides in the
    /// Queue, so a pane that waits for it waits for ever. Following stops
    /// here; `isSettled` stays the stricter "nothing can ever change".
    public var isAtRest: Bool { children.allSatisfy { !$0.state.isLive || $0.state == .held } }
}

public struct BatchChild: Codable, Hashable, Sendable, Identifiable {
    public let index: Int
    public let jobId: String
    public let state: BatchChildState
    public let error: String?
    public let errorCode: String?
    public let retryable: Bool?
    /// THE ordering token. Not `updatedAtMs` -- see `supersedes`.
    public let revision: UInt64?
    public let updatedAtMs: Int64?
    public let result: BatchResult?

    public var id: String { jobId }

    /// Whether `self` is a later view of this child than `other`.
    ///
    /// `POST /api/queue/{id}/retry` is the one route that moves a child
    /// BACKWARD (held -> accepted). Comparing timestamps drops that update, so
    /// the revision is the authority. A revision of 0 or absent is a
    /// pre-migration row with no authority, and only then does the timestamp
    /// decide.
    public func supersedes(_ other: BatchChild) -> Bool {
        switch (revision ?? 0, other.revision ?? 0) {
        case (0, 0): (updatedAtMs ?? 0) >= (other.updatedAtMs ?? 0)
        case let (mine, theirs): mine >= theirs
        }
    }
}

public enum BatchChildState: String, OpenWireEnum {
    case accepted, paused, cancelling, running, complete, failed, cancelled, held
    case unknown

    public var isLive: Bool {
        switch self {
        case .accepted, .paused, .cancelling, .running, .held: true
        case .complete, .failed, .cancelled, .unknown: false
        }
    }
}

/// `POST /api/generation-batches/status`'s answer -- authoritative state for
/// every batch the caller asked about, plus the ids that named nothing on
/// this machine. A READ despite the route's verb: `spawn_queue_read`
/// (`routes.rs:3396-3401`), a POST only because up to
/// `QueueBatchStatusLimit.identities` ids do not fit in a query string.
public struct BatchStatusListing: Codable, Sendable {
    public let instanceId: String
    public let batches: [BatchStatus]
    public let missing: Missing

    /// The ids this call asked about that named no batch on this machine --
    /// never a failure, since a stale cached id is an ordinary outcome.
    public struct Missing: Codable, Hashable, Sendable {
        public let clientBatchIds: [String]
        public let batchIds: [String]
    }
}

/// `MAX_GENERATION_BATCH_STATUS_IDENTITIES` (`routes.rs:2896`) -- the cap a
/// caller must chunk against; `batchStatuses(batchIds:)` does not chunk for
/// you.
public enum QueueBatchStatusLimit {
    public static let identities = 256
}

public struct BatchResult: Codable, Hashable, Sendable {
    public let filename: String?
    public let seed: UInt64?
    public let generationTimeMs: Int?
    public let gpu: Int?

    /// Spelled out so a result with no batch behind it can be built -- an
    /// automatically chained clip settles into one of these from its own
    /// job's stream. The seed and the timing belong to a chain's STAGES, not
    /// to the print, so they are absent there rather than invented.
    public init(filename: String?, seed: UInt64? = nil,
                generationTimeMs: Int? = nil, gpu: Int? = nil) {
        self.filename = filename
        self.seed = seed
        self.generationTimeMs = generationTimeMs
        self.gpu = gpu
    }
}

/// A running job's live progress, polled from `GET /api/queue/{id}/preview`.
public struct JobProgress: Codable, Hashable, Sendable {
    public let step: Int?
    public let total: Int?
    public let stage: String?
    public let queuePosition: Int?
    /// A base64 PNG of the centred latent, once there is one to show.
    public let previewImage: String?
    public let updatedAtMs: Int64?

    public var previewData: Data? {
        previewImage.flatMap { Data(base64Encoded: $0) }
    }
}
