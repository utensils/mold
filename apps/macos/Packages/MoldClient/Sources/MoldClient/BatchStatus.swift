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

public struct BatchResult: Codable, Hashable, Sendable {
    public let filename: String?
    public let seed: UInt64?
    public let generationTimeMs: Int?
    public let gpu: Int?
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
