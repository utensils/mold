import Foundation

/// Everything one held row's retry has to name.
///
/// `GenerationRetryRequest` (`types.rs:11081-11086`). Three of the four
/// fields belong to the CLIENT's batch and one to the server's run, and
/// `retry_queue_job` checks all four before mutating anything
/// (`routes.rs:7617-7650`) -- so a bare job id is never enough and `nil` here
/// means "this row is not a durable batch child", which is a real answer, not
/// a missing one.
public struct QueueAuthority: Codable, Hashable, Sendable {
    public let instanceId, batchId, clientBatchId, jobId: String

    public init(instanceId: String, batchId: String, clientBatchId: String, jobId: String) {
        self.instanceId = instanceId
        self.batchId = batchId
        self.clientBatchId = clientBatchId
        self.jobId = jobId
    }
}

public extension QueueEntry {
    /// Mirrors `QueueJobEntryWire::retry_request` (`types.rs:11038-11047`):
    /// `nil` unless both `batchId` and `clientBatchId` are present.
    func authority(instanceId: String) -> QueueAuthority? {
        guard let batchId, let clientBatchId else { return nil }
        return QueueAuthority(
            instanceId: instanceId, batchId: batchId, clientBatchId: clientBatchId, jobId: id)
    }
}

/// `GET /api/queue/:id` -- the one place a durably admitted job's settings
/// can be read before it dispatches, because the LISTING's projection never
/// selects `request_json` (`routes.rs:6817-6828`).
public struct QueueJobDetail: Codable, Sendable {
    public let job: QueueEntry
}

/// `DELETE /api/queue`. A count, never a list of ids (`routes.rs:7880`).
public struct QueueCancelResult: Codable, Sendable {
    public let cancelled: Int
}
