/// Worker dispatch is the authoritative Rendering boundary. A source with
/// the reservation protocol can freeze waiting, paused and Held work safely.
public enum QueueTransferEligibility {
    public static func allows(_ state: QueueState, reservedProtocol: Bool) -> Bool {
        state == .held || (reservedProtocol && (state == .queued || state == .paused))
    }
}

public extension QueueEntry {
    func transferAuthority(instanceId: String, reservedProtocol: Bool) -> QueueAuthority? {
        if let authority = authority(instanceId: instanceId) { return authority }
        guard reservedProtocol, durable == true else { return nil }
        return QueueAuthority(instanceId: instanceId, batchId: "", clientBatchId: "", jobId: id)
    }
}

/// The destination binding survives source restart; retry only that same
/// destination before attempting to send elsewhere.
public struct QueueTransferReservation: Codable, Sendable {
    public let transferId: String
    public let destinationTransferIdentity: String
}

public struct QueueTransferReservationRequest: Codable, Sendable {
    public let instanceId, batchId, clientBatchId, jobId, transferId, destinationTransferIdentity: String
    public var abortReceipt: String? = nil
    public init(authority: QueueAuthority, transferId: String, destinationTransferIdentity: String) {
        instanceId = authority.instanceId; batchId = authority.batchId
        clientBatchId = authority.clientBatchId; jobId = authority.jobId
        self.transferId = transferId; self.destinationTransferIdentity = destinationTransferIdentity
    }
}

public struct QueueTransferAbortRequest: Codable, Sendable {
    public let transferId: String
    public let destinationTransferIdentity: String
    public init(transferId: String, destinationTransferIdentity: String) { self.transferId = transferId; self.destinationTransferIdentity = destinationTransferIdentity }
}
public struct QueueTransferAbortResult: Codable, Sendable {
    public let transferId: String
    public let destinationTransferIdentity: String
    public let abortReceipt: String?
}
