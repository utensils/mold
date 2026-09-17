import Foundation

/// A batch is one atomic admission of up to 64 ordered children. There is no
/// separate "single render" path on the server -- a one-off is a batch of one.
public struct BatchAdmission: Codable, Sendable {
    /// Minted on the device and PERSISTED BEFORE SENDING. This is the
    /// idempotency fence: if the response is lost, the work is recovered by
    /// asking the host about this id, never by submitting again.
    public let clientBatchId: String
    public let requests: [GenerateRequest]

    public init(clientBatchId: String = UUID().uuidString, requests: [GenerateRequest]) {
        self.clientBatchId = clientBatchId
        self.requests = requests
    }
}
