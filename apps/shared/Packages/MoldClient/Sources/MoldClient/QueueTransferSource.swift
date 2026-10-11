import Foundation

/// A failed lookup is never evidence that the source disappeared. Only a
/// definite 404 while recovering a prior destination can establish absence.
public enum QueueTransferSource: Hashable, Sendable {
    case eligible(QueueAuthority)
    case ineligible
    case missing

    public var authority: QueueAuthority? {
        guard case let .eligible(authority) = self else { return nil }
        return authority
    }

    public static func read(jobID: String, instanceID: String, reservedProtocol: Bool,
                            hasPriorAttempt: Bool, backend: any MoldBackend) async throws -> Self {
        do {
            let detail = try await backend.queueJob(id: jobID)
            try Task.checkCancellation()
            guard QueueTransferEligibility.allows(detail.job.state, reservedProtocol: reservedProtocol),
                  let authority = detail.job.transferAuthority(instanceId: instanceID, reservedProtocol: reservedProtocol)
            else { return .ineligible }
            return .eligible(authority)
        } catch let error as MoldClientError where hasPriorAttempt && TransferPlan.isNotFound(error) {
            try Task.checkCancellation()
            return .missing
        }
    }
}
