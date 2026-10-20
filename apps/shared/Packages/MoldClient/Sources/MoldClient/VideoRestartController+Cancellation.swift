import Foundation

public extension VideoRestartController {
    func cancel(destination: VideoRestartTarget, backend: any MoldBackend,
                current: @escaping @MainActor () -> Bool) async {
        guard var record = admission, record.matches(destination), !isCancelling,
              current(), backend.host == destination.host,
              detail.map({ ChainJobRecoveryPolicy.canCancel(state: $0.state) }) ?? true else { return }
        do {
            record.cancelRequested = true
            try save(record); admission = record
            await refresh(destination: destination, backend: backend, current: current)
        } catch { self.error = UserFacingError.describe(error) }
    }
}

extension VideoRestartController {
    /// Replaying the same cancellation fence is safe. A create request is
    /// never replayed; a missing acknowledgement remains durable on disk.
    func reconcileCancellation(_ record: VideoRestartAdmission, destination: VideoRestartTarget,
                               backend: any MoldBackend, current: @escaping @MainActor () -> Bool) async throws -> VideoRestartAdmission {
        isCancelling = true
        defer { isCancelling = false }
        try await verify(destination, backend: backend, current: current)
        let capabilities = try await backend.capabilities()
        guard current() else { throw changed() }
        guard capabilities.canControlRestartedChain else {
            throw SequenceDraftError.invalid("Update the destination machine before stopping this video attempt.")
        }
        try await backend.cancelChainOperation(id: record.operationId, operationId: record.operationId,
                                              destinationInstanceId: record.destinationInstanceId)
        var acknowledged = record
        acknowledged.cancelAcknowledged = true
        try save(acknowledged)
        if admission?.operationId == record.operationId { admission = acknowledged }
        return acknowledged
    }
}
