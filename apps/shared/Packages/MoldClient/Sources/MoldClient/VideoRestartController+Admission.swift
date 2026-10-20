import Foundation

public extension VideoRestartController {
    func start(destination: VideoRestartTarget, backend: any MoldBackend,
               current: @escaping @MainActor () -> Bool) async {
        guard !isBusy, !isLoading, admission == nil, journalError == nil,
              let source, let sourceBackend, let sourceJobId, let options else { return }
        let mine = revision, sourceCurrent = sourceCurrent
        let isCurrent = { @MainActor in mine == self.revision && sourceCurrent() && current() }
        let permission = options.permission(destinationInstanceId: destination.instanceId)
        guard permission.supported else { error = permission.reason ?? "This video cannot restart on that machine."; return }
        isBusy = true; error = nil
        var persisted: VideoRestartAdmission?
        defer { isBusy = false }
        do {
            try await verify(destination, backend: backend, current: isCurrent)
            guard isCurrent() else { throw changed() }
            let exportRequest = ChainRestartExportRequest(sourceInstanceId: source.instanceId,
                sourceRevision: options.sourceRevision, destinationInstanceId: destination.instanceId)
            let exported = try await sourceBackend.exportChainRestart(id: sourceJobId, request: exportRequest)
            guard isCurrent() else { throw changed() }
            let raw = try exported.validated(for: exportRequest)
            // Recheck immediately before the journal/POST boundary. The target
            // must positively guarantee installation-header enforcement.
            try await verify(destination, backend: backend, current: isCurrent)
            guard isCurrent() else { throw changed() }
            let record = VideoRestartAdmission(operationId: UUID().uuidString.lowercased(),
                destinationHostId: destination.host.id, destinationOrigin: destination.host.baseURL.absoluteString,
                destinationInstanceId: destination.instanceId, destinationMode: destination.host.connectionMode,
                sourceHostId: source.host.id, sourceJobId: sourceJobId,
                sourceInstanceId: source.instanceId, sourceRevision: options.sourceRevision)
            try save(record)
            persisted = record; admission = record
            let accepted = try await backend.createChainJob(requestJSON: raw, operationId: record.operationId,
                                                           destinationInstanceId: destination.instanceId)
            guard accepted.jobId == record.operationId else { throw MoldClientError.malformedResponse }
            if mine == revision { notice = accepted.requestWarnings?.joined(separator: "\n") }
        } catch {
            if let persisted, Self.definitive(error) {
                do {
                    try forget(persisted.operationId)
                    if admission?.operationId == persisted.operationId { admission = nil }
                } catch { self.error = UserFacingError.describe(error); return }
            }
            if mine == revision { self.error = UserFacingError.describe(error) }
        }
    }

    func refresh(destination: VideoRestartTarget, backend: any MoldBackend,
                 current: @escaping @MainActor () -> Bool) async {
        guard var record = admission, record.matches(destination), !isCancelling,
              current(), backend.host == destination.host else { return }
        let mine = revision
        do {
            if record.cancelRequested && !record.cancelAcknowledged {
                record = try await reconcileCancellation(record, destination: destination, backend: backend,
                    current: { mine == self.revision && current() })
            }
            let status = try await backend.status()
            guard mine == revision, current(), backend.host == destination.host,
                  status.instanceId == record.destinationInstanceId else { return }
            let value = try await backend.chainJob(id: record.operationId)
            guard mine == revision, current(), admission?.operationId == record.operationId else { return }
            guard value.id == record.operationId else { throw MoldClientError.malformedResponse }
            detail = value
            if value.state.isTerminal { try forget(record.operationId) }
        } catch {
            guard mine == revision, current() else { return }
            if case let .http(status, _, _) = error as? MoldClientError, status == 404, record.cancelAcknowledged {
                cancelledBeforeAdmission = true
                try? forget(record.operationId)
            } else { self.error = UserFacingError.describe(error) }
        }
    }

}
