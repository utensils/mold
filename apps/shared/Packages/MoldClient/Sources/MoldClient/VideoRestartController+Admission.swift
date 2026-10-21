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
        defer {
            isBusy = false
            if mine == revision { progress = nil }
        }
        do {
            try await verify(destination, backend: backend, current: isCurrent)
            guard isCurrent() else { throw changed() }
            let exportRequest = ChainRestartExportRequest(sourceInstanceId: source.instanceId,
                sourceRevision: options.sourceRevision, destinationInstanceId: destination.instanceId)
            let prepared: (raw: String, proof: String?, prefix: VideoSourceUploadRef?)
            if protocolKind == .videoJob {
                prepared = try await prepareParentRestart(source: source, sourceBackend: sourceBackend, jobId: sourceJobId,
                    destination: destination, backend: backend, exportRequest: exportRequest, current: isCurrent)
            } else {
                let exported = try await sourceBackend.exportChainRestart(id: sourceJobId, request: exportRequest)
                guard isCurrent() else { throw changed() }
                prepared = (try exported.validated(for: exportRequest), nil, nil)
            }
            // Recheck immediately before the journal/POST boundary. The target
            // must positively guarantee installation-header enforcement.
            try await verify(destination, backend: backend, current: isCurrent)
            guard isCurrent() else { throw changed() }
            let record = VideoRestartAdmission(operationId: UUID().uuidString.lowercased(),
                destinationHostId: destination.host.id, destinationOrigin: destination.host.baseURL.absoluteString,
                destinationInstanceId: destination.instanceId, destinationMode: destination.host.connectionMode,
                sourceHostId: source.host.id, sourceJobId: sourceJobId,
                sourceInstanceId: source.instanceId, sourceRevision: options.sourceRevision, protocolKind: protocolKind == .videoJob ? .videoJob : nil)
            try save(record)
            persisted = record; admission = record
            if protocolKind == .videoJob {
                let accepted = try await backend.createVideoJob(requestJSON: prepared.raw,
                    validationRevision: prepared.proof ?? "", operationId: record.operationId,
                    destinationInstanceId: destination.instanceId, sourceUpload: prepared.prefix)
                guard accepted.id == record.operationId else { throw MoldClientError.malformedResponse }
                if mine == revision, current() {
                    parentDetail = accepted
                    try acknowledgeKnownParent(record.operationId, destination: destination, settled: accepted.isSettled)
                    notice = "New attempt queued. The original remains on its machine."
                }
            } else {
                let accepted = try await backend.createChainJob(requestJSON: prepared.raw, operationId: record.operationId,
                                                               destinationInstanceId: destination.instanceId)
                guard accepted.jobId == record.operationId else { throw MoldClientError.malformedResponse }
                if mine == revision { notice = accepted.requestWarnings?.joined(separator: "\n") }
            }
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
            if (record.protocolKind ?? .chain) == .videoJob {
                let targetBackend: any MoldBackend = if let http = backend as? HTTPBackend {
                    http.pinningVideoDestination(record.destinationInstanceId)
                } else { backend }
                let value = try await targetBackend.videoJob(id: record.operationId)
                guard mine == revision, current(), admission?.operationId == record.operationId else { return }
                guard value.id == record.operationId else { throw MoldClientError.malformedResponse }
                parentDetail = value
                try acknowledgeKnownParent(record.operationId, destination: destination, settled: value.isSettled)
            } else {
                let value = try await backend.chainJob(id: record.operationId)
                guard mine == revision, current(), admission?.operationId == record.operationId else { return }
                guard value.id == record.operationId else { throw MoldClientError.malformedResponse }
                detail = value
                if value.state.isTerminal { try forget(record.operationId) }
            }
        } catch {
            guard mine == revision, current() else { return }
            if case let .http(status, code, _) = error as? MoldClientError,
               (record.protocolKind == .videoJob ? status == 410 && code == "VIDEO_OPERATION_CANCELLED" : status == 404 && record.cancelAcknowledged) {
                cancelledBeforeAdmission = true
                try? forget(record.operationId)
            } else { self.error = UserFacingError.describe(error) }
        }
    }

    /// Acknowledged parents are discoverable from their durable Queue. Keep
    /// only uncertainty here; an unacknowledged Stop must remain replayable.
    private func acknowledgeKnownParent(_ id: String, destination: VideoRestartTarget, settled: Bool) throws {
        guard let saved = pending.first(where: { $0.operationId == id }),
              saved.matches(destination), saved.protocolKind == .videoJob,
              !saved.cancelRequested || (saved.cancelAcknowledged && settled) else { return }
        try forget(id)
    }

}
