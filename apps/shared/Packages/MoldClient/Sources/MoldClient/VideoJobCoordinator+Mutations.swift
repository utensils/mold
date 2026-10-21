import Foundation

public extension VideoJobCoordinator {
    func cancel(_ job: ObservedVideoJob, backend: any MoldBackend,
                current: @escaping @MainActor () -> Bool) async {
        let backend = scoped(backend, owner: job.owner)
        guard !job.cancelledBeforeAdmission, job.detail?.canCancel ?? true,
              !mutating.contains(job.id), current() else { return }
        mutating.insert(job.id); readTokens[job.id] = UUID()
        var acknowledged = false
        do {
            try await verify(job.owner, backend: backend, current: current)
            var record = pending.first { $0.operationId == job.operationId && $0.matches(job.owner) }
                ?? .init(operationId: job.operationId, owner: job.owner)
            record.cancelRequested = true
            try save(record)
            try await backend.cancelVideoJob(id: job.operationId, destinationInstanceId: job.owner.instanceId)
            record.cancelAcknowledged = true
            try save(record); acknowledged = true
        } catch { if current() { message(UserFacingError.describe(error), id: job.operationId, owner: job.owner) } }
        mutating.remove(job.id)
        if acknowledged { await refresh(job.operationId, owner: job.owner, backend: backend, current: current) }
    }

    func resume(_ job: ObservedVideoJob, backend: any MoldBackend,
                current: @escaping @MainActor () -> Bool) async {
        let backend = scoped(backend, owner: job.owner)
        guard job.detail?.canResume == true, !mutating.contains(job.id) else { return }
        mutating.insert(job.id); readTokens[job.id] = UUID()
        defer { mutating.remove(job.id) }
        do {
            try await verify(job.owner, backend: backend, current: current)
            // A lost Resume response must not replay an earlier Stop on relaunch.
            let record = VideoJobAdmission(operationId: job.operationId, owner: job.owner)
            try save(record)
            let value = try await backend.resumeVideoJob(id: job.operationId, destinationInstanceId: job.owner.instanceId)
            guard current(), !Task.isCancelled else { return }
            guard value.id == job.operationId else { throw MoldClientError.malformedResponse }
            publish(value, owner: job.owner)
            follow(record, owner: job.owner, backend: backend, current: current)
        } catch { if current() { message(UserFacingError.describe(error), id: job.operationId, owner: job.owner) } }
    }

    /// The app confirms deleting recovery data; saved Library outputs remain.
    func delete(_ job: ObservedVideoJob, backend: any MoldBackend,
                current: @escaping @MainActor () -> Bool) async {
        let backend = scoped(backend, owner: job.owner)
        guard job.detail?.canDelete == true, !mutating.contains(job.id) else { return }
        mutating.insert(job.id); deleting.insert(job.id); readTokens[job.id] = UUID()
        defer { mutating.remove(job.id); deleting.remove(job.id) }
        do {
            try await verify(job.owner, backend: backend, current: current)
            do {
                try await backend.deleteVideoJob(id: job.operationId, destinationInstanceId: job.owner.instanceId)
            } catch {
                guard case let .http(status, code, _) = error as? MoldClientError,
                      status == 404, code == "VIDEO_JOB_NOT_FOUND" else { throw error }
            }
            guard current(), !Task.isCancelled else { return }
            try recordRemoval(job.operationId, owner: job.owner)
            try forget(job.operationId, owner: job.owner)
            retireReads(job.id)
            jobs.removeAll { $0.id == job.id && $0.owner == job.owner }
        } catch { if current() { message(UserFacingError.describe(error), id: job.operationId, owner: job.owner) } }
    }
}
