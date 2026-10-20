import Foundation

public extension VideoJobCoordinator {
    /// Prepare exactly one captured snapshot, then validate its actual inputs.
    /// The draft fence is used only before POST; admitted work owns its identity.
    func submit(_ request: VideoJobRequest, owner: VideoJobOwner, backend: any MoldBackend,
                current: @escaping @MainActor () -> Bool, unchanged: @escaping @MainActor () -> Bool,
                prepare: (@MainActor () async throws -> VideoJobRequest)? = nil) async {
        guard !isSubmitting, journalError == nil else { return }
        let token = UUID(); submissionToken = token
        let fresh = { @MainActor in !Task.isCancelled && token == self.submissionToken && current() && unchanged() }
        isSubmitting = true; error = nil
        var saved: VideoJobAdmission?
        defer { if token == submissionToken { isSubmitting = false } }
        do {
            try await verify(owner, backend: backend, current: fresh)
            var prepared = try await prepare?() ?? request
            guard fresh() else { throw changed() }
            let plan = try await backend.validateVideoJob(prepared)
            guard fresh(), !plan.validationRevision.isEmpty else { throw changed() }
            prepared.validationRevision = plan.validationRevision
            try await verify(owner, backend: backend, current: fresh)
            guard fresh() else { throw changed() }
            let record = VideoJobAdmission(operationId: UUID().uuidString.lowercased(), owner: owner)
            try save(record); saved = record
            ensure(record.operationId, owner: owner)
            let accepted = try await backend.createVideoJob(prepared, operationId: record.operationId, destinationInstanceId: owner.instanceId)
            guard accepted.id == record.operationId else { throw MoldClientError.malformedResponse }
            guard current() else { return }
            publish(accepted, owner: owner)
            follow(record, owner: owner, backend: backend, current: current)
        } catch {
            if let saved, VideoRestartController.definitive(error) {
                do {
                    try forget(saved.operationId, owner: owner)
                    jobs.removeAll { $0.operationId == saved.operationId && $0.owner == owner }
                } catch { self.error = UserFacingError.describe(error); return }
            }
            if token == submissionToken { self.error = UserFacingError.describe(error) }
            if let saved, pending.contains(where: { $0.operationId == saved.operationId && $0.matches(owner) }) {
                message(UserFacingError.describe(error), id: saved.operationId, owner: owner)
            }
        }
    }

    func recover(owner: VideoJobOwner, backend: any MoldBackend, current: @escaping @MainActor () -> Bool) async {
        guard current() else { return }
        for record in pending where record.matches(owner) {
            guard current() else { return }
            ensure(record.operationId, owner: owner)
            await refresh(record.operationId, owner: owner, backend: backend, current: current)
            if let detail = jobs.first(where: { $0.operationId == record.operationId && $0.owner == owner })?.detail,
               detail.state == .queued || detail.state == .running {
                follow(record, owner: owner, backend: backend, current: current)
            }
        }
    }

    func discover(owner: VideoJobOwner, backend: any MoldBackend, activityIDs: [String] = [],
                  current: @escaping @MainActor () -> Bool) async {
        await loadPage(owner: owner, cursor: nil, backend: backend, current: current)
        for id in Set(activityIDs) where index(id, owner: owner) == nil {
            guard current(), !Task.isCancelled else { return }
            await refresh(id, owner: owner, backend: backend, current: current)
            follow(.init(operationId: id, owner: owner), owner: owner, backend: backend, current: current)
        }
    }
}
