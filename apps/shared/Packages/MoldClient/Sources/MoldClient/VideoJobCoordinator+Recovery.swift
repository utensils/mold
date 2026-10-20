import Foundation

public extension VideoJobCoordinator {
    func refresh(_ id: String, owner: VideoJobOwner, backend: any MoldBackend,
                 current: @escaping @MainActor () -> Bool) async {
        let key = identity(id, owner: owner)
        guard !mutating.contains(key), current(), !Task.isCancelled else { return }
        let token = UUID(); readTokens[key] = token
        let fresh = { @MainActor in !Task.isCancelled && current() && self.readTokens[key] == token }
        do {
            try await verify(owner, backend: backend, current: fresh)
            if var record = pending.first(where: { $0.operationId == id && $0.matches(owner) }),
               record.cancelRequested && !record.cancelAcknowledged {
                mutating.insert(key)
                do {
                    try await backend.cancelVideoJob(id: id, destinationInstanceId: owner.instanceId)
                    record.cancelAcknowledged = true
                    try save(record)
                    mutating.remove(key)
                } catch { mutating.remove(key); throw error }
            }
            guard fresh() else { return }
            let value = try await backend.videoJob(id: id)
            guard fresh() else { return }
            guard value.id == id else { throw MoldClientError.malformedResponse }
            await releaseTransport(id, owner: owner)
            guard fresh() else { return }
            publish(value, owner: owner)
            if value.state == .completed || value.state == .cancelled { try forget(id, owner: owner) }
        } catch {
            guard fresh() else { return }
            if case let .http(status, code, _) = error as? MoldClientError,
               status == 410, code == "VIDEO_OPERATION_CANCELLED" {
                await releaseTransport(id, owner: owner)
                guard fresh() else { return }
                ensure(id, owner: owner)
                if let index = index(id, owner: owner) {
                    jobs[index].cancelledBeforeAdmission = true; jobs[index].message = "Cancelled before queueing"
                }
                do { try forget(id, owner: owner) } catch { message(UserFacingError.describe(error), id: id, owner: owner) }
            } else { message(UserFacingError.describe(error), id: id, owner: owner) }
        }
    }

    func dismiss(_ job: ObservedVideoJob) {
        guard job.cancelledBeforeAdmission || job.detail?.state == .completed || job.detail?.state == .cancelled else { return }
        do { try forget(job.operationId, owner: job.owner) }
        catch { message(UserFacingError.describe(error), id: job.operationId, owner: job.owner); return }
        retireReads(job.id)
        jobs.removeAll { $0.id == job.id && $0.owner == job.owner }
    }
}

extension VideoJobCoordinator {
    func follow(_ record: VideoJobAdmission, owner: VideoJobOwner, backend: any MoldBackend,
                current: @escaping @MainActor () -> Bool) {
        let key = owner.host.id.uuidString + ":" + record.operationId
        guard followers[key] == nil, current() else { return }
        let token = UUID(); followTokens[key] = token
        followers[key] = Task { [weak self] in
            guard let self else { return }
            defer { if self.followTokens[key] == token { self.followers[key] = nil; self.followTokens[key] = nil } }
            while !Task.isCancelled, current(), self.followTokens[key] == token {
                await self.refresh(record.operationId, owner: owner, backend: backend, current: current)
                guard !Task.isCancelled, self.followTokens[key] == token,
                      let job = self.jobs.first(where: { $0.operationId == record.operationId && $0.owner == owner }),
                      !job.cancelledBeforeAdmission,
                      job.detail?.state == .running || job.detail?.state == .queued || job.detail == nil else { return }
                do { try await Task.sleep(for: .seconds(2)) } catch { return }
            }
        }
    }
}
