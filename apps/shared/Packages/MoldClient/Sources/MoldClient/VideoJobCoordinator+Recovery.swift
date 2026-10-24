import Foundation

public extension VideoJobCoordinator {
    func refresh(_ id: String, owner: VideoJobOwner, backend: any MoldBackend,
                 current: @escaping @MainActor () -> Bool) async {
        let backend = scoped(backend, owner: owner)
        let key = identity(id, owner: owner)
        guard !isRemoved(id, owner: owner), !mutating.contains(key), current(), !Task.isCancelled else { return }
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

    /// The per-job owner re-check the Queue runs on its ordinary refresh and
    /// poll (it replaced a per-row "Check queue" button). Live parents are
    /// re-read; a finished one re-verifies its saved Library video only when
    /// `force` (the pane's own refresh), since its status cannot change.
    func recheck(backend: (VideoJobOwner) -> (any MoldBackend)?,
                 current: @escaping @MainActor (VideoJobOwner) -> Bool, force: Bool = false) async {
        for job in jobs where !job.cancelledBeforeAdmission {
            let owner = job.owner
            guard !Task.isCancelled, let service = backend(owner), current(owner) else { continue }
            if job.detail?.isSettled != true {
                await refresh(job.operationId, owner: owner, backend: service, current: { current(owner) })
            }
            guard let live = jobs.first(where: { $0.id == job.id && $0.owner == owner }),
                  live.detail?.outputFilename != nil else { continue }
            await checkOutput(live, backend: service, current: { current(owner) }, force: force)
        }
    }

    func dismiss(_ job: ObservedVideoJob) {
        guard let live = jobs.first(where: { $0.id == job.id && $0.owner == job.owner }),
              live.cancelledBeforeAdmission || live.detail?.state == .completed else { return }
        do {
            try recordRemoval(job.operationId, owner: job.owner)
            try forget(job.operationId, owner: job.owner)
        }
        catch { message(UserFacingError.describe(error), id: job.operationId, owner: job.owner); return }
        retireReads(job.id)
        jobs.removeAll { $0.id == job.id && $0.owner == job.owner }
    }
}

extension VideoJobCoordinator {
    func follow(_ record: VideoJobAdmission, owner: VideoJobOwner, backend: any MoldBackend,
                current: @escaping @MainActor () -> Bool) {
        let key = owner.host.id.uuidString + ":" + record.operationId
        guard !isRemoved(record.operationId, owner: owner), current() else { return }
        if let existing = followers[key], !existing.isCancelled, followOwners[key] == owner { return }
        followers[key]?.cancel()
        let token = UUID(); followTokens[key] = token; followOwners[key] = owner
        followers[key] = Task { [weak self] in
            guard let self else { return }
            defer { if self.followTokens[key] == token { self.followers[key] = nil; self.followTokens[key] = nil; self.followOwners[key] = nil } }
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
