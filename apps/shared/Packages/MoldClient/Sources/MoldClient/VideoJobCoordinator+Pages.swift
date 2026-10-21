import Foundation

public extension VideoJobCoordinator {
    func loadMore(owner: VideoJobOwner, backend: any MoldBackend, current: @escaping @MainActor () -> Bool) async {
        guard let page = pagination[owner.host.id], page.owner == owner, !page.isLoading else { return }
        if page.error != nil {
            await loadPage(owner: owner, cursor: page.retryCursor, backend: backend, current: current)
        } else if page.hasMore, let cursor = page.nextCursor {
            await loadPage(owner: owner, cursor: cursor, backend: backend, current: current)
        }
    }
}

extension VideoJobCoordinator {
    /// Queue pages contain recoverable parents before the server applies its
    /// limit. Opaque keyset cursors survive earlier rows completing/deleting.
    func loadPage(owner: VideoJobOwner, cursor: String?, backend: any MoldBackend,
                  current: @escaping @MainActor () -> Bool) async {
        let backend = scoped(backend, owner: owner)
        guard current(), !Task.isCancelled else { return }
        let token = UUID(); pageTokens[owner.host.id] = token
        let previous = pagination[owner.host.id].flatMap { $0.owner == owner ? $0 : nil }
        pagination[owner.host.id] = .init(owner: owner, hasMore: previous?.hasMore ?? false,
            nextCursor: previous?.nextCursor, isLoading: true, error: nil, retryCursor: cursor)
        let fresh = { @MainActor in !Task.isCancelled && current() && self.pageTokens[owner.host.id] == token }
        defer {
            if pageTokens[owner.host.id] == token { pagination[owner.host.id]?.isLoading = false }
        }
        do {
            try await verify(owner, backend: backend, current: fresh)
            let listing = try await backend.listVideoJobs(limit: 50, cursor: cursor)
            guard fresh() else { return }
            guard !listing.hasMore || listing.nextCursor != cursor else { throw MoldClientError.malformedResponse }
            for value in listing.jobs {
                if value.state == .completed, !jobs.contains(where: {
                    $0.operationId == value.id && $0.owner.host.id == owner.host.id && $0.owner.instanceId == owner.instanceId
                }) { continue }
                publish(value, owner: owner)
                if value.state == .queued || value.state == .running {
                    follow(.init(operationId: value.id, owner: owner), owner: owner, backend: backend, current: current)
                }
            }
            pagination[owner.host.id] = .init(owner: owner, hasMore: listing.hasMore,
                nextCursor: listing.nextCursor, isLoading: false, error: nil)
        } catch {
            if fresh() {
                let message = UserFacingError.describe(error)
                pagination[owner.host.id]?.error = message
            }
        }
    }
}
