import Foundation
import MoldClient

/// The one writer of shared collection visibility at a time: ordinary
/// Hide/Show reconciliation skips while it is held, and Library Sync waits
/// its turn -- on a continuation, never by polling.
@MainActor
final class VisibilityWriterLock {
    private(set) var isHeld = false
    private var waiters: [(id: UUID, continuation: CheckedContinuation<Void, any Error>)] = []
    /// Cancellations that arrived before their waiter was queued.
    private var cancelled: Set<UUID> = []

    func tryAcquire() -> Bool {
        guard !isHeld else { return false }
        isHeld = true
        return true
    }

    /// Waits its turn; a cancelled waiter leaves the queue and throws,
    /// never holding the lock.
    func acquire() async throws {
        try Task.checkCancellation()
        guard isHeld else {
            isHeld = true
            return
        }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                MainActor.assumeIsolated { self.enqueue(id, continuation) }
            }
        } onCancel: {
            Task { @MainActor in self.cancel(id) }
        }
    }

    private func enqueue(_ id: UUID, _ continuation: CheckedContinuation<Void, any Error>) {
        if cancelled.remove(id) != nil {
            continuation.resume(throwing: CancellationError())
        } else {
            waiters.append((id, continuation))
        }
    }

    private func cancel(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else {
            cancelled.insert(id)
            return
        }
        waiters.remove(at: index).continuation.resume(throwing: CancellationError())
    }

    /// Hands the lock straight to the next waiter, if there is one.
    func release() {
        if waiters.isEmpty { isHeld = false } else { waiters.removeFirst().continuation.resume() }
    }

    /// Waiters queued now. For tests.
    var waiting: Int { waiters.count }
}

struct CollectionVisibilitySync {
    var updated: [Collection] = []
    var failures: [(slug: String, message: String)] = []
}

@MainActor
extension LibraryStore {
    var reconcilingCollectionVisibility: Bool { visibilityWriters.isHeld }

    /// Share the visibility writer with ordinary Hide/Show reconciliation.
    /// Re-read intent after creation and after every write; an explicit newer
    /// Show must never be reversed by a sync snapshot captured before it.
    /// Every shelf in one pass under one lock, and This Mac's collection
    /// inventory assigned once at the end.
    func syncCollectionVisibility(
        _ shelves: [(Collection, Bool)], generation: UUID,
        destination: MoldHost, backend: any MoldBackend
    ) async -> CollectionVisibilitySync {
        var result = CollectionVisibilitySync()
        guard !shelves.isEmpty else { return result }
        do {
            try await visibilityWriters.acquire()
        } catch {
            result.failures = shelves.map { ($0.0.slug, "Sync stopped before its visibility was checked.") }
            return result
        }
        defer { visibilityWriters.release() }
        for (collection, fallbackHidden) in shelves {
            do {
                result.updated.append(try await syncVisibility(
                    of: collection, fallbackHidden: fallbackHidden, generation: generation,
                    destination: destination, backend: backend))
            } catch {
                result.failures.append((collection.slug, error.sentence))
            }
        }
        if hosts.host(destination.id) == destination, !result.updated.isEmpty {
            let ids = Set(result.updated.map(\.id))
            var collections = collectionsPerHost[destination.id] ?? []
            collections.removeAll { ids.contains($0.id) }
            collections.append(contentsOf: result.updated)
            collectionsPerHost[destination.id] = collections
        }
        return result
    }

    private func syncVisibility(
        of collection: Collection, fallbackHidden: Bool, generation: UUID,
        destination: MoldHost, backend: any MoldBackend
    ) async throws -> Collection {
        var current = collection
        while true {
            try Task.checkCancellation()
            guard hosts.host(destination.id) == destination else {
                throw MoldClientError.unreachable("The destination machine changed while syncing collections.")
            }
            let revision = collectionVisibility.generation
            let fallback = revision == generation ? fallbackHidden
                : shelf(slug: collection.slug)?.hidden ?? current.hidden ?? false
            let hidden = collectionVisibility.desiredHidden(slug: collection.slug, fallback: fallback)
            if (current.hidden ?? false) != hidden {
                current = try await backend.updateCollection(id: collection.id, change: CollectionChange(hidden: hidden))
                guard hosts.host(destination.id) == destination else {
                    throw MoldClientError.unreachable("The destination machine changed while syncing collections.")
                }
            }
            if revision != collectionVisibility.generation { continue }
            return current
        }
    }
}
