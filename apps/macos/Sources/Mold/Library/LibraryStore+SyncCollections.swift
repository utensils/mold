import Foundation
import MoldClient

@MainActor
extension LibraryStore {
    /// Share the visibility writer with ordinary Hide/Show reconciliation.
    /// Re-read intent after creation and after every write; an explicit newer
    /// Show must never be reversed by a sync snapshot captured before it.
    func syncCollectionVisibility(
        _ collection: Collection, fallbackHidden: Bool, generation: UUID,
        destination: MoldHost, backend: any MoldBackend
    ) async throws -> Collection {
        while reconcilingCollectionVisibility {
            try await Task.sleep(for: .milliseconds(10))
        }
        reconcilingCollectionVisibility = true
        defer { reconcilingCollectionVisibility = false }
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
            var collections = collectionsPerHost[destination.id] ?? []
            collections.removeAll { $0.id == current.id }
            collections.append(current)
            collectionsPerHost[destination.id] = collections
            return current
        }
    }
}
