import Foundation
import MoldClient

// A shelf's lifecycle: made, renamed, removed, hidden. Split from
// `LibraryStore+Organization` for size -- that file reads the shelves and
// tags, this one changes what shelves exist.
@MainActor
extension LibraryStore {
    /// A shelf is made on the machine you are looking at; the others get their
    /// copy the first time something is filed into it there.
    @discardableResult
    func createShelf(named name: String, on hostID: MoldHost.ID) async -> CollectionShelf? {
        guard let client = hosts.backend(for: hostID) else { return nil }
        var created: Collection?
        do {
            created = try await client.createCollection(name: name, description: nil)
            hosts.succeeded(on: hostID)
        } catch {
            hosts.report(error, on: hostID, doing: "create the collection “\(name)”")
        }
        await reloadCollections()
        guard let created else { return nil }
        return CollectionShelf.merge([hostID: [created]]).first
    }

    /// Renames every machine's copy, so the shelf does not split in two.
    func renameShelf(_ shelf: CollectionShelf, to name: String) async {
        await runShelfOperation(
            shelf, progress: "Renaming collection",
            errorVerb: "rename the collection “\(shelf.name)”"
        ) { client, id in
            _ = try await client.updateCollection(id: id, change: CollectionChange(name: name))
        }
    }

    /// Removes the shelf from every machine. The prints stay -- only the
    /// membership goes.
    func deleteShelf(_ shelf: CollectionShelf) async {
        await runShelfOperation(
            shelf, progress: "Deleting collection",
            errorVerb: "delete the collection “\(shelf.name)”"
        ) { client, id in
            try await client.deleteCollection(id: id)
        }
    }

    func setShelfHidden(_ shelf: CollectionShelf, hidden: Bool) async {
        collectionVisibility.set(shelf.slug, hidden: hidden, hosts: hosts.hosts)
        collectionVisibility.persist(to: AppStorageSuite.defaults, key: "library.collectionVisibility")
        await reconcileCollectionVisibility()
        await reloadCollections()
    }

    func reconcileCollectionVisibility() async {
        guard !reconcilingCollectionVisibility else { return }
        reconcilingCollectionVisibility = true
        defer {
            reconcilingCollectionVisibility = false
            collectionVisibility.persist(to: AppStorageSuite.defaults, key: "library.collectionVisibility")
        }
        await collectionVisibility.reconcile(hosts: { self.hosts.hosts },
            collections: { self.collectionsPerHost }, available: { self.collectionInventoryAvailable }) { host, collection, hidden in
            do {
                let updated = try await self.hosts.backend(for: host).updateCollection(id: collection.id, change: CollectionChange(hidden: hidden))
                guard self.hosts.host(host.id) == host else { return false }
                self.collectionsPerHost[host.id] = self.collectionsPerHost[host.id]?.map { $0.id == updated.id ? updated : $0 }
                return true
            } catch {
                self.hosts.report(error, on: host.id, doing: hidden ? "hide the shared collection" : "show the shared collection")
                return false
            }
        }
    }

    private func runShelfOperation(
        _ shelf: CollectionShelf,
        progress: String,
        errorVerb: String,
        send: (any MoldBackend, String) async throws -> Void
    ) async {
        let destinations: [(host: MoldHost, client: any MoldBackend, collectionID: String)] =
            shelf.hosts.compactMap { hostID, collectionID in
                guard let host = hosts.host(hostID), let client = hosts.backend(for: hostID)
                else { return nil }
                return (host, client, collectionID)
            }
            .sorted { $0.host.id.uuidString < $1.host.id.uuidString }
        let activity = beginBulkActivity(
            "\(progress) on 0 of \(destinations.count.formatted()) machines…"
        )
        defer { endBulkActivity(activity) }
        for (index, destination) in destinations.enumerated() {
            guard hosts.host(destination.host.id) == destination.host else { continue }
            updateBulkActivity(
                activity,
                "\(progress) on \((index + 1).formatted()) of \(destinations.count.formatted()) machines…"
            )
            do {
                try await send(destination.client, destination.collectionID)
                guard hosts.host(destination.host.id) == destination.host else { continue }
                hosts.succeeded(on: destination.host.id)
            } catch {
                guard hosts.host(destination.host.id) == destination.host else { continue }
                hosts.report(error, on: destination.host.id, doing: errorVerb)
            }
        }
        updateBulkActivity(activity, "Refreshing collections…")
        await reloadCollections()
    }
}
