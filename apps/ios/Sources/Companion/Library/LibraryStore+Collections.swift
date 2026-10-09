import Foundation
import MoldClient

extension LibraryStore {
    var machineIDs: Set<MoldHost.ID> { machineID.map { [$0] } ?? [] }
    var scopedPool: [LibraryEntry] {
        scopedPool(on: machineIDs)
    }
    func scopedPool(on ids: Set<MoldHost.ID>) -> [LibraryEntry] {
        ids.isEmpty ? pool : pool.compactMap { $0.presented(onAnyOf: ids) }
    }
    var hiddenCollectionIDs: [MoldHost.ID: Set<String>] { CollectionShelf.hiddenIDs(in: shelves) }

    func shelfPresence(_ shelf: CollectionShelf) -> CollectionShelf.Presence {
        shelfPresence(shelf, on: machineIDs)
    }

    func shelfPresence(_ shelf: CollectionShelf, on ids: Set<MoldHost.ID>) -> CollectionShelf.Presence {
        shelf.presence(on: ids, available: collectionInventoryAvailable.intersection(Set(hosts.upHosts.map(\.id))))
    }

    /// Visibility is a shared shelf attribute, even while browsing one machine.
    func setShelfHidden(_ shelf: CollectionShelf, hidden: Bool) async {
        collectionVisibility.set(shelf.slug, hidden: hidden, hosts: hosts.hosts)
        collectionVisibility.persist(to: .standard, key: "library.collectionVisibility")
        rebuildNow()
        await reconcileCollectionVisibility()
    }

    func reconcileCollectionVisibility() async {
        guard !reconcilingCollectionVisibility else { return }
        reconcilingCollectionVisibility = true
        defer {
            reconcilingCollectionVisibility = false
            collectionVisibility.persist(to: .standard, key: "library.collectionVisibility")
            rebuildNow()
        }
        await collectionVisibility.reconcile(hosts: { self.hosts.hosts },
            collections: { self.collectionInventory }, available: { self.collectionInventoryAvailable }) { host, collection, hidden in
            guard let backend = self.hosts.backend(for: host.id) else { return false }
            do {
                let updated = try await backend.updateCollection(id: collection.id, change: CollectionChange(hidden: hidden))
                guard self.hosts.host(host.id) == host else { return false }
                self.acceptCollection(updated, on: host.id)
                return true
            } catch {
                self.hosts.report(host, doing: hidden ? String(localized: "hide the shared collection") : String(localized: "show the shared collection"), error)
                return false
            }
        }
    }
}
