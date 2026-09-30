import Foundation
import MoldClient

extension LibraryStore {
    /// Host-local ids are the membership values in each print's metadata.
    var hiddenCollectionIDs: [MoldHost.ID: Set<String>] {
        var result: [MoldHost.ID: Set<String>] = [:]
        for shelf in shelves where shelf.hidden {
            for (host, id) in shelf.hosts { result[host, default: []].insert(id) }
        }
        return result
    }

    /// Hiding a merged shelf changes every machine's copy of it.
    func setShelfHidden(_ shelf: CollectionShelf, hidden: Bool) async {
        for (id, collectionID) in shelf.hosts.sorted(by: { $0.key.uuidString < $1.key.uuidString }) {
            guard let host = hosts.host(id), let backend = hosts.backend(for: id) else { continue }
            do {
                _ = try await backend.updateCollection(id: collectionID, change: CollectionChange(hidden: hidden))
            } catch {
                hosts.report(host, doing: hidden ? String(localized: "hide the collection") : String(localized: "show the collection"), error)
            }
            await reload(id)
        }
    }
}
