import Foundation
import MoldClient

// A shelf's lifecycle: made, renamed, removed, hidden. Split from
// `LibraryStore+Organization` for size -- that file reads the shelves and
// tags, this one changes what shelves exist.
@MainActor
extension LibraryStore {
    /// A shelf is made on the machine you are looking at; the others get their
    /// copy the first time something is filed into it there.
    func createShelf(named name: String, on hostID: MoldHost.ID) async {
        guard let client = hosts.backend(for: hostID) else { return }
        do {
            _ = try await client.createCollection(name: name, description: nil)
            hosts.succeeded(on: hostID)
        } catch {
            hosts.report(error, on: hostID, doing: "create the collection “\(name)”")
        }
        await reloadCollections()
    }

    /// Renames every machine's copy, so the shelf does not split in two.
    func renameShelf(_ shelf: CollectionShelf, to name: String) async {
        for (hostID, id) in shelf.hosts {
            guard let client = hosts.backend(for: hostID) else { continue }
            do {
                _ = try await client.updateCollection(id: id, change: CollectionChange(name: name))
                hosts.succeeded(on: hostID)
            } catch {
                hosts.report(error, on: hostID, doing: "rename the collection “\(shelf.name)”")
            }
        }
        await reloadCollections()
    }

    /// Removes the shelf from every machine. The prints stay -- only the
    /// membership goes.
    func deleteShelf(_ shelf: CollectionShelf) async {
        for (hostID, id) in shelf.hosts {
            guard let client = hosts.backend(for: hostID) else { continue }
            do {
                try await client.deleteCollection(id: id)
                hosts.succeeded(on: hostID)
            } catch {
                hosts.report(error, on: hostID, doing: "delete the collection “\(shelf.name)”")
            }
        }
        await reloadCollections()
    }

    func setShelfHidden(_ shelf: CollectionShelf, hidden: Bool) async {
        for (hostID, id) in shelf.hosts {
            guard let client = hosts.backend(for: hostID) else { continue }
            do {
                _ = try await client.updateCollection(id: id, change: CollectionChange(hidden: hidden))
                hosts.succeeded(on: hostID)
            } catch {
                let verb = hidden ? "hide the collection “\(shelf.name)”" : "show the collection “\(shelf.name)”"
                hosts.report(error, on: hostID, doing: verb)
            }
        }
        await reloadCollections()
    }
}
