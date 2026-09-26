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
        let progress = hidden ? "Hiding collection" : "Showing collection"
        let verb = hidden ? "hide the collection “\(shelf.name)”" : "show the collection “\(shelf.name)”"
        await runShelfOperation(shelf, progress: progress, errorVerb: verb) { client, id in
            _ = try await client.updateCollection(id: id, change: CollectionChange(hidden: hidden))
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
