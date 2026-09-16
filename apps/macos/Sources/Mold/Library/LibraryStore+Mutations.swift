import Foundation
import MoldClient

// Changing what is in the library. Split from the fetching half purely
// for size.
@MainActor
extension LibraryStore {


    /// Applies a change locally first, then tells the host.
    ///
    /// A star that waits for a round trip feels broken on a remote machine, so
    /// the tile turns immediately and a failure puts it back -- rather than the
    /// UI and the host quietly disagreeing.
    func setFavorite(_ favorite: Bool, on entries: [LibraryEntry],
                     backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        let previous = perHost
        mutateLocally(entries) { $0.favorite = favorite }

        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            let mutation = GalleryBulkMutation(
                filenames: group.map(\.print.filename), favorite: favorite)
            do { try await client.mutate(mutation) } catch {
                perHost = previous
                rebuild()
                failures[hostID] = "Couldn't update those prints."
                return
            }
        }
    }

    /// Trash keeps the bytes and starts a purge countdown; it is not a delete.
    func moveToTrash(_ entries: [LibraryEntry],
                     backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        let previous = perHost
        let ids = Set(entries.map(\.id))
        for (hostID, list) in perHost {
            perHost[hostID] = list.filter { !ids.contains($0.id) }
        }
        rebuild()

        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            do { try await client.trash(group.map(\.print.filename)) } catch {
                perHost = previous
                rebuild()
                failures[hostID] = "Couldn't move those to the trash."
                return
            }
        }
        trashEtags.removeAll()
    }

    func restore(_ entries: [LibraryEntry],
                 backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            try? await client.restoreFromTrash(group.map(\.print.filename))
        }
        etags.removeAll()
        trashEtags.removeAll()
    }

    /// Permanent on the host. Nothing here can undo it.
    func deleteForever(_ entries: [LibraryEntry],
                       backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            try? await client.deleteForever(group.map(\.print.filename))
        }
        trashEtags.removeAll()
    }

    private func mutateLocally(_ entries: [LibraryEntry],
                               _ change: (inout GalleryPrint.Mutable) -> Void) {
        let ids = Set(entries.map(\.id))
        for (hostID, list) in perHost {
            perHost[hostID] = list.map { entry in
                guard ids.contains(entry.id) else { return entry }
                var mutable = GalleryPrint.Mutable(entry.print)
                change(&mutable)
                return LibraryEntry(hostID: entry.hostID, hostName: entry.hostName,
                                    print: mutable.build())
            }
        }
        rebuild()
    }
}
