import Foundation
import MoldClient

// What the UI asks for. Each of these narrows a request to the prints it would
// actually change and hands it to `apply`, which is the one place that mutates,
// registers the undo and talks to the machines.
@MainActor
extension LibraryStore {

    func setFavorite(_ favorite: Bool, on entries: [LibraryEntry],
                     backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) {
        apply(PrintEdit.plan(.favorite(favorite), over: entries), backend: backend)
    }

    func setTag(_ tag: String, adding: Bool, on entries: [LibraryEntry],
                backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) {
        let clean = tag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }
        apply(PrintEdit.plan(.tag(clean, adding: adding), over: entries), backend: backend)
    }

    /// Names one print. The old name travels with the change so undo can put
    /// it back -- see `PrintChange.title`.
    func setTitle(_ title: String, on entry: LibraryEntry,
                  backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) {
        let clean = title.trimmingCharacters(in: .whitespacesAndNewlines)
        apply(PrintEdit.plan(.title(from: entry.print.title ?? "", to: clean), over: [entry]),
              backend: backend)
    }

    /// Trash keeps the bytes and starts a purge countdown; it is not a delete.
    ///
    /// Deliberately NOT on the undo stack. It already has a better answer --
    /// the print sits in Recently Deleted with its own countdown and its own
    /// Put Back, which survives quitting the app in a way an undo stack does
    /// not.
    func moveToTrash(_ entries: [LibraryEntry],
                     backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) async {
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
                 backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) async {
        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            try? await client.restoreFromTrash(group.map(\.print.filename))
        }
        etags.removeAll()
        trashEtags.removeAll()
    }

    /// Permanent on the host. Nothing here can undo it, which is why the panes
    /// ask first.
    func deleteForever(_ entries: [LibraryEntry],
                       backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) async {
        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            try? await client.deleteForever(group.map(\.print.filename))
        }
        trashEtags.removeAll()
    }
}
