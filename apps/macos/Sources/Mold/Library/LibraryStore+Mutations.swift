import Foundation
import MoldClient

// What the UI asks for. Each of these narrows a request to the prints it would
// actually change and hands it to `apply`, which is the one place that mutates,
// registers the undo and talks to the machines.
@MainActor
extension LibraryStore {

    // MARK: - Trash

    func refreshTrash() async {
        await withTaskGroup(of: (MoldHost, [LibraryEntry]?, String?).self) { group in
            for host in hosts.hosts {
                let client = hosts.backend(for: host)
                let etag = trashEtags[host.id]
                group.addTask {
                    guard let fetched = try? await client.trashedPrints(etag: etag)
                    else { return (host, nil, nil) }
                    guard let prints = fetched.value else { return (host, nil, nil) }
                    return (host, prints.map { LibraryEntry(host: host, print: $0) }, fetched.etag)
                }
            }
            for await (host, entries, etag) in group {
                if let entries { trashPerHost[host.id] = entries }
                if let etag { trashEtags[host.id] = etag }
            }
        }
        trashed = trashPerHost.values.flatMap(\.self)
            .sorted { ($0.print.trashedAt ?? 0) > ($1.print.trashedAt ?? 0) }
    }

    // MARK: - Mutations

    func setFavorite(_ favorite: Bool, on entries: [LibraryEntry]) {
        apply(PrintEdit.plan(.favorite(favorite), over: entries))
    }

    func setTag(_ tag: String, adding: Bool, on entries: [LibraryEntry]) {
        let clean = tag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }
        apply(PrintEdit.plan(.tag(clean, adding: adding), over: entries))
    }

    /// Names one print. The old name travels with the change so undo can put
    /// it back -- see `PrintChange.title`.
    func setTitle(_ title: String, on entry: LibraryEntry) {
        let clean = title.trimmingCharacters(in: .whitespacesAndNewlines)
        apply(PrintEdit.plan(.title(from: entry.print.title ?? "", to: clean), over: [entry]))
    }

    /// Trash keeps the bytes and starts a purge countdown; it is not a delete.
    ///
    /// Deliberately NOT on the undo stack. It already has a better answer --
    /// the print sits in Recently Deleted with its own countdown and its own
    /// Put Back, which survives quitting the app in a way an undo stack does
    /// not.
    func moveToTrash(_ entries: [LibraryEntry]) async {
        let previous = perHost
        let ids = Set(entries.map(\.id))
        for (hostID, list) in perHost {
            perHost[hostID] = list.filter { !ids.contains($0.id) }
        }
        rebuild()

        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = hosts.backend(for: hostID) else { continue }
            do { try await client.trash(group.map(\.print.filename)) } catch {
                perHost = previous
                rebuild()
                failures[hostID] = "Couldn't move those to the trash."
                return
            }
        }
        trashEtags.removeAll()
    }

    func restore(_ entries: [LibraryEntry]) async {
        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = hosts.backend(for: hostID) else { continue }
            try? await client.restoreFromTrash(group.map(\.print.filename))
        }
        etags.removeAll()
        trashEtags.removeAll()
    }

    /// Permanent on the host. Nothing here can undo it, which is why the panes
    /// ask first.
    func deleteForever(_ entries: [LibraryEntry]) async {
        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = hosts.backend(for: hostID) else { continue }
            try? await client.deleteForever(group.map(\.print.filename))
        }
        trashEtags.removeAll()
    }

    /// Empties every machine's trash at once. The confirmation lives in
    /// `LibraryActions+Destructive`; this is what runs once someone agrees.
    func emptyTrash() async {
        for host in hosts.hosts {
            try? await hosts.backend(for: host).emptyTrash()
        }
        await refreshTrash()
    }
}
