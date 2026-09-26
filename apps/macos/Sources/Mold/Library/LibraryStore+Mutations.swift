import Foundation
import MoldClient

// What the UI asks for. Each of these narrows a request to the prints it would
// actually change and hands it to `apply`, which is the one place that mutates,
// registers the undo and talks to the machines.
@MainActor
extension LibraryStore {

    // MARK: - Trash

    func refreshTrash() async {
        await withTaskGroup(of: (MoldHost, Result<Fetched<[GalleryPrint]>, Error>).self) { group in
            for host in hosts.hosts {
                let client = hosts.backend(for: host)
                let etag = trashEtags[host.id]
                group.addTask {
                    do { return (host, .success(try await client.trashedPrints(etag: etag))) }
                    catch { return (host, .failure(error)) }
                }
            }
            for await (host, result) in group {
                guard !Task.isCancelled, hosts.host(host.id) == host else { continue }
                switch result {
                case let .success(.fresh(prints, etag)):
                    trashPerHost[host.id] = prints.map { LibraryEntry(host: host, print: $0) }
                    if let etag { trashEtags[host.id] = etag }
                    // Scoped: this passive listing runs right after
                    // `emptyTrash` too, and must not clear what THAT reported.
                    hosts.succeeded(on: host.id, doing: "list its trash")
                case .success(.notModified):
                    hosts.succeeded(on: host.id, doing: "list its trash")
                case let .failure(error):
                    hosts.report(error, on: host.id, doing: "list its trash")
                }
            }
        }
        rebuildTrash()
        rows.bump()
    }

    // MARK: - Mutations

    /// A merged tile stands for every copy of its print, so an edit reaches
    /// all of them -- starring the This Mac copy and leaving the original
    /// unstarred would split one print in two again.
    func withCopies(_ entries: [LibraryEntry]) -> [LibraryEntry] {
        var seen = Set<PrintID>()
        return entries.flatMap(\.everyCopy).filter { seen.insert($0.id).inserted }
    }

    func setFavorite(_ favorite: Bool, on entries: [LibraryEntry]) {
        apply(PrintEdit.plan(.favorite(favorite), over: withCopies(entries)))
    }

    func setTag(_ tag: String, adding: Bool, on entries: [LibraryEntry]) {
        let clean = tag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }
        apply(PrintEdit.plan(.tag(clean, adding: adding), over: withCopies(entries)))
    }

    /// Names one print. The old name travels with the change so undo can put
    /// it back -- see `PrintChange.title`.
    func setTitle(_ title: String, on entry: LibraryEntry) {
        let clean = title.trimmingCharacters(in: .whitespacesAndNewlines)
        // One edit per previous title, so undo puts back what EACH copy was
        // called rather than the lead's name on all of them. Registered in
        // the same run-loop turn, they are one undo group.
        let byPrevious = Dictionary(grouping: withCopies([entry])) { $0.print.title ?? "" }
        for (previous, copies) in byPrevious.sorted(by: { $0.key < $1.key }) {
            apply(PrintEdit.plan(.title(from: previous, to: clean), over: copies))
        }
    }

    /// Trash keeps the bytes and starts a purge countdown; it is not a delete.
    ///
    /// Deliberately NOT on the undo stack. It already has a better answer --
    /// the print sits in Recently Deleted with its own countdown and its own
    /// Put Back, which survives quitting the app in a way an undo stack does
    /// not.
    func moveToTrash(_ entries: [LibraryEntry]) async {
        await runBulk(.trash, entries: entries)
    }

    func restore(_ entries: [LibraryEntry]) async {
        await runBulk(.restore, entries: entries)
    }

    func deleteForever(_ entries: [LibraryEntry]) async {
        await runBulk(.delete, entries: entries)
    }

    func emptyTrash() async {
        await runEmptyTrash()
    }
}
