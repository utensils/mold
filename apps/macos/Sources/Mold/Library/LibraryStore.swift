import Foundation
import MoldClient

/// Every machine's prints, in one timeline.
///
/// `GET /api/gallery` has no pagination: a host answers with its entire index
/// in one array. So the app holds each host's index whole, merges them, and
/// filters locally -- and refreshes by ETag, which a live host answers with a
/// 304 and zero bytes instead of re-serializing 1.2 MB.
@MainActor
@Observable
final class LibraryStore {
    /// The one object that knows which machines exist and how to reach them.
    let hosts: HostStore

    /// `internal(set)` for the same reason `trashed` is: `rebuild()` lives in
    /// `+Rows` and `private(set)` does not cross a file boundary.
    internal(set) var items: [LibraryEntry] = []
    /// A count, not a flag: two overlapping refreshes (a manual ⌘R while an
    /// automatic one is still in flight, say) used to have the first one's
    /// `defer` turn this off while the second was still running.
    private var loads = 0
    var isLoading: Bool { loads > 0 }

    /// Set from here and from `+Mutations`'s `refreshTrash()`; `private(set)`
    /// does not cross that file boundary.
    internal(set) var trashed: [LibraryEntry] = []

    var perHost: [MoldHost.ID: [LibraryEntry]] = [:]
    var trashPerHost: [MoldHost.ID: [LibraryEntry]] = [:]
    var etags: [MoldHost.ID: String] = [:]
    var trashEtags: [MoldHost.ID: String] = [:]

    /// Collections and tags as each MACHINE holds them. A collection's id is
    /// that machine's, so these are never merged in storage -- only when they
    /// are read, by `CollectionShelf.merge`.
    var collectionsPerHost: [MoldHost.ID: [Collection]] = [:]
    var tagsPerHost: [MoldHost.ID: [TagCount]] = [:]

    /// The Edit menu's Undo, for the edits this store makes. Its manager is
    /// the window's, handed over by the pane -- see `MoldUndo`.
    let undo = MoldUndo()

    /// Organization edits on their way to the machines. See
    /// `LibraryStore+Outbox`.
    var outbox = MutationOutbox()
    /// The machines whose chain a task is already walking.
    var draining: Set<MoldHost.ID> = []
    /// Which live frames are this app's own edit coming back, and which
    /// machines are owed a re-list because one was skipped. See `GalleryEcho`.
    var echo = GalleryEcho()

    /// Bumped whenever the rows change, so anything derived from them knows to
    /// rebuild without comparing thousands of entries -- two libraries of the
    /// same size differ by one print's favourite star. Stored here because a
    /// stored property cannot live in an extension, and `internal(set)` for
    /// the same reason `trashed` is -- `rowsChanged()` lives in `+Rows` and
    /// `private(set)` does not cross a file boundary. See
    /// `LibraryShowingCache`.
    internal(set) var revision = 0

    init(hosts: HostStore) {
        self.hosts = hosts
        // Listening starts with the store, not with a pane. The Library used
        // to register on appearing, so a print made while Generate was showing
        // reached nobody and the timeline only caught up on the next ⌘R.
        // What still waits for a pane is the first LISTING -- these are
        // deltas, and a client that has read nothing has nothing to apply
        // them to. See `LibraryStore+Live`.
        hosts.onEvent { [weak self] host, event in
            self?.apply(event, from: host)
        }
    }

    /// Prints from every host, newest first.
    func refresh() async {
        loads += 1
        defer { loads -= 1 }
        // A machine that was removed must not keep contributing prints to a
        // merged timeline nobody can attribute them from.
        prune(to: hosts.hosts)

        await withTaskGroup(of: (MoldHost, Result<Fetched<[GalleryPrint]>, Error>).self) { group in
            for host in hosts.hosts {
                let client = hosts.backend(for: host)
                let etag = etags[host.id]
                group.addTask {
                    do { return (host, .success(try await client.gallery(etag: etag))) }
                    catch { return (host, .failure(error)) }
                }
            }
            for await (host, result) in group {
                apply(result, for: host)
            }
        }
        rebuild()
    }

    /// Lists, then re-lists the trash and the shelves. Shelves and tags travel
    /// with the index: reloading one without the other leaves a renamed
    /// collection still reading its old name.
    func reload() async {
        await refresh()
        await refreshTrash()
        await refreshOrganization()
    }

    private func apply(_ result: Result<Fetched<[GalleryPrint]>, Error>, for host: MoldHost) {
        switch result {
        case let .success(.fresh(prints, etag)):
            if let etag { etags[host.id] = etag }
            perHost[host.id] = prints.map { LibraryEntry(host: host, print: $0) }
            // Scoped: `refresh` is also `reload`'s own passive listing, run
            // right after other actions, and must not clear what one of
            // THOSE just reported.
            hosts.succeeded(on: host.id, doing: "list its prints")
        case .success(.notModified):
            // Nothing changed. Keeping the cached rows is the whole point of
            // having asked conditionally.
            hosts.succeeded(on: host.id, doing: "list its prints")
        case let .failure(error):
            // A host going down must not erase what it already showed us --
            // the other machines' prints stay, and so do this one's.
            hosts.report(error, on: host.id, doing: "list its prints")
        }
    }
}
