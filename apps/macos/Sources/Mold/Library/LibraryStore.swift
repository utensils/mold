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

    private(set) var items: [LibraryEntry] = []
    private(set) var isLoading = false
    var failures: [MoldHost.ID: String] = [:]

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

    /// The registration with `HostStore`'s event fan-out. See
    /// `LibraryStore+Live`.
    var listening: UUID?

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    /// Prints from every host, newest first.
    func refresh() async {
        isLoading = true
        defer { isLoading = false }
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
            failures[host.id] = nil
            if let etag { etags[host.id] = etag }
            perHost[host.id] = prints.map { LibraryEntry(host: host, print: $0) }
        case .success(.notModified):
            // Nothing changed. Keeping the cached rows is the whole point of
            // having asked conditionally.
            failures[host.id] = nil
        case let .failure(error):
            failures[host.id] = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
            // A host going down must not erase what it already showed us --
            // the other machines' prints stay, and so do this one's.
        }
    }

    /// Drops machines that are no longer in the list, so their prints don't
    /// linger. Called from `refresh`, because removing a machine is exactly
    /// when nobody thinks to reload the library.
    func prune(to hostList: [MoldHost]) {
        let live = Set(hostList.map(\.id))
        guard perHost.contains(where: { !live.contains($0.key) })
            || trashPerHost.contains(where: { !live.contains($0.key) })
        else { return }
        perHost = perHost.filter { live.contains($0.key) }
        trashPerHost = trashPerHost.filter { live.contains($0.key) }
        etags = etags.filter { live.contains($0.key) }
        trashEtags = trashEtags.filter { live.contains($0.key) }
        failures = failures.filter { live.contains($0.key) }
        collectionsPerHost = collectionsPerHost.filter { live.contains($0.key) }
        tagsPerHost = tagsPerHost.filter { live.contains($0.key) }
        rebuild()
        trashed = trashPerHost.values.flatMap(\.self)
            .sorted { ($0.print.trashedAt ?? 0) > ($1.print.trashedAt ?? 0) }
    }

    func rebuild() {
        items = perHost.values.flatMap(\.self)
            .filter { $0.print.trashedAt == nil }
            .sorted { $0.print.timestamp > $1.print.timestamp }
    }

    func count(for host: MoldHost.ID) -> Int { perHost[host]?.count ?? 0 }
}
