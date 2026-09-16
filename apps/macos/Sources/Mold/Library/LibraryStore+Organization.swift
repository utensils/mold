import Foundation
import MoldClient

// Shelves and tags. Split from the fetching and mutating halves for size.
@MainActor
extension LibraryStore {

    /// Every machine's collections, folded into the shelves a person sees.
    var shelves: [CollectionShelf] { CollectionShelf.merge(collectionsPerHost) }

    /// Tag names and how many prints carry them, summed across machines.
    /// Sorted by how much they are used, because a tag suggestion list is only
    /// useful if the tags you actually use are at the top.
    var tagCounts: [TagCount] {
        var totals: [String: Int] = [:]
        for counts in tagsPerHost.values {
            for tag in counts { totals[tag.name, default: 0] += tag.count }
        }
        return totals
            .map { TagCount(name: $0.key, count: $0.value) }
            .sorted {
                $0.count == $1.count
                    ? $0.name.localizedStandardCompare($1.name) == .orderedAscending
                    : $0.count > $1.count
            }
    }

    /// Which collections each machine hides, as that machine's own ids -- the
    /// form a print's `collections` are in.
    var hiddenCollectionIDs: [MoldHost.ID: Set<String>] {
        collectionsPerHost.mapValues { collections in
            Set(collections.filter { $0.hidden == true }.map(\.id))
        }
    }

    func shelf(slug: String) -> CollectionShelf? { shelves.first { $0.slug == slug } }

    func refreshOrganization(hosts: [MoldHost],
                             using backend: (MoldHost) -> any MoldBackend) async {
        await withTaskGroup(of: (MoldHost.ID, [Collection]?, [TagCount]?).self) { group in
            for host in hosts {
                let client = backend(host)
                group.addTask {
                    guard let client = client as? HTTPBackend else { return (host.id, nil, nil) }
                    // Two independent asks: a host with no organization tables
                    // answers neither, and one of them failing must not blank
                    // the other.
                    async let collections = try? await client.collections()
                    async let tags = try? await client.tags()
                    return (host.id, await collections, await tags)
                }
            }
            for await (id, collections, tags) in group {
                if let collections { collectionsPerHost[id] = collections }
                if let tags { tagsPerHost[id] = tags }
            }
        }
    }

    // MARK: - Filing

    /// Files prints into a shelf by NAME, on each machine that holds them.
    ///
    /// The name, never an id: a collection's id belongs to one machine, and a
    /// print on another machine must land in that machine's own copy of the
    /// shelf. The host resolves the name by slug and creates it if it has
    /// never seen it, which is what keeps one shelf one shelf.
    func file(_ entries: [LibraryEntry], into name: String,
              backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            let mutation = GalleryBulkMutation(
                filenames: group.map(\.print.filename),
                addToCollection: .named(name))
            do { try await client.mutate(mutation) } catch {
                failures[hostID] = "Couldn't file those into \(name)."
            }
        }
        await reloadOrganization(backend)
    }

    /// Takes prints off a shelf. By slug, for the same reason filing is by
    /// name -- it is the identity the app and every machine agree on.
    func unfile(_ entries: [LibraryEntry], from slug: String,
                backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        for (hostID, group) in Dictionary(grouping: entries, by: \.hostID) {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            let mutation = GalleryBulkMutation(
                filenames: group.map(\.print.filename),
                removeFromCollectionSlug: slug)
            try? await client.mutate(mutation)
        }
        await reloadOrganization(backend)
    }

    // MARK: - Shelf lifecycle

    /// A shelf is made on the machine you are looking at; the others get their
    /// copy the first time something is filed into it there.
    func createShelf(named name: String, on hostID: MoldHost.ID,
                     backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        guard let client = backend(hostID) as? HTTPBackend else { return }
        _ = try? await client.createCollection(name: name)
        await reloadOrganization(backend)
    }

    /// Renames every machine's copy, so the shelf does not split in two.
    func renameShelf(_ shelf: CollectionShelf, to name: String,
                     backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        for (hostID, id) in shelf.hosts {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            _ = try? await client.updateCollection(id: id, change: CollectionChange(name: name))
        }
        await reloadOrganization(backend)
    }

    /// Removes the shelf from every machine. The prints stay -- only the
    /// membership goes.
    func deleteShelf(_ shelf: CollectionShelf,
                     backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        for (hostID, id) in shelf.hosts {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            try? await client.deleteCollection(id: id)
        }
        await reloadOrganization(backend)
    }

    func setShelfHidden(_ shelf: CollectionShelf, hidden: Bool,
                        backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        for (hostID, id) in shelf.hosts {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            _ = try? await client.updateCollection(id: id,
                                                   change: CollectionChange(hidden: hidden))
        }
        await reloadOrganization(backend)
    }

    private func reloadOrganization(_ backend: (MoldHost.ID) -> (any MoldBackend)?) async {
        // Membership changed, so every machine's index is stale.
        etags.removeAll()
        for hostID in collectionsPerHost.keys {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            if let collections = try? await client.collections() {
                collectionsPerHost[hostID] = collections
            }
        }
    }
}
