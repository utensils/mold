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

    func refreshOrganization() async {
        await withTaskGroup(of: (MoldHost.ID, Result<[Collection], Error>, Result<[TagCount], Error>).self) { group in
            for host in hosts.hosts {
                let client = hosts.backend(for: host)
                group.addTask {
                    // Two independent asks: a host with no organization tables
                    // answers neither, and one of them failing must not blank
                    // the other.
                    async let collections: Result<[Collection], Error> = {
                        do { return .success(try await client.collections()) }
                        catch { return .failure(error) }
                    }()
                    async let tags: Result<[TagCount], Error> = {
                        do { return .success(try await client.tags()) }
                        catch { return .failure(error) }
                    }()
                    return (host.id, await collections, await tags)
                }
            }
            for await (id, collectionsResult, tagsResult) in group {
                if case let .success(collections) = collectionsResult { collectionsPerHost[id] = collections }
                if case let .success(tags) = tagsResult { tagsPerHost[id] = tags }
                switch (collectionsResult, tagsResult) {
                case (.success, .success):
                    // Scoped: this is `reload()`'s own passive listing, run
                    // after other actions too, and must not clear what one of
                    // THOSE just reported.
                    hosts.succeeded(on: id, doing: "read its collections and tags")
                case let (.failure(error), .failure):
                    // One host with no organization tables answers neither --
                    // only BOTH failing says the machine itself is the
                    // problem, so that is the only case worth a line.
                    hosts.report(error, on: id, doing: "read its collections and tags")
                default:
                    break
                }
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
    func file(_ entries: [LibraryEntry], into shelf: CollectionShelf) {
        apply(filing(entries, shelf, true))
    }

    /// Takes prints off a shelf.
    func unfile(_ entries: [LibraryEntry], from shelf: CollectionShelf) {
        apply(filing(entries, shelf, false))
    }

    private func filing(_ entries: [LibraryEntry], _ shelf: CollectionShelf,
                        _ filing: Bool) -> PrintEdit {
        PrintEdit.plan(
            .collection(name: shelf.name, slug: shelf.slug, filing: filing),
            over: entries,
            // Each machine's OWN id for the shelf, which is what a print's
            // `collections` are spelled in.
            collectionIDs: shelf.hosts)
    }
}
