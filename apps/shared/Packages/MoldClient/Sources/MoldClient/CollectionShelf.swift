import Foundation

/// One collection as the Library shows it: a shelf, across every machine.
///
/// A collection's id is a UUID in ONE machine's database, but a person filing
/// pictures under "Smurf Village" means one shelf whatever machine rendered
/// them. mold's own rule is that collections merge by `slug`, and the server
/// is the authority for the slug -- this never computes one.
public struct CollectionShelf: Identifiable, Hashable, Sendable {
    public let slug: String
    /// The spelling to show. Machines can disagree; see `merge`.
    public let name: String
    /// Every machine's count, added up.
    public let count: Int
    /// Omitted from the default grid. True when ANY machine holding it hides
    /// it, which is `mergeCollectionsAcrossHosts`'s rule
    /// (`studio/lib/libraryOrganization.ts:167, 205`).
    ///
    /// The hiding mutation FANS OUT to every copy, so a mixed state is an
    /// edit that half-landed rather than a considered disagreement, and
    /// answering "hidden" answers with the intent instead of with the
    /// failure. This app used to require every machine to agree -- defensible
    /// read alone, but two different rules for the same shelf across a fleet
    /// is not, and someone who hid "Drafts" on one machine was told by one
    /// app that it was hidden and by another that it was not.
    public let hidden: Bool
    /// The collection's id on each machine that has it. Needed to open a
    /// shelf and to remove a print from one; never used to ADD, because an id
    /// is only ever right on one machine.
    public let hosts: [MoldHost.ID: String]

    public var id: String { slug }

    /// How many of these prints are actually in front of you.
    ///
    /// NOT `count`, which is the host's own number: that includes trashed
    /// members, which keep their membership until they are purged, and it can
    /// outlive the prints entirely. A sidebar badge is a promise about what
    /// opening the row shows, so it is counted from the index the grid draws
    /// -- and a print counts only under the id ITS OWN machine gave the shelf.
    public func count(in entries: [LibraryEntry]) -> Int {
        // Any copy filed on its own machine counts the print once -- the
        // same rule the collection token filters by.
        entries.count { entry in
            entry.everyCopy.contains { copy in
                hosts[copy.hostID].map { copy.print.collectionList.contains($0) } ?? false
            }
        }
    }

    /// Folds every machine's collections into one list of shelves.
    public static func merge(_ perHost: [MoldHost.ID: [Collection]]) -> [CollectionShelf] {
        var bySlug: [String: [(host: MoldHost.ID, collection: Collection)]] = [:]
        for (host, collections) in perHost {
            for collection in collections {
                bySlug[collection.slug, default: []].append((host, collection))
            }
        }
        return bySlug.values.map(shelf(from:)).sorted { lhs, rhs in
            // Sorted the way a person reads a sidebar, not the way bytes
            // compare: "Éclair" belongs between "apple" and "zebra".
            lhs.name.localizedStandardCompare(rhs.name) == .orderedAscending
        }
    }

    private static func shelf(from members: [(host: MoldHost.ID, collection: Collection)])
        -> CollectionShelf
    {
        // The spelling shown is the one on the machine holding the most of it
        // -- the copy actually in use. Ties break alphabetically so the
        // sidebar does not reorder itself between launches.
        let leader = members.max { lhs, rhs in
            let (left, right) = (lhs.collection.count ?? 0, rhs.collection.count ?? 0)
            if left != right { return left < right }
            return lhs.collection.name.localizedStandardCompare(rhs.collection.name)
                == .orderedDescending
        }
        return CollectionShelf(
            slug: members[0].collection.slug,
            name: leader?.collection.name ?? members[0].collection.name,
            count: members.reduce(0) { $0 + ($1.collection.count ?? 0) },
            hidden: members.contains { $0.collection.hidden == true },
            hosts: Dictionary(members.map { ($0.host, $0.collection.id) }) { first, _ in first }
        )
    }
}
