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
    /// Omitted from the default grid. True only when EVERY machine holding it
    /// hides it -- one machine still showing it means those prints belong in
    /// All Prints.
    public let hidden: Bool
    /// The collection's id on each machine that has it. Needed to open a
    /// shelf and to remove a print from one; never used to ADD, because an id
    /// is only ever right on one machine.
    public let hosts: [MoldHost.ID: String]

    public var id: String { slug }

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
            hidden: members.allSatisfy { $0.collection.hidden == true },
            hosts: Dictionary(members.map { ($0.host, $0.collection.id) }) { first, _ in first }
        )
    }
}
