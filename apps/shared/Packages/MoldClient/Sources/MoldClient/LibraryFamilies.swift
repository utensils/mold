import Foundation

/// One row of the Filter menu's Model Family submenu.
public struct LibraryFamilyFacet: Hashable, Sendable, Identifiable {
    public let slug: String
    public let label: String
    /// Merged prints of this family in the rows it was counted over.
    public let count: Int
    public let color: FamilyColor?
    public var id: String { slug }
}

/// The Model Family submenu: only families the Library actually holds, so
/// the menu never offers a choice that shows nothing.
public enum LibraryFamilies {
    /// Families present in `entries`, by label then slug (two slugs can share
    /// a label, and an unstable order reshuffles the menu between refreshes).
    public static func facets(in entries: [LibraryEntry]) -> [LibraryFamilyFacet] {
        var counts: [String: Int] = [:]
        for entry in entries {
            let slugs = Set(entry.everyCopy.compactMap(\.print.familySlug))
            for slug in slugs { counts[slug, default: 0] += 1 }
        }
        return counts.map { slug, count in
            LibraryFamilyFacet(slug: slug, label: Model.familyLabel(forFamily: slug), count: count,
                               color: Model.familyColor(forFamily: slug))
        }
        .sorted { lhs, rhs in
            let order = lhs.label.localizedStandardCompare(rhs.label)
            return order == .orderedSame ? lhs.slug < rhs.slug : order == .orderedAscending
        }
    }

    /// The families the query filters to; empty means every family.
    public static func selected(in query: LibraryQuery) -> Set<String> {
        Set(query.tokens.compactMap { if case let .family(slug, _) = $0 { slug } else { nil } })
    }

    /// Adds or removes one family, keeping every other token. Families widen
    /// each other (a print has one family), so several may be checked.
    public static func toggling(_ slug: String, label: String, in query: LibraryQuery) -> LibraryQuery {
        var next = query
        if selected(in: query).contains(slug) {
            next.tokens.removeAll { if case let .family(other, _) = $0 { other == slug } else { false } }
        } else {
            next.tokens.append(.family(slug: slug, name: label))
        }
        return next
    }

    /// Clears the family submenu, keeping every other token.
    public static func clearing(_ query: LibraryQuery) -> LibraryQuery {
        var next = query
        next.tokens.removeAll { if case .family = $0 { true } else { false } }
        return next
    }
}

/// The family list is a whole-library scan; derive it once per row revision
/// rather than on every toolbar redraw.
@MainActor public final class LibraryFamilyFacetCache {
    private struct Key: Equatable {
        let revision: Int
        let partition: LibraryScope
    }
    private var key: Key?
    private var facets: [LibraryFamilyFacet] = []
    public private(set) var derivations = 0
    public init() {}

    /// `partition` is the pool the rows came from (the Library or Trash):
    /// both share one row revision.
    public func facets(in entries: [LibraryEntry], revision: Int,
                       partition: LibraryScope = .all) -> [LibraryFamilyFacet] {
        let next = Key(revision: revision, partition: partition)
        if key == next { return facets }
        key = next
        facets = LibraryFamilies.facets(in: entries)
        derivations += 1
        return facets
    }
}
