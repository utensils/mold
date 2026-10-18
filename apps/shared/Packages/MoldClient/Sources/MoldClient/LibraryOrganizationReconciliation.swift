import Foundation

/// Host-independent organization names; collection IDs never cross machines.
public struct LibraryOrganizationSnapshot: Codable, Equatable, Sendable {
    public var collections: Set<String>
    public var tags: Set<String>
    public var title: String
    public var favorite: Bool
    public init(collections: Set<String> = [], tags: Set<String> = [], title: String = "", favorite: Bool = false) {
        self.collections = collections; self.tags = tags; self.title = title; self.favorite = favorite
    }
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.collections == rhs.collections && lhs.title == rhs.title && lhs.favorite == rhs.favorite
            && Set(lhs.tags.map(libraryOrganizationTagKey)) == Set(rhs.tags.map(libraryOrganizationTagKey))
    }
}

private func libraryOrganizationTagName(_ value: String) -> String {
    value.split(whereSeparator: \.isWhitespace).joined(separator: " ")
}
private func libraryOrganizationTagKey(_ value: String) -> String {
    libraryOrganizationTagName(value).lowercased()
}

/// A verified contributor owns only changes it actually made. Existing values
/// are destination-owned; observation alone never grants removal authority.
public struct LibraryOrganizationState: Codable, Equatable, Sendable {
    public var version = 1
    public var observed: LibraryOrganizationSnapshot
    public var ownedCollections: Set<String> = []
    public var ownedTags: Set<String> = []
    public var suppressedCollections: Set<String> = []
    public var suppressedTags: Set<String> = []
    public var appliedTitle: String?
    public var appliedFavorite: Bool?
    public var titleOverridden: Bool?
    public var favoriteOverridden: Bool?
    public init(observed: LibraryOrganizationSnapshot) { self.observed = observed }
}

public enum LibraryOrganizationReconciliation {
    public struct Plan: Sendable {
        public let result: LibraryOrganizationSnapshot
        public let state: LibraryOrganizationState
    }

    public static func plan(source: LibraryOrganizationSnapshot, destination: LibraryOrganizationSnapshot,
                            previous: LibraryOrganizationState?, hidden: Set<String>, newCopy: Bool = false,
                            explicitCopy: Bool = false, otherClaims: LibraryOrganizationSnapshot = .init(),
                            scalarAuthority: Bool = true) -> Plan {
        let previous = previous.flatMap { $0.version == 1 ? $0 : nil }
        var state = previous ?? LibraryOrganizationState(observed: source)
        var result = destination
        func membership(_ incoming: Set<String>, _ current: Set<String>, _ observed: Set<String>,
                        owned: inout Set<String>, suppressed: inout Set<String>, mandatory: Set<String>,
                        other: Set<String>) -> Set<String> {
            var values = current
            let removedLocally = owned.intersection(incoming).subtracting(current).subtracting(mandatory)
            suppressed.formUnion(removedLocally)
            // A deliberate ordinary removal relinquishes ownership. If the
            // user later adds that membership locally, it is their claim.
            owned.subtract(removedLocally)
            // A real source remove/re-add starts a new contribution.
            suppressed.formIntersection(incoming)
            for value in owned.subtracting(incoming) where !other.contains(value) { values.remove(value) }
            owned.formIntersection(incoming)
            let additions = incoming.subtracting(current)
            for value in additions {
                let allowed = mandatory.contains(value) || explicitCopy || newCopy
                    || (previous != nil && !observed.contains(value) && !suppressed.contains(value))
                if allowed { values.insert(value); owned.insert(value); suppressed.remove(value) }
                else { suppressed.insert(value) }
            }
            return values
        }
        result.collections = membership(source.collections, destination.collections, state.observed.collections,
            owned: &state.ownedCollections, suppressed: &state.suppressedCollections,
            mandatory: hidden, other: otherClaims.collections)
        // Tags use the server's whitespace/case identity, while existing
        // destination spelling remains a presentation choice on that machine.
        func tagNames(_ tags: Set<String>) -> [String: String] {
            Dictionary(tags.sorted().map { (libraryOrganizationTagKey($0), libraryOrganizationTagName($0)) },
                uniquingKeysWith: { first, _ in first })
        }
        let incomingTags = tagNames(source.tags), existingTags = tagNames(destination.tags)
        state.ownedTags = Set(state.ownedTags.map(libraryOrganizationTagKey))
        state.suppressedTags = Set(state.suppressedTags.map(libraryOrganizationTagKey))
        let desiredTagKeys = membership(Set(incomingTags.keys), Set(existingTags.keys), Set(state.observed.tags.map(libraryOrganizationTagKey)),
            owned: &state.ownedTags, suppressed: &state.suppressedTags, mandatory: [],
            other: Set(otherClaims.tags.map(libraryOrganizationTagKey)))
        result.tags = Set(desiredTagKeys.compactMap { existingTags[$0] ?? incomingTags[$0] })
        if previous == nil && !newCopy && !explicitCopy {
            if destination.title != source.title { state.titleOverridden = true }
            if destination.favorite != source.favorite { state.favoriteOverridden = true }
        }
        if scalarAuthority {
            if let applied = state.appliedTitle {
                if destination.title == applied { result.title = source.title; state.appliedTitle = source.title }
                else { state.appliedTitle = nil; state.titleOverridden = true }
            } else if state.titleOverridden != true && destination.title.isEmpty && !source.title.isEmpty
                && ((previous == nil && (newCopy || explicitCopy)) || (previous != nil && state.observed.title.isEmpty)) {
                result.title = source.title; state.appliedTitle = source.title
            }
            if let applied = state.appliedFavorite {
                if destination.favorite == applied { result.favorite = source.favorite; state.appliedFavorite = source.favorite }
                else { state.appliedFavorite = nil; state.favoriteOverridden = true }
            } else if state.favoriteOverridden != true && !destination.favorite && source.favorite
                && ((previous == nil && (newCopy || explicitCopy)) || (previous != nil && !state.observed.favorite)) {
                result.favorite = true; state.appliedFavorite = true
            }
        }
        state.observed = source
        return Plan(result: result, state: state)
    }
}
