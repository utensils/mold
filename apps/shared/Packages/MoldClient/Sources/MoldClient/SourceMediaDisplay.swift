import Foundation

/// A filtered set of exact usages; presentation stays in each native app.
public struct SourceMediaDisplay: Identifiable, Sendable {
    public let id: String
    public let usages: [SourceMediaCatalog.Row]
    public init(id: String, usages: [SourceMediaCatalog.Row]) { self.id = id; self.usages = usages }

    @MainActor
    public func visibleUsages(access: SourceMediaAccessIndex.Snapshot, catalog: SourceMediaCatalog,
                              gallery: Bool) -> [SourceMediaCatalog.Row] {
        usages.filter { access.allows($0.authority, detail: !gallery) && (!gallery || catalog.contains($0)) }
    }

    public func preferredUsage(isReachable: (UUID) -> Bool) -> SourceMediaCatalog.Row? {
        usages.first { $0.member.contentKind == "image" && isReachable($0.owner.hostID) }
            ?? usages.first { isReachable($0.owner.hostID) }
            ?? usages.first { $0.member.contentKind == "image" }
            ?? usages.first
    }

    public static func filter(_ sources: [SourceMediaCatalog.Source], allowed: [PrintID: SourceMediaAuthority],
                              kind: SourceMediaFilter, search: String, locale: Locale = .current) -> [Self] {
        let term = search.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: locale)
        return sources.compactMap { source in
            let usages = source.usages.filter { allowed[$0.owner.id] == $0.authority && kind.includes($0.member) }
            guard !usages.isEmpty else { return nil }
            let key = usages.count == source.usages.count ? source.searchKey : usages.map {
                "\($0.member.sourceLabel) \($0.member.displayName) \($0.owner.print.displayName) \($0.owner.hostName)"
            }.joined(separator: " ").folding(options: [.caseInsensitive, .diacriticInsensitive], locale: locale)
            guard term.isEmpty || key.contains(term) else { return nil }
            return Self(id: source.id, usages: usages)
        }
    }
}
