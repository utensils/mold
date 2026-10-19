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
                              kind: SourceMediaFilter, search: String, queryKinds: Set<PrintKind> = [], sort: LibrarySort = .newest, locale: Locale = .current) -> [Self] {
        let term = search.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: locale)
        let visible: [Self] = sources.compactMap { source in
            let usages = source.usages.filter { row in
                guard allowed[row.owner.id] == row.authority, kind.includes(row.member) else { return false }
                guard !queryKinds.isEmpty else { return true }
                let contentKind = row.member.contentKind
                return queryKinds.contains { kind in
                    switch kind { case .picture: contentKind == "image"; case .clip: contentKind == "video"; case .mesh: contentKind == "mesh" }
                }
            }
            guard !usages.isEmpty else { return nil }
            let key = usages.count == source.usages.count ? source.searchKey : usages.map {
                "\($0.member.sourceLabel) \($0.member.displayName) \($0.owner.print.displayName) \($0.owner.hostName)"
            }.joined(separator: " ").folding(options: [.caseInsensitive, .diacriticInsensitive], locale: locale)
            guard term.isEmpty || key.contains(term) else { return nil }
            return Self(id: source.id, usages: usages)
        }
        // Sources have their own names and byte sizes. Dates describe their
        // matching owning prints; derive keys once rather than per comparison.
        return visible.map { source in
            let times = source.usages.map { $0.owner.print.timestamp }
            let time = sort == .oldest ? times.min() ?? 0 : times.max() ?? 0
            let size = source.usages.map { $0.member.sizeBytes }.max() ?? 0
            let name = source.usages.map { $0.member.displayName }.min { left, right in
                let order = left.localizedStandardCompare(right)
                return order == .orderedSame ? left < right : order == .orderedAscending
            } ?? ""
            return (source: source, time: time, size: size, name: name)
        }.sorted { left, right in
            switch sort {
            case .newest where left.time != right.time: return left.time > right.time
            case .oldest where left.time != right.time: return left.time < right.time
            case .largest where left.size != right.size: return left.size > right.size
            default: break
            }
            let names = left.name.localizedStandardCompare(right.name)
            if names != .orderedSame { return names == .orderedAscending }
            return left.source.id < right.source.id
        }.map(\.source)
    }
}
