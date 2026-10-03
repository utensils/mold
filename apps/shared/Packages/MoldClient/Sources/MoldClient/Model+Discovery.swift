import Foundation

public extension Model {
    /// Curated models are the host's manifest inventory, not provider search
    /// results. Search all words across their human title and stable identity.
    func matchesDiscovery(_ query: CatalogQuery) -> Bool {
        guard !Self.isCatalogName(name) else { return false }
        if let family = query.family, family != self.family { return false }
        if let source = query.source {
            guard source == "hf", hfRepo != nil else { return false }
        }
        let haystack = [headline, name, family, description, hfRepo ?? ""].joined(separator: " ")
        return (query.text ?? "").split(whereSeparator: { $0.isWhitespace }).allSatisfy {
            haystack.range(of: String($0), options: [.caseInsensitive, .diacriticInsensitive]) != nil
        }
    }
}
