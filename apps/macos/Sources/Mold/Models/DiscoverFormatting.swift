import Foundation
import MoldClient

// The pure functions `DiscoverTable` renders from -- split out purely for
// size, the same reason `ModelsPane+Grouping.swift` exists beside the view
// it feeds.
extension DiscoverTable {
    /// Exactly one row per entry, de-duplicated by id -- never a placeholder
    /// for "load more" or the provider note baked in beside the data (design
    /// S6b test 3: with nothing more to load, this returns exactly
    /// `entries.count` rows).
    static func rows(for entries: [CatalogEntry]) -> [CatalogEntry] {
        var seen = Set<CatalogEntry.ID>()
        return entries.filter { seen.insert($0.id).inserted }
    }

    /// "Civitai didn't answer." -- one provider failing beside rows the
    /// other did return, never a banner (design S6 test 2).
    static func providerNote(_ errors: [CatalogProviderError]) -> String? {
        guard !errors.isEmpty else { return nil }
        let names = errors.map { $0.source.capitalized }.joined(separator: ", ")
        return "\(names) didn't answer."
    }

    static func sizeText(_ entry: CatalogEntry) -> String {
        guard let bytes = entry.sizeBytes else { return "—" }
        return Int64(bytes).formatted(.byteCount(style: .file))
    }

    static func downloadsText(_ entry: CatalogEntry) -> String {
        let downloads = entry.downloadCount.formatted(.number.notation(.compactName))
        guard let rating = entry.rating else { return downloads }
        return "\(downloads) · \(rating.formatted(.number.precision(.fractionLength(1))))★"
    }

    /// `false` draws nothing -- never an affirmative "Safe" claim for the
    /// ordinary case (design S6 test 7).
    static func nsfwBadge(_ entry: CatalogEntry) -> String? { entry.nsfw ? "NSFW" : nil }
}
