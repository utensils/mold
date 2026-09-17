import MoldClient

// The table's own filtering and ordering, split out from the view (S3): pure
// so a test can ask the whole question -- narrow, sort, attach a refusal --
// with no `Table` in sight, the same split `DiscoverRow.resolve` uses for one
// cell instead of a whole row.
extension AdvancedSettings {
    struct Row: Identifiable, Equatable {
        let entry: ConfigEntry
        /// The sentence a machine most recently gave for THIS key, or `nil`.
        /// Never folded into a fleet-wide banner -- a 422 on `expand.max_tokens`
        /// is about that row, nowhere else (`ConfigStore.recordFailure`).
        let refusal: String?
        var id: String { entry.key }
    }

    /// Every row for the table: folded key search (the same folding
    /// `LibraryEntry.searchKey` uses, so a search behaves the way the
    /// Library's does), sorted by key with `models.*` last -- sixteen
    /// per-model rows in the middle of the alphabet would bury everything
    /// else (design decision, S3).
    static func rows(_ listing: ConfigListing?, query: String, refusals: [String: String]) -> [Row] {
        guard let listing else { return [] }
        let tokens = LibraryEntry.fold(query).split(separator: " ")
        return listing.entries
            .filter { entry in
                guard !tokens.isEmpty else { return true }
                let key = LibraryEntry.fold(entry.key)
                return tokens.allSatisfy { key.contains($0) }
            }
            .sorted(by: keyOrder)
            .map { Row(entry: $0, refusal: refusals[$0.key]) }
    }

    private static func keyOrder(_ lhs: ConfigEntry, _ rhs: ConfigEntry) -> Bool {
        let lhsModel = lhs.key.hasPrefix("models.")
        let rhsModel = rhs.key.hasPrefix("models.")
        if lhsModel != rhsModel { return rhsModel }
        return lhs.key < rhs.key
    }

    /// "showing of total", never a bare count -- a filter that matches
    /// nothing reads as "0 of 63", not as an empty machine (design S3, the
    /// bug M5 S6 UAT found in Discover, not repeated here).
    static func subtitle(showing: Int, total: Int) -> String {
        "\(showing) of \(total)"
    }
}
