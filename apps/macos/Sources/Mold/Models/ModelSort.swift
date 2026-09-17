import MoldClient

/// Which column drives the Installed table's order, and which way -- pure so
/// a test can ask the table's own question without a view.
struct ModelSort: Equatable {
    enum Column: String, CaseIterable {
        case model, variant, tradeoff, size, state
    }

    var column: Column = .model
    var ascending = true
}

extension ModelSort {
    /// A stable order: two rows that compare equal on the chosen column keep
    /// the order they arrived in, so nothing shuffles between renders.
    static func sorted(_ rows: [Model], by sort: ModelSort) -> [Model] {
        rows.enumerated()
            .sorted { lhs, rhs in
                let byColumn = compare(lhs.element, rhs.element, column: sort.column)
                if byColumn != 0 { return sort.ascending ? byColumn < 0 : byColumn > 0 }
                return lhs.offset < rhs.offset
            }
            .map(\.element)
    }

    /// Grouped by the server's OWN family string (decision 9, M5), families
    /// sorted, and each group's rows in the same order the table shows.
    static func grouped(_ rows: [Model], by sort: ModelSort) -> [(family: String, rows: [Model])] {
        Dictionary(grouping: rows, by: \.family)
            .map { (family: $0.key, rows: sorted($0.value, by: sort)) }
            .sorted { $0.family.localizedStandardCompare($1.family) == .orderedAscending }
    }

    private static func compare(_ a: Model, _ b: Model, column: Column) -> Int {
        switch column {
        case .model: compareStrings(a.sortHeadline, b.sortHeadline)
        case .variant: compareStrings(a.sortVariant, b.sortVariant)
        case .tradeoff: compareStrings(a.sortTradeOff, b.sortTradeOff)
        case .size: compareInts(a.sortSize, b.sortSize)
        case .state: compareInts(a.sortState, b.sortState)
        }
    }

    private static func compareStrings(_ a: String, _ b: String) -> Int {
        switch a.localizedStandardCompare(b) {
        case .orderedAscending: -1
        case .orderedDescending: 1
        case .orderedSame: 0
        }
    }

    private static func compareInts(_ a: Int, _ b: Int) -> Int {
        a == b ? 0 : (a < b ? -1 : 1)
    }
}

extension Model {
    // `nonisolated`: `Table`'s `sortOrder` wants `KeyPathComparator<Model>`,
    // and forming a key path to a MainActor-isolated property -- the app
    // module's default isolation, which an extension inherits -- does not
    // typecheck (`KeyPath` itself then fails the `Sendable` `SortComparator`
    // needs). `Model` is a plain value type with no actor of its own, so
    // there is nothing unsafe about reading these off the main actor.

    /// `baseTitle` then `headline`, tab-separated so the base name always
    /// wins the comparison: variants of one model stay adjacent under this
    /// column whatever their trade-off or size say. Only the Model column
    /// reads this -- every other column sorts by its own value alone.
    nonisolated var sortHeadline: String { "\(baseTitle)\t\(headline)" }
    nonisolated var sortVariant: String { tag ?? "" }
    nonisolated var sortTradeOff: String { tradeOff ?? "" }
    /// The honest number for the row whatever its state (design fact 3, M5):
    /// what is on disk, or failing that what it would cost to fetch. Never
    /// the machine's own `models_disk` figure -- that is the footer's alone.
    nonisolated var sortSize: Int { diskUsageBytes ?? remainingDownloadBytes ?? 0 }
    nonisolated var sortState: Int { installState.sortRank }
}
