import Foundation
import MoldClient

/// What a catalog id resolves to for the Discover table's State column --
/// pure, so a test can ask the question without a view. `installed` wins
/// over `supported`: a row this machine already has stays "Installed"
/// whatever the catalog says about it today (design S6).
enum DiscoverRow: Equatable {
    case install
    case installed
    case unsupported(pageURL: URL?)

    static func resolve(_ entry: CatalogEntry) -> DiscoverRow {
        if entry.installed { return .installed }
        guard entry.supported else { return .unsupported(pageURL: entry.pageUrl.flatMap(URL.init(string:))) }
        return .install
    }
}

/// The row's contextual menu, which is the row's own controls a second way --
/// and the only way to reach its details from the keyboard or a right click,
/// since opening them was a double-click and nothing else. Declared here,
/// beside the state it mirrors, so the menu cannot offer Install on a row
/// whose State column does not.
extension DiscoverRow {
    enum Item: Equatable, Identifiable {
        /// What the double-click opens.
        case details
        case install
        /// The machine cannot take this one; its own page is all there is.
        case openPage(URL)

        var id: String { title }

        var title: String {
            switch self {
            case .details: "Details…"
            case .install: "Install"
            case .openPage: "Open Page"
            }
        }
    }

    /// In the row's own reading order: the name cell leads, the State column
    /// trails.
    static func menuItems(for entry: CatalogEntry) -> [Item] {
        var items: [Item] = [.details]
        switch resolve(entry) {
        case .install:
            items.append(.install)
        case let .unsupported(pageURL):
            if let pageURL { items.append(.openPage(pageURL)) }
        case .installed:
            // Nothing to do here: an installed row is managed from the
            // Installed table, which has the whole install/load/delete menu.
            break
        }
        return items
    }
}
