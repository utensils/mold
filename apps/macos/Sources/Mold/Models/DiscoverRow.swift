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
        guard entry.supported else { return .unsupported(pageURL: entry.pageUrl.flatMap(webPage)) }
        return .install
    }
}

/// The row's contextual menu, which is the row's own controls a second way --
/// and the only way to reach its details from the keyboard or a right click,
/// since opening them was a double-click and nothing else. Declared here,
/// beside the state it mirrors, so the menu cannot offer Install on a row
/// whose State column does not.
extension DiscoverRow {
    enum Item: Hashable {
        /// What the double-click opens.
        case details
        case install
        /// The machine cannot take this one; its own page is all there is.
        case openPage(URL)

        var title: String {
            switch self {
            case .details: "Details…"
            case .install: "Install"
            case .openPage: "Open Page"
            }
        }
    }

    /// The machine's string, as a page this Mac will open -- or nothing. It is
    /// the MACHINE that sends it, and it goes to `NSWorkspace.open`, so a
    /// `file:`, `ssh:` or another app's deep link is not a page: only http(s)
    /// with a host is. No page means no item and no cell, never one that opens
    /// something else.
    static func webPage(_ string: String) -> URL? {
        guard let url = URL(string: string), let scheme = url.scheme?.lowercased(),
              scheme == "https" || scheme == "http", url.host()?.isEmpty == false
        else { return nil }
        return url
    }

    /// In the row's own reading order: the name cell leads, the State column
    /// trails. Nothing here is ever disabled -- the same "absent, not
    /// disabled" rule the State column follows -- and nothing destructive
    /// happens to a catalog row, so there is no divider to draw.
    static func menuItems(for entry: CatalogEntry) -> [RowAction<Item>] {
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
        return items.map { RowAction(kind: $0, title: $0.title) }
    }
}
