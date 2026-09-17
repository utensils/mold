import MoldClient
import Foundation

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
