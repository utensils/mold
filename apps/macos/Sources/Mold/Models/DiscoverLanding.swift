import Foundation
import MoldClient

/// The host's manifest is the curated starting point, including models it owns.
enum DiscoverLanding {
    static func showsFeatured(text: String, family: String?, browsingCatalog: Bool) -> Bool {
        text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && family == nil && !browsingCatalog
    }

    static func featured(_ models: [Model]) -> [Model] {
        models.filter { $0.isGenerator && $0.matchesDiscovery(CatalogQuery(includeNSFW: false)) }
    }
}
