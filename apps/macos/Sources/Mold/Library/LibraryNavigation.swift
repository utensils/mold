import Foundation
import MoldClient
import SwiftUI

/// Where you are in the library, and what you are looking for.
///
/// Held outside the pane because the SIDEBAR drives it: picking a collection
/// there is the same act as picking All Prints, and both have to survive the
/// pane being torn down when you go to Generate and come back.
@MainActor
@Observable
final class LibraryNavigation {
    var scope: LibraryScope = .all {
        didSet { remember() }
    }
    var query = LibraryQuery()
    /// Thumbnail edge. A view setting, but one people expect to persist.
    var edge: CGFloat
    /// A print a notification click named -- `LibraryPane` opens it once on
    /// appearance (or on this changing under an already-open pane) and clears
    /// it right back. Transient on purpose: nothing about a click survives a
    /// relaunch the way `scope` and `edge` do.
    var reveal: PrintID?

    private static let scopeKey = "libraryScope"
    private static let edgeKey = "libraryEdge"

    init(defaults: UserDefaults = AppStorageSuite.defaults) {
        self.defaults = defaults
        self.edge = defaults.object(forKey: Self.edgeKey) as? Double ?? 132
        if let data = defaults.data(forKey: Self.scopeKey),
           let saved = try? MoldJSON.localDecoder.decode(LibraryScope.self, from: data) {
            scope = saved
        }
    }

    private let defaults: UserDefaults

    /// A collection that no longer exists -- deleted, or on a machine that is
    /// gone -- lands you back on All Prints rather than on an empty shelf with
    /// no explanation.
    func reconcile(with shelves: [CollectionShelf]) {
        guard let slug = scope.collectionSlug else { return }
        if !shelves.contains(where: { $0.slug == slug }) { scope = .all }
    }

    func rememberEdge() { defaults.set(Double(edge), forKey: Self.edgeKey) }

    private func remember() {
        defaults.set(try? MoldJSON.localEncoder.encode(scope), forKey: Self.scopeKey)
    }
}
