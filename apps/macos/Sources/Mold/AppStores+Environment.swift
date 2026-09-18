import SwiftUI

/// Putting the store graph into the environment.
///
/// One `.environment()` per store, and SwiftUI registers each BY TYPE -- so a
/// store left out of this list is not a compile error anywhere, it is an
/// `@Environment` crash the first time the pane that wants it is opened. That
/// is the whole reason both lists live beside each other in one place.
extension View {
    /// Everything the main window's panes read.
    func moldEnvironment(_ stores: AppStores) -> some View {
        self
            .environment(stores.hosts)
            .environment(stores.library)
            .environment(stores.libraryNavigation)
            .environment(stores.thumbnails)
            .environment(stores.materializer)
            .environment(stores.models)
            .environment(stores.generate)
            .environment(stores.queue)
            .environment(stores.transfers)
            .environment(stores.licenses)
            .environment(stores.downloads)
            .environment(stores.catalog)
            .environment(stores.machines)
            .environment(stores.pairing)
            .environment(stores.promptHistory)
            .environment(stores.modelDefaults)
            .environment(stores.adapters)
            .environment(stores.landedPrints)
            .environment(stores.engine)
            .environment(stores.upscales)
            .environment(stores.activity)
            .environment(stores.reuse)
            .environment(stores.expansions)
        // A NEW STORE'S `.environment()` GOES HERE.
    }

    /// And the smaller set Settings reads. Deliberately its own list rather
    /// than the whole graph: a Settings pane that wants a store nothing in
    /// this list provides is a pane reaching outside its own scope, and the
    /// crash says so immediately.
    func moldSettingsEnvironment(_ stores: AppStores) -> some View {
        self
            .environment(stores.hosts)
            .environment(stores.engine)
            .environment(stores.materializer)
            // Settings ▸ Empty Now empties BOTH caches.
            .environment(stores.thumbnails)
            .environment(stores.catalog)
            .environment(stores.modelDefaults)
            .environment(stores.library)
    }
}
