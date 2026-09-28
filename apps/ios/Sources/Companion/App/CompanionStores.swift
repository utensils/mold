import MoldClient
import SwiftUI

/// The composition root: every store, built once, in dependency order, and
/// injected through the environment (the Mac's `AppStores`). `HostStore` is
/// first because every other store talks to machines through it. Shared by
/// every window; where a window IS (tab, sheets) is its own `AppRouter`.
@Observable
final class CompanionStores {
    let hosts: HostStore
    let library: LibraryStore
    let thumbnails: ThumbnailLoader
    let generate: GenerateController
    let queue: QueueStore
    let transfers: TransferStore
    let models: ModelStore
    let catalog: CatalogStore
    let nearby: NearbyBrowser
    let claimPairing: PairingClaimer

    init(list: HostListFile = .shared,
         credentials: any CredentialStore = KeychainCredentialStore(),
         makeBackend: @escaping (MoldHost) -> any MoldBackend = CompanionStores.http,
         claimPairing: @escaping PairingClaimer = CompanionStores.claim) {
        self.claimPairing = claimPairing
        hosts = HostStore(list: list, credentials: credentials, makeBackend: makeBackend)
        library = LibraryStore(hosts: hosts)
        thumbnails = ThumbnailLoader(hosts: hosts)
        generate = GenerateController(hosts: hosts)
        queue = QueueStore(hosts: hosts)
        transfers = TransferStore(hosts: hosts, queue: queue)
        models = ModelStore(hosts: hosts, queue: queue)
        catalog = CatalogStore(hosts: hosts)
        nearby = NearbyBrowser()
    }

    /// The foreground: reconcile everything, then keep it live.
    func becameActive() async {
        await hosts.refreshAll()
        hosts.startWatching()
        async let library: Void = library.reload()
        async let queue: Void = queue.reload()
        async let models: Void = models.resume()
        _ = await (library, queue, models)
    }

    func enteredBackground() {
        generate.saveDraft()
        hosts.stopWatching()
        models.stop()
        nearby.stop()
    }
}

extension View {
    /// Every shared store into the environment, once, at the root.
    func injecting(_ stores: CompanionStores) -> some View {
        environment(stores)
            .environment(stores.hosts)
            .environment(stores.library)
            .environment(stores.thumbnails)
            .environment(stores.generate)
            .environment(stores.queue)
            .environment(stores.transfers)
            .environment(stores.models)
            .environment(stores.catalog)
            .environment(stores.nearby)
    }
}
