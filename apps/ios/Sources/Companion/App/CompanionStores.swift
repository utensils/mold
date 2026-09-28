import MoldClient
import SwiftUI
import UIKit

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
    let notifier: Notifier
    let activities: ActivityCoordinator
    let widgets: WidgetSnapshotWriter
    /// Between `becameActive` and `enteredBackground`. Streams start only
    /// while it holds: a return to the background mid-reconcile must not
    /// leave them running behind the app's back.
    @ObservationIgnored private(set) var isForeground = false
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
        notifier = Notifier()
        activities = ActivityCoordinator(generate: generate, hosts: hosts, notifier: notifier, library: library)
        widgets = WidgetSnapshotWriter(hosts: hosts, library: library, queue: queue, generate: generate,
                                       thumbnails: thumbnails)
        notifier.favourite = { [hosts, library] id in
            // Pressed on a notification: the app may have just woken, with no
            // machine checked yet. Ask it, apply, and wait for the machine
            // to have it before iOS suspends the app again.
            guard let host = hosts.host(id.host) else { return }
            let task = UIApplication.shared.beginBackgroundTask(withName: "Favourite")
            defer { UIApplication.shared.endBackgroundTask(task) }
            await hosts.refresh(host)
            await library.reload(id.host)
            guard let entry = library.pool.first(where: { $0.everyCopy.contains { $0.id == id } }) else { return }
            library.apply(.favorite(true), to: [entry], undoable: false)
            await library.flush()
        }
        nearby = NearbyBrowser()
    }

    /// The foreground: reconcile everything, then keep it live.
    func becameActive() async {
        isForeground = true
        // The saved library first: the grid is there before any machine
        // answers, and stays there if none does.
        await library.restoreSaved()
        await hosts.refreshAll()
        guard isForeground else { return }
        hosts.startWatching()
        generate.resumeFollowing()
        async let library: Void = library.reload()
        async let queue: Void = queue.reload()
        async let models: Void = models.resume()
        _ = await (library, queue, models)
        // Gone again while those were answering: nothing stays streaming.
        guard isForeground else { return pauseStreams() }
        // What finished while the app was away (the batch on screen is
        // `resumeFollowing`'s), then what the widgets show.
        for batch in generate.ledger.batches where batch.clientBatchId != generate.activeBatch?.clientBatchId {
            await reconcile(batch)
        }
        await widgets.refresh()
        // The newest prints' thumbnails, kept for offline browsing: what is
        // already saved is skipped, so this costs nothing on a quiet day.
        if thumbnails.saving == nil { thumbnails.save(Array(self.library.pool.prefix(Self.savedAhead))) }
    }

    /// How many of the newest prints are always kept for offline.
    static let savedAhead = 200

    func enteredBackground() {
        isForeground = false
        generate.saveDraft()
        pauseStreams()
        activities.enteredBackground()
        scheduleRefresh()
        Task { await widgets.refresh() }
        nearby.stop()
    }

    private func pauseStreams() {
        hosts.stopWatching()
        models.stop()
        generate.suspendFollowing()
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
            .environment(stores.notifier)
            .environment(stores.nearby)
    }
}
