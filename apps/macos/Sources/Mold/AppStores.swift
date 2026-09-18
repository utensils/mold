import AppKit
import MoldClient
import SwiftUI

/// The store graph, composed once.
///
/// This is the only type allowed to know which backend the app is built on.
/// `make lint` fails the build if `MoldClient`'s concrete backends are
/// constructed anywhere else -- that single rule is what keeps the UI honest
/// about its dependencies, and it is why swapping in an in-process Rust engine
/// later is a change to this file rather than to the app.
///
/// It lives apart from `MoldApp` because `@State` initializers cannot
/// reference each other, so composition has to be explicit and ORDERED, and
/// that order is a page of code on its own. `MoldApp` is left as what it
/// should be: the scenes, the menu bar, and one `@State` holding this.
///
/// ADDING A STORE: one `let` in the list below, one line in `init` in
/// dependency order, and one `.environment()` in `AppStores+Environment.swift`.
@MainActor
final class AppStores {
    let engine: MoldEngine
    let hosts: HostStore
    let library: LibraryStore
    let libraryNavigation = LibraryNavigation()
    let thumbnails = ThumbnailCache()
    let materializer = PrintMaterializer()
    let models: ModelStore
    let generate: GenerateController
    let queue: QueueStore
    let transfers: TransferStore
    let licenses: LicenseStore
    let downloads: DownloadStore
    let catalog: CatalogStore
    let machines: MachineStore
    let pairing: PairingStore
    let promptHistory: PromptHistoryStore
    let modelDefaults: ConfigStore
    let adapters: LoraStore
    let landedPrints: LandedPrints
    let upscales: UpscaleStore
    let activity: ActivityStore
    let notifications: MoldNotifications
    let heartbeat: HostHeartbeat

    /// `HostStore` first, since every other store is built by asking it which
    /// machines exist.
    init() {
        // FIRST, before a single store exists and so before URLSession or
        // `HostStore`'s polling has a thread of its own (review 05-M4).
        engine = MoldEngine.bootstrapped()
        hosts = HostStore(hosts: HostStore.seededHosts())
        engine.dropsItsMachine(from: hosts)
        library = LibraryStore(hosts: hosts)
        models = ModelStore(hosts: hosts)
        queue = QueueStore(hosts: hosts)
        transfers = TransferStore(hosts: hosts, queue: queue)
        licenses = LicenseStore(hosts: hosts)
        downloads = DownloadStore(hosts: hosts, licenses: licenses)
        catalog = CatalogStore(hosts: hosts)
        modelDefaults = ConfigStore(hosts: hosts)
        promptHistory = PromptHistoryStore(hosts: hosts)
        generate = GenerateController(hosts: hosts, defaults: modelDefaults)
        machines = MachineStore(hosts: hosts)
        pairing = PairingStore(hosts: hosts)
        adapters = LoraStore(hosts: hosts)
        landedPrints = LandedPrints(hosts: hosts)
        upscales = UpscaleStore(hosts: hosts, models: models, library: library)
        activity = ActivityStore(hosts: hosts)
        // NEW STORES GO HERE -- after the stores they depend on, before the
        // two below, which are built FROM the others.
        notifications = MoldNotifications(
            landedPrints: landedPrints, queue: queue, hosts: hosts, library: library)
        heartbeat = HostHeartbeat(hosts: hosts, queue: queue)
    }
}
