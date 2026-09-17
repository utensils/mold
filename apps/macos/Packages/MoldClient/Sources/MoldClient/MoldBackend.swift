import Foundation

/// The one seam between the UI and wherever generation actually happens.
///
/// `HTTPBackend` is the only conformance, and it covers two of the three
/// intended cases on its own: a remote `mold serve` over the network, and
/// mold's own Rust engine running in-process on loopback. `mold-server` speaks
/// ONE wire contract whether it is bound to a Tailscale address or to
/// `127.0.0.1`, so embedding the engine changes a URL, not this protocol. The
/// third is a fake for previews and tests.
///
/// It carries every route the app calls. It used to carry thirteen, and the
/// rest were reached by downcasting to `HTTPBackend` at the call site -- which
/// in `QueueStore` meant an optional chain where a failed cast was `nil`,
/// nothing threw, and a cancel that never left the machine reported success.
///
/// Split into one protocol per concern across `MoldBackend+*.swift`: M1.5 S2
/// kept every requirement in this one file while it fit under the 150-line
/// file lint, but M5 S1b's models/catalog/downloads growth pushed the total
/// well past that. Each concern is now its own small protocol -- one file,
/// one `MARK`'s worth of routes -- and this file is just the composition.
/// `any MoldBackend` and every conformance (`HTTPBackend`, `FakeBackend`) are
/// unchanged: this is a pure refactor of where the requirements live, not a
/// new seam or a new call convention.
public protocol MoldBackend:
    MoldStatusBackend, MoldGenerationBackend, MoldCreateBackend, MoldQueueBackend,
    MoldDownloadsBackend, MoldLicencesBackend, MoldMachinesBackend, MoldGalleryBackend,
    MoldOrganizationBackend, MoldStreamsBackend, MoldModelsBackend, MoldCatalogBackend,
    MoldConfigBackend, MoldChainBackend, MoldUpscaleBackend
{}
