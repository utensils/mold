import Foundation
import MoldClient
import Testing

@testable import Mold

/// The two stores that used to depend on a pane being on screen: the library
/// only started listening when the Library pane appeared, and a download
/// stream was never reconciled against the machine list at all.
///
/// Split from `HostStoreLifecycleTests` by subject rather than by size --
/// those pin what `HostStore` decides, these pin what hangs off it.
@MainActor
struct StoreLifecycleTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// The pane used to call `listen()`. A print made while Generate was
    /// showing therefore reached nobody, and the Library only caught up when
    /// somebody hit ⌘R.
    @Test func aLibraryStoreIsListeningBeforeAnyPaneAppears() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.serverStatus = FakeFixtures.serverStatus()
        backend.capabilityBlock = FakeFixtures.capabilities(events: true)
        backend.exportBlock = FakeFixtures.exportOptions()
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        // Built the way the composition root builds it, and nothing else.
        let library = LibraryStore(hosts: hosts)

        await hosts.refresh(plato)
        // Explicit, so these isolate their own defect rather than riding
        // on whether `refresh` reconciles yet.
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }
        backend.emit(.gallery(.added(filename: "a.png", row: FakeFixtures.print("a.png"))))
        await settle { !library.items.isEmpty }

        #expect(library.items.map(\.print.filename) == ["a.png"])
    }

    /// **Fails today**: `stopWatching()` has no callers and the streams are
    /// never checked against the machine list, so a removed machine keeps a
    /// live connection for the rest of the launch.
    @Test func aDownloadStreamStopsWhenItsMachineIsRemoved() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.downloadTicket = FakeFixtures.downloadTicket("job-1")
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let licenses = LicenseStore(hosts: hosts)
        let downloads = DownloadStore(hosts: hosts, licenses: licenses)

        await downloads.install("flux-dev:q4", on: plato)
        await settle { backend.callCount("downloadEvents") == 1 }
        #expect(downloads.streams[plato.id] != nil)

        hosts.remove(plato)
        downloads.reconcile()

        #expect(downloads.streams[plato.id] == nil)
        await settle { backend.downloadStreamEnded }
        #expect(backend.downloadStreamEnded)
    }
}
