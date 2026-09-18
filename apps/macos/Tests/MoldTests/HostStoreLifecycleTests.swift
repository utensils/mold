import Foundation
import MoldClient
import Testing

@testable import Mold

/// Who owns an app-lifetime connection.
///
/// It used to be one pane's view lifecycle: the Library's `.onChange` was the
/// only thing that reconciled event streams, so a machine added while Generate
/// was showing went unwatched, and every sidebar switch tore the streams down
/// and built them again. `HostStore` is the one object that knows which
/// machines exist -- these pin that it is also the one that decides what is
/// being watched.
@MainActor
struct HostStoreLifecycleTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// A machine that answers, advertises `/api/events`, and lists prints.
    private func fake(for host: MoldHost, instanceId: String? = nil) -> FakeBackend {
        let fake = FakeBackend(host: host)
        fake.serverStatus = FakeFixtures.serverStatus(instanceId: instanceId)
        fake.capabilityBlock = FakeFixtures.capabilities(events: true)
        fake.exportBlock = FakeFixtures.exportOptions()
        return fake
    }

    /// **Fails today**: nothing reconciles from `refresh`, and `wantsEvents`
    /// reads `.checking` as down -- so a machine being asked about loses the
    /// stream it already had, and every event until the answer lands with it.
    @Test func checkingAMachineThatIsAlreadyUpKeepsItsWatcher() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }

        await hosts.refresh(workstation)
        await settle { backend.callCount("events") == 1 }
        #expect(hosts.watchers[workstation.id] != nil)

        // Asking the machine again is not a reason to hang up on it.
        hosts.reachability[workstation.id] = .checking
        hosts.reconcileEventStreams()
        #expect(hosts.watchers[workstation.id] != nil)

        await hosts.refresh(workstation)
        #expect(backend.callCount("events") == 1)
    }

    /// **Fails today**: `remove` clears two of the per-host dictionaries and
    /// leaves the watcher, the export formats, the fleet identity and the
    /// machine's failure banner behind.
    @Test func forgettingAMachineClearsEverythingItLeftBehind() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }

        await hosts.refresh(workstation)
        // Explicit, so these isolate their own defect rather than riding
        // on whether `refresh` reconciles yet.
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }
        backend.emit(.authority(instanceID: "run-1"))
        await settle { hosts.instanceIDs[workstation.id] != nil }
        hosts.report(MoldClientError.unreachable("boom"), on: workstation.id, doing: "list its prints")

        hosts.remove(workstation)

        #expect(hosts.reachability[workstation.id] == nil)
        #expect(hosts.capabilities[workstation.id] == nil)
        #expect(hosts.exportOptions[workstation.id] == nil)
        #expect(hosts.instanceIDs[workstation.id] == nil)
        #expect(hosts.watchers[workstation.id] == nil)
        #expect(hosts.failures.isEmpty)
    }

    /// **Fails today**: a machine whose address changed keeps the old box's
    /// fleet identity, so the next authority frame matches and the "this is a
    /// different library" repair never fires.
    @Test func changingAMachinesAddressForgetsTheFleetIdentityItHad() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }

        await hosts.refresh(workstation)
        // Explicit, so these isolate their own defect rather than riding
        // on whether `refresh` reconciles yet.
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }
        backend.emit(.authority(instanceID: "run-1"))
        await settle { hosts.instanceIDs[workstation.id] != nil }

        hosts.update(MoldHost(id: workstation.id, name: "workstation",
                              baseURL: URL(string: "http://workstation-2:7680")!))

        #expect(hosts.instanceIDs[workstation.id] == nil)
        #expect(hosts.instanceID(of: workstation.id) == nil)
    }

    /// One answer, wherever it came from: the stream's opening frame when
    /// there has been one, and what `/api/status` said until then.
    @Test func theInstanceIdComesFromTheAuthorityFrameThenTheStatus() async {
        let workstation = machine()
        let backend = fake(for: workstation, instanceId: "from-status")
        let hosts = HostStore(hosts: [workstation]) { _ in backend }

        await hosts.refresh(workstation)
        #expect(hosts.instanceID(of: workstation.id) == "from-status")

        // Explicit, so these isolate their own defect rather than riding
        // on whether `refresh` reconciles yet.
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }
        backend.emit(.authority(instanceID: "from-the-stream"))
        await settle { hosts.instanceIDs[workstation.id] != nil }

        #expect(hosts.instanceID(of: workstation.id) == "from-the-stream")
    }
}
