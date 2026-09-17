import Foundation
import MoldClient
import Testing

@testable import Mold

/// Asking the machines what they are doing, on a timer.
///
/// **Fails today**: nothing in this app reads `/api/activity`, so the Queue
/// pane shows an idle list while a machine is preparing, rewriting a prompt
/// or running a durable sequence.
@MainActor
struct ActivityStoreTests {

    private let becameActive = Notification.Name("f3.active")
    private let resignedActive = Notification.Name("f3.inactive")

    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func fake(for host: MoldHost) -> FakeBackend {
        let fake = FakeBackend(host: host)
        fake.serverStatus = FakeFixtures.serverStatus()
        fake.capabilityBlock = FakeFixtures.capabilities(events: true)
        fake.exportBlock = FakeFixtures.exportOptions()
        fake.extras.activitySnapshot = snapshot([row("w-1", kind: "prompt_expansion")])
        return fake
    }

    private func row(_ id: String, kind: String = "generation", phase: String = "running",
                     created: Int = 1_000) -> String {
        #"""
        {"id": "\#(id)", "kind": "\#(kind)", "phase": "\#(phase)",
         "created_at_unix_ms": \#(created), "updated_at_unix_ms": \#(created),
         "can_cancel": false}
        """#
    }

    private func snapshot(_ rows: [String], instance: String = "inst-1") -> ActiveWorkSnapshot {
        let json = #"""
        {"instance_id": "\#(instance)", "observed_at_unix_ms": 9,
         "items": [\#(rows.joined(separator: ","))], "unavailable_kinds": []}
        """#
        return try! MoldJSON.decoder.decode(ActiveWorkSnapshot.self, from: Data(json.utf8))
    }

    private func bench(_ backend: FakeBackend, host: MoldHost,
                       interval: Duration = .milliseconds(2))
        async -> (ActivityStore, HostStore, NotificationCenter) {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        await hosts.refresh(host)
        let centre = NotificationCenter()
        let store = ActivityStore(hosts: hosts, interval: interval, activeCenter: centre,
                                  activeName: becameActive, inactiveName: resignedActive)
        return (store, hosts, centre)
    }

    @Test func oneReadFillsTheFleetsRows() async {
        let plato = machine()
        let backend = fake(for: plato)
        let (store, _, _) = await bench(backend, host: plato)

        await store.refresh()

        #expect(store.rows.map(\.item.id) == ["w-1"])
        #expect(store.rows.first?.host == plato.id)
        #expect(backend.callCount("activity") == 1)
    }

    /// The tick stops while somebody is in another app -- nobody is reading a
    /// pane, and the first tick on becoming active catches up.
    @Test func itTicksOnlyWhileTheAppIsFrontmost() async {
        let plato = machine()
        let backend = fake(for: plato)
        let (store, _, centre) = await bench(backend, host: plato)

        centre.post(name: becameActive, object: nil)
        await settle { store.isTicking }
        await settle { backend.callCount("activity") >= 2 }

        centre.post(name: resignedActive, object: nil)
        await settle { !store.isTicking }
        let asks = backend.callCount("activity")
        // Twenty-five chances to ask again at a 2 ms interval.
        try? await Task.sleep(for: .milliseconds(50))
        #expect(backend.callCount("activity") == asks)

        centre.post(name: becameActive, object: nil)
        await settle { backend.callCount("activity") > asks }
    }

    /// Becoming active twice is one loop, not two.
    @Test func becomingActiveTwiceDoesNotDoubleTheLoop() async {
        let plato = machine()
        let backend = fake(for: plato)
        let (store, _, centre) = await bench(backend, host: plato, interval: .milliseconds(30))

        centre.post(name: becameActive, object: nil)
        await settle { store.isTicking }
        centre.post(name: becameActive, object: nil)
        await settle { backend.callCount("activity") >= 1 }

        try? await Task.sleep(for: .milliseconds(45))
        #expect(backend.callCount("activity") <= 2, "one loop at 30 ms, not two")
        store.stop()
    }

    /// A machine that is down is not asked at all, and its rows are kept and
    /// marked stale rather than dropped -- being asleep is not evidence that
    /// its work has gone.
    @Test func aMachineThatIsDownKeepsItsLastRowsAndIsNotAsked() async {
        let plato = machine()
        let backend = fake(for: plato)
        let (store, hosts, _) = await bench(backend, host: plato)

        await store.refresh()
        #expect(store.rows.map(\.item.id) == ["w-1"])

        backend.plantedErrors["status"] = MoldClientError.unreachable("it is asleep")
        await hosts.refresh(plato)
        let asks = backend.callCount("activity")
        await store.refresh()

        #expect(backend.callCount("activity") == asks, "a machine known to be down is not asked")
        #expect(store.rows.map(\.item.id) == ["w-1"])
        #expect(store.rows.first?.stale == true)
    }

    /// A machine that has been forgotten stops contributing rows nobody can
    /// attribute to a machine.
    @Test func aForgottenMachineStopsContributingRows() async {
        let plato = machine()
        let backend = fake(for: plato)
        let (store, hosts, _) = await bench(backend, host: plato)

        await store.refresh()
        #expect(!store.rows.isEmpty)

        hosts.hosts = []
        await store.refresh()
        #expect(store.rows.isEmpty)
    }

    /// Two ticks landing at once are one read, and the second joins the
    /// first's promise -- `SingleFlight`'s contract, not a second mechanism.
    @Test func concurrentTicksAreOneReadPlusAtMostOneBehindIt() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.delays["activity"] = .milliseconds(20)
        let (store, _, _) = await bench(backend, host: plato)

        async let first: Void = store.refresh()
        async let second: Void = store.refresh()
        async let third: Void = store.refresh()
        _ = await (first, second, third)

        #expect(backend.callCount("activity") <= 2)
    }
}
