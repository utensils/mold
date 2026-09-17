import Foundation
import MoldClient
import Testing

@testable import Mold

/// What keeps the fleet honest when nobody is pressing ⌘R.
///
/// Three machines were unreachable by design before this: one that was off at
/// launch, one too old to advertise `/api/events`, and one on the other side
/// of a sleep. `HostStore+Events`' reconnect rule and `HostHeartbeat` are the
/// two halves; these pin both without a rendered view, a real `NSWorkspace`
/// or a real clock -- the tick interval and the wake centre are constructor
/// parameters exactly so.
@MainActor
struct HeartbeatTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// A machine that answers, and advertises `/api/events` unless told
    /// otherwise.
    private func fake(for host: MoldHost, events: Bool = true) -> FakeBackend {
        let fake = FakeBackend(host: host)
        fake.serverStatus = FakeFixtures.serverStatus()
        fake.capabilityBlock = FakeFixtures.capabilities(events: events)
        fake.exportBlock = FakeFixtures.exportOptions()
        return fake
    }

    private let wakeName = Notification.Name("HeartbeatTests.didWake")

    private func heartbeat(
        _ hosts: HostStore, _ queue: QueueStore, every interval: Duration = .milliseconds(5),
        wakeCenter: NotificationCenter = NotificationCenter()
    ) -> HostHeartbeat {
        HostHeartbeat(hosts: hosts, queue: queue, interval: interval,
                      wakeCenter: wakeCenter, wakeName: wakeName)
    }

    // MARK: - The reconnect rule

    /// **Fails today**: `deliver` emits `.resyncRequired` only when the
    /// instance id CHANGES, and `instance_id` is persisted per
    /// data-dir-and-port (`instance.rs:20-28`), so it survives a restart --
    /// the Mac sleeping through three renders and `mold serve` restarting are
    /// both reconnects at the SAME identity, and neither reconciled anything.
    @Test func aReconnectAsksEveryListenerToResyncEvenOnTheSameIdentity() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        var resyncs = 0
        hosts.onEvent { _, event in if case .resyncRequired = event { resyncs += 1 } }

        await hosts.refresh(plato)
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }
        backend.emit(.authority(instanceID: "run-1"))
        await settle { hosts.instanceIDs[plato.id] != nil }
        // The FIRST connection's opening frame is not news: there is no gap
        // behind it.
        #expect(resyncs == 0)

        hosts.reconnectEventStreams()
        await settle { backend.callCount("events") == 2 }
        backend.emit(.authority(instanceID: "run-1"))

        await settle { resyncs == 1 }
        #expect(resyncs == 1)
        #expect(hosts.instanceIDs[plato.id] == "run-1")
    }

    /// The gap is repaired the way the server prescribes -- a fresh
    /// `GET /api/queue` (`routes.rs:11751-11755`) -- not merely announced.
    @Test func aReconnectMakesTheQueueReadTheMachineAgain() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts, coalesceDelay: .seconds(60))

        await hosts.refresh(plato)
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }
        backend.emit(.authority(instanceID: "run-1"))
        await settle { hosts.instanceIDs[plato.id] != nil }

        hosts.reconnectEventStreams()
        await settle { backend.callCount("events") == 2 }
        backend.emit(.authority(instanceID: "run-1"))

        // `.resyncRequired` skips the coalescing delay by design, so this
        // awaits the exact task the frame started rather than a budget.
        await settle { queue.coalescers[plato.id] != nil }
        await queue.coalescers[plato.id]?.value
        #expect(backend.callCount("queue") == 1)
        #expect(queue.entries(on: plato.id).map(\.id) == ["job-1"])
    }

    // MARK: - The tick

    /// **Fails today**: `QueueStore.wantsPoll` documents this fallback and has
    /// no production caller, so a machine with no event route is read once and
    /// never again.
    @Test func aMachineWithNoEventRouteIsPolledOnItsOwn() async {
        let plato = machine()
        let backend = fake(for: plato, events: false)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        await hosts.refresh(plato)
        let beat = heartbeat(hosts, queue)

        beat.start()
        await settle { backend.callCount("queue") >= 2 }
        beat.stop()

        #expect(backend.callCount("queue") >= 2)
        #expect(queue.entries(on: plato.id).map(\.id) == ["job-1"])
    }

    /// A machine whose stream is live already hears everything. Polling it
    /// too would be a second authority arguing with the first.
    @Test func aMachineWithALiveStreamIsNeverPolled() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        await hosts.refresh(plato)
        let beat = heartbeat(hosts, queue)

        for _ in 0 ..< 3 { await beat.tick() }

        #expect(backend.callCount("queue") == 0)
        #expect(backend.callCount("batchStatuses") == 0)
    }

    /// **Fails today**: nothing probes a machine twice, so one that was off
    /// when Mold launched stays dead for the whole session -- `wantsEvents`
    /// never returns true for it and no pane ever asks again.
    @Test func aMachineThatWasDownAtLaunchJoinsWithoutAnybodyPressingRefresh() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)

        // Nothing planted: `status()` throws, which is a machine that is off.
        await hosts.refresh(plato)
        #expect(!hosts.isUp(plato))

        // It comes back up while Mold is running.
        backend.serverStatus = FakeFixtures.serverStatus()
        backend.capabilityBlock = FakeFixtures.capabilities(events: true)
        backend.exportBlock = FakeFixtures.exportOptions()
        await heartbeat(hosts, queue).tick()

        #expect(hosts.isUp(plato))
        await settle { backend.callCount("events") == 1 }
        #expect(hosts.watchers[plato.id] != nil)
    }

    /// **Fails today**: `tick()` asks for capabilities only through
    /// `hosts.refresh(host)`, and calls it only for a host that is NOT up.
    /// `/api/capabilities` is fetched in exactly one place and with `try?`
    /// (`HostStore+Reachability.swift:46-50`), so one transient failure at
    /// launch leaves `capabilities[id]` nil forever: `wantsEvents` is false,
    /// so the machine gets no event stream ever, and `wantsPoll` is true, so
    /// it is polled at full rate instead. The one state where ⌘R is still
    /// the only way out -- hidden behind a poll that looks like it works.
    @Test func anUpMachineThatNeverSaidWhatItCanDoIsAskedAgain() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.serverStatus = FakeFixtures.serverStatus()
        backend.exportBlock = FakeFixtures.exportOptions()
        // Answers `/api/status` and refuses `/api/capabilities`.
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        await hosts.refresh(plato)
        #expect(hosts.isUp(plato))
        #expect(hosts.capabilities[plato.id] == nil)
        let beat = heartbeat(hosts, queue)

        backend.capabilityBlock = FakeFixtures.capabilities(events: true)
        await beat.tick()

        #expect(hosts.capabilities[plato.id] != nil)
        await settle { backend.callCount("events") == 1 }
        #expect(hosts.watchers[plato.id] != nil)
    }

    /// And it is not asked on every tick forever. A machine that answers
    /// `/api/status` and keeps failing `/api/capabilities` backs off, rather
    /// than costing a request every ten seconds for the life of the app.
    @Test func aMachineThatKeepsRefusingItsCapabilitiesIsAskedLessOften() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.serverStatus = FakeFixtures.serverStatus()
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        await hosts.refresh(plato)
        let asked = backend.callCount("capabilities")
        let beat = heartbeat(hosts, queue)

        await beat.tick()
        #expect(backend.callCount("capabilities") == asked + 1)

        // The next tick is inside the backoff window.
        await beat.tick()
        #expect(backend.callCount("capabilities") == asked + 1)

        // And it does come back round, rather than giving up.
        for _ in 0 ..< 4 { await beat.tick() }
        #expect(backend.callCount("capabilities") > asked + 1)
    }

    /// **Fails today**: `tick()` is a plain `for` with an `await` inside, so
    /// a machine that is off holds the loop for its whole connect timeout and
    /// every machine behind it waits. `HostStore.refreshAll` already uses a
    /// task group for exactly this reason.
    @Test func oneSlowMachineDoesNotHoldUpTheRest() async {
        let slow = machine("plato"), quick = machine("hal9000")
        let slowBackend = FakeBackend(host: slow)
        slowBackend.statusHeldOpen = true
        let quickBackend = fake(for: quick, events: false)
        quickBackend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [slow, quick]) { host in
            host.id == slow.id ? slowBackend : quickBackend
        }
        let queue = QueueStore(hosts: hosts)
        // `quick` is up and cannot stream, so a tick owes it a queue read.
        await hosts.refresh(quick)
        let beat = heartbeat(hosts, queue)

        // `slow` is first in the list and its status never answers.
        let tick = Task { await beat.tick() }
        await settle { quickBackend.callCount("queue") == 1 }

        #expect(quickBackend.callCount("queue") == 1, "asked while the first machine was still answering")
        slowBackend.releaseStatus()
        await tick.value
    }

    /// A tick that was cancelled -- `stop()` while one is in flight -- asks
    /// nothing.
    @Test func aCancelledTickAsksNothing() async {
        let plato = machine()
        let backend = fake(for: plato, events: false)
        backend.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        await hosts.refresh(plato)
        let before = backend.calls.count
        let beat = heartbeat(hosts, queue)

        let tick = Task { await beat.tick() }
        tick.cancel()
        await tick.value

        #expect(backend.calls.count == before)
    }

    // MARK: - Waking up

    /// **Fails today**: nothing observes `NSWorkspace.didWakeNotification`, so
    /// after a sleep the watcher is holding a dead socket or waiting out a
    /// backoff of up to 32 s and the pane shows what was true yesterday.
    @Test func wakingTheMacReconnectsEveryStreamAndAsksForAResync() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        var resyncs = 0
        hosts.onEvent { _, event in if case .resyncRequired = event { resyncs += 1 } }
        let center = NotificationCenter()
        // Held, or the observer it registered goes away with it.
        let beat = heartbeat(hosts, queue, wakeCenter: center)

        await hosts.refresh(plato)
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }
        backend.emit(.authority(instanceID: "run-1"))
        await settle { hosts.instanceIDs[plato.id] != nil }

        center.post(name: wakeName, object: nil)

        await settle { backend.callCount("events") == 2 }
        #expect(backend.callCount("events") == 2)
        backend.emit(.authority(instanceID: "run-1"))
        await settle { resyncs == 1 }
        #expect(resyncs == 1)
        beat.stop()
    }
}
