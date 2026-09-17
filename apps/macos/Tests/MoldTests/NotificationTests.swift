import Foundation
import MoldClient
import Testing

@testable import Mold

/// `MoldNotifications`, its coalescing, its bundle guard, and the pure half
/// of a click's routing -- the way `LandedPrintsTests` pins the Dock badge
/// these share a source with (design M6 S5, decision 22).
@MainActor
struct NotificationTests {
    /// A real `.app` executable, and a bare one -- the shape the bundle
    /// guard actually reads (`MoldNotifications.isInsideBundle`).
    private let insideBundle = "/Applications/Mold.app/Contents/MacOS/Mold"
    private let outsideBundle = "/Users/dev/.build/debug/mold"

    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func fake(for host: MoldHost) -> FakeBackend {
        let fake = FakeBackend(host: host)
        fake.serverStatus = FakeFixtures.serverStatus()
        fake.capabilityBlock = FakeFixtures.capabilities(events: true)
        fake.exportBlock = FakeFixtures.exportOptions()
        return fake
    }

    private func connect(_ host: MoldHost, hosts: HostStore, backend: FakeBackend) async {
        await hosts.refresh(host)
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }
    }

    private func scratchDefaults(_ name: String = #function) -> UserDefaults {
        let suite = "notification-tests-\(name)-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    // MARK: - Finished, coalesced

    /// **Fails today**: there is no `MoldNotifications` type.
    @Test func fourPrintsOnOneMachineWithinTwoSecondsAreOneNotification() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaults = scratchDefaults()
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        let center = FakeNotificationCenter()
        let notifications = MoldNotifications(
            landedPrints: landed, queue: QueueStore(hosts: hosts), hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(20), executablePath: insideBundle)
        await connect(plato, hosts: hosts, backend: backend)

        for i in 0 ..< 4 { backend.emit(.gallery(.added(filename: "\(i).png", row: nil))) }

        await settle { !center.posted.isEmpty }
        #expect(center.posted.count == 1)
        #expect(center.posted.first?.title == "4 prints finished on plato")
        #expect(notifications.enabled)
    }

    @Test func printsOnTwoMachinesAreTwoNotifications() async {
        let plato = machine("plato")
        let bender = machine("bender")
        let platoBackend = fake(for: plato)
        let benderBackend = fake(for: bender)
        let hosts = HostStore(hosts: [plato, bender]) { host in
            host.id == plato.id ? platoBackend : benderBackend
        }
        let defaults = scratchDefaults()
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        let center = FakeNotificationCenter()
        // Bound, not `_ = `: the coalescing `Task`s only reach this object
        // through a WEAK `self` (both `landedPrints.onLanding` and, in other
        // tests, `queue.onOutcome`), so a discarded result is free to be
        // deallocated the moment `init` returns -- the closures then fire
        // against nothing and nothing is ever posted.
        let notifications = MoldNotifications(
            landedPrints: landed, queue: QueueStore(hosts: hosts), hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(20), executablePath: insideBundle)
        await connect(plato, hosts: hosts, backend: platoBackend)
        await connect(bender, hosts: hosts, backend: benderBackend)

        platoBackend.emit(.gallery(.added(filename: "a.png", row: nil)))
        benderBackend.emit(.gallery(.added(filename: "b.png", row: nil)))

        await settle { center.posted.count == 2 }
        #expect(Set(center.posted.map(\.title)) == ["Finished on plato", "Finished on bender"])
        #expect(notifications.enabled)
    }

    // MARK: - Failed / unretryable holds

    /// **Fails today**: `QueueStore.onOutcome` does not exist.
    @Test func aFailedChildNotifiesOnce() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry("job-1", state: "running", batchId: "batch-1", clientBatchId: "client-1"),
        ])
        backend.batchListings = [FakeFixtures.batchStatusListing([
            FakeFixtures.batchStatus(id: "batch-1", children: [FakeFixtures.batchChild("job-1", state: "running")]),
        ])]
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        await queue.refresh(on: plato.id)

        let defaults = scratchDefaults()
        let center = FakeNotificationCenter()
        let notifications = MoldNotifications(
            landedPrints: LandedPrints(hosts: hosts, defaults: defaults), queue: queue, hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(20), executablePath: insideBundle)

        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry("job-1", state: "failed", batchId: "batch-1", clientBatchId: "client-1"),
        ])
        let failed = FakeFixtures.batchStatusListing([
            FakeFixtures.batchStatus(id: "batch-1", children: [
                FakeFixtures.batchChild("job-1", state: "failed", error: "GPU crashed"),
            ]),
        ])
        backend.batchListings = [failed, failed]
        await queue.refresh(on: plato.id)
        // A second reconcile of the same settled state -- must not fire again.
        await queue.refresh(on: plato.id)

        #expect(center.posted.count == 1)
        #expect(center.posted.first?.title == "Failed on plato")
        #expect(center.posted.first?.body == "GPU crashed")
        #expect(notifications.enabled)
    }

    @Test func aRetryableHoldDoesNotNotify() async {
        let plato = machine()
        let backend = fake(for: plato)
        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry("job-1", state: "running", batchId: "batch-1", clientBatchId: "client-1"),
        ])
        backend.batchListings = [FakeFixtures.batchStatusListing([
            FakeFixtures.batchStatus(id: "batch-1", children: [FakeFixtures.batchChild("job-1", state: "running")]),
        ])]
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        await queue.refresh(on: plato.id)

        let defaults = scratchDefaults()
        let center = FakeNotificationCenter()
        let notifications = MoldNotifications(
            landedPrints: LandedPrints(hosts: hosts, defaults: defaults), queue: queue, hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(20), executablePath: insideBundle)

        // A transient hold the host says WILL resolve itself -- notifying
        // here is the noise a 429 hold produces on every retry loop.
        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry(
                "job-1", state: "held", batchId: "batch-1", clientBatchId: "client-1",
                heldReason: "GPU ran out of memory.", retryable: true),
        ])
        backend.batchListings = [FakeFixtures.batchStatusListing([
            FakeFixtures.batchStatus(id: "batch-1", children: [
                FakeFixtures.batchChild("job-1", state: "held", retryable: true),
            ]),
        ])]
        await queue.refresh(on: plato.id)

        #expect(center.posted.isEmpty)
        #expect(notifications.enabled)
    }

    // MARK: - The guard and the preference

    @Test func notificationsOffPostNothing() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaults = scratchDefaults()
        defaults.set(false, forKey: "notifyRenders")
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        let center = FakeNotificationCenter()
        let notifications = MoldNotifications(
            landedPrints: landed, queue: QueueStore(hosts: hosts), hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(20), executablePath: insideBundle)
        await connect(plato, hosts: hosts, backend: backend)

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { landed.count > 0 }

        #expect(center.posted.isEmpty)
        #expect(notifications.enabled == false)
    }

    /// **Fails today**: nothing guards against a bare `swift run` binary.
    @Test func outsideABundleNothingIsPosted() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaults = scratchDefaults()
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        let center = FakeNotificationCenter()
        // Bound for the same reason as `printsOnTwoMachinesAreTwoNotifications`
        // -- a discarded result could pass this test for the wrong reason
        // (deallocated before it could post) rather than for the guard's.
        let notifications = MoldNotifications(
            landedPrints: landed, queue: QueueStore(hosts: hosts), hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(20), executablePath: outsideBundle)
        await connect(plato, hosts: hosts, backend: backend)

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { landed.count > 0 }

        #expect(center.posted.isEmpty)
        #expect(center.authorizationRequests == 0)
        #expect(notifications.enabled)
    }

    // MARK: - Routing, pure over `userInfo`

    @Test func aFinishedClickOpensTheLibraryOnThatPrint() {
        let host = UUID()
        let userInfo = ["kind": "print", "host": host.uuidString, "filename": "a.png"]
        #expect(NotificationRoute.route(userInfo: userInfo) == .openPrint(PrintID(host: host, filename: "a.png")))
    }

    @Test func aFailedClickOpensTheQueue() {
        #expect(NotificationRoute.route(userInfo: ["kind": "failure"]) == .openQueue)
    }

    @Test func anUnrecognisedPayloadRoutesNowhere() {
        #expect(NotificationRoute.route(userInfo: [:]) == nil)
        #expect(NotificationRoute.route(userInfo: ["kind": "print"]) == nil)
    }

    // MARK: - The badge, cleared from under itself

    /// **Fails today**: turning the preference off does not touch `recent`.
    @Test func turningTheBadgeOffClearsIt() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaults = scratchDefaults()
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        await connect(plato, hosts: hosts, backend: backend)
        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { landed.count == 1 }

        defaults.set(false, forKey: "badgeLandedPrints")

        await settle { landed.count == 0 }
        #expect(landed.count == 0)
    }
}
