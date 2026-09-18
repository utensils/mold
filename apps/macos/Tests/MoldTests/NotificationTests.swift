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

    private func machine(_ name: String = "workstation") -> MoldHost {
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
        let workstation = machine()
        let backend = fake(for: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaults = scratchDefaults()
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        let center = FakeNotificationCenter()
        let notifications = MoldNotifications(
            landedPrints: landed, queue: QueueStore(hosts: hosts), hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(20), executablePath: insideBundle)
        await connect(workstation, hosts: hosts, backend: backend)

        for i in 0 ..< 4 { backend.emit(.gallery(.added(filename: "\(i).png", row: nil))) }

        await settle { !center.posted.isEmpty }
        #expect(center.posted.count == 1)
        #expect(center.posted.first?.title == "4 prints finished on workstation")
        #expect(notifications.enabled)
    }

    @Test func printsOnTwoMachinesAreTwoNotifications() async {
        let workstation = machine("workstation")
        let bender = machine("bender")
        let workstationBackend = fake(for: workstation)
        let benderBackend = fake(for: bender)
        let hosts = HostStore(hosts: [workstation, bender]) { host in
            host.id == workstation.id ? workstationBackend : benderBackend
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
        await connect(workstation, hosts: hosts, backend: workstationBackend)
        await connect(bender, hosts: hosts, backend: benderBackend)

        workstationBackend.emit(.gallery(.added(filename: "a.png", row: nil)))
        benderBackend.emit(.gallery(.added(filename: "b.png", row: nil)))

        await settle { center.posted.count == 2 }
        #expect(Set(center.posted.map(\.title)) == ["Finished on workstation", "Finished on bender"])
        #expect(notifications.enabled)
    }

    // MARK: - Failed / unretryable holds

    /// **Fails today**: `QueueStore.onOutcome` does not exist.
    @Test func aFailedChildNotifiesOnce() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry("job-1", state: "running", batchId: "batch-1", clientBatchId: "client-1"),
        ])
        backend.batchListings = [FakeFixtures.batchStatusListing([
            FakeFixtures.batchStatus(id: "batch-1", children: [FakeFixtures.batchChild("job-1", state: "running")]),
        ])]
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        await queue.refresh(on: workstation.id)

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
        await queue.refresh(on: workstation.id)
        // A second reconcile of the same settled state -- must not fire again.
        await queue.refresh(on: workstation.id)
        // Delivery waits for authorization to be ANSWERED, so this awaits
        // the chain rather than polling for its side effect.
        await notifications.deliveries?.value

        #expect(center.posted.count == 1)
        #expect(center.posted.first?.title == "Failed on workstation")
        #expect(center.posted.first?.body == "GPU crashed")
        #expect(notifications.enabled)
    }

    /// **Fails today**: `hydrate(on:)` snapshots `children[host]` BEFORE its
    /// first `await` and writes it back only at the end, and nothing stops
    /// two running at once -- `refresh(on:)` has several concurrent callers
    /// (the SSE coalescer, `QueuePane.load()`, `MachinesPane`'s `.task(id:)`
    /// and its Refresh). Both then compute `before` from the same snapshot,
    /// both see the same `held → failed` transition, and both post -- and a
    /// failure notification is deliberately never coalesced. Pressing ⌘R
    /// while a `job_state_committed` frame is in flight is enough.
    @Test func twoOverlappingHydrationsNotifyAboutOneFailureOnce() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        // The only route here that actually suspends, so two callers can
        // genuinely interleave rather than each running straight through.
        backend.batchStatusesYields = true
        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry("job-1", state: "running", batchId: "batch-1", clientBatchId: "client-1"),
        ])
        backend.batchListings = [FakeFixtures.batchStatusListing([
            FakeFixtures.batchStatus(id: "batch-1", children: [FakeFixtures.batchChild("job-1", state: "running")]),
        ])]
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        await queue.refresh(on: workstation.id)

        let defaults = scratchDefaults()
        let center = FakeNotificationCenter()
        let notifications = MoldNotifications(
            landedPrints: LandedPrints(hosts: hosts, defaults: defaults), queue: queue, hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(20), executablePath: insideBundle)

        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry("job-1", state: "failed", batchId: "batch-1", clientBatchId: "client-1"),
        ])
        await queue.poll(workstation.id)
        let failed = FakeFixtures.batchStatusListing([
            FakeFixtures.batchStatus(id: "batch-1", children: [
                FakeFixtures.batchChild("job-1", state: "failed", error: "GPU crashed"),
            ]),
        ])
        backend.batchListings = [failed, failed]

        async let first: Void = queue.hydrate(on: workstation.id)
        async let second: Void = queue.hydrate(on: workstation.id)
        _ = await (first, second)
        await notifications.deliveries?.value

        // Both callers still get a fresh read -- serialized, not coalesced.
        // Three in all: the one the setup's `refresh` made, plus these two.
        #expect(backend.callCount("batchStatuses") == 3)
        #expect(center.posted.count == 1)
        #expect(center.posted.first?.title == "Failed on workstation")
        #expect(notifications.enabled)
    }

    @Test func aRetryableHoldDoesNotNotify() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        backend.queueListing = FakeFixtures.queueListing(entries: [
            FakeFixtures.queueEntry("job-1", state: "running", batchId: "batch-1", clientBatchId: "client-1"),
        ])
        backend.batchListings = [FakeFixtures.batchStatusListing([
            FakeFixtures.batchStatus(id: "batch-1", children: [FakeFixtures.batchChild("job-1", state: "running")]),
        ])]
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let queue = QueueStore(hosts: hosts)
        await queue.refresh(on: workstation.id)

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
        await queue.refresh(on: workstation.id)

        #expect(center.posted.isEmpty)
        #expect(notifications.enabled)
    }

    // MARK: - The guard and the preference

    @Test func notificationsOffPostNothing() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaults = scratchDefaults()
        defaults.set(false, forKey: "notifyRenders")
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        let center = FakeNotificationCenter()
        let notifications = MoldNotifications(
            landedPrints: landed, queue: QueueStore(hosts: hosts), hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(20), executablePath: insideBundle)
        await connect(workstation, hosts: hosts, backend: backend)

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { landed.count > 0 }

        #expect(center.posted.isEmpty)
        #expect(notifications.enabled == false)
    }

    /// **Fails today**: nothing guards against a bare `swift run` binary.
    @Test func outsideABundleNothingIsPosted() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
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
        await connect(workstation, hosts: hosts, backend: backend)

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { landed.count > 0 }

        #expect(center.posted.isEmpty)
        #expect(center.authorizationRequests == 0)
        #expect(notifications.enabled)
    }

    // MARK: - Authorization

    /// **Fails today**: `requestAuthorizationIfNeeded()` is fire-and-forget
    /// and `add` runs in the same turn, so the very first notification of a
    /// session is handed to the centre while authorization is still
    /// `.notDetermined` -- and dropped. The person sees the permission alert
    /// and no notification, which reads as the toggle not working.
    ///
    /// Asking on FIRST NEED is right and stays; what changes is waiting for
    /// the ANSWER.
    @Test func theFirstNotificationWaitsForTheAnswerInsteadOfBeingDropped() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaults = scratchDefaults()
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        let center = FakeNotificationCenter()
        // The person has not answered the alert yet -- which is the whole
        // state this bug lives in.
        center.defersAuthorization = true
        let notifications = MoldNotifications(
            landedPrints: landed, queue: QueueStore(hosts: hosts), hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(20), executablePath: insideBundle)
        await connect(workstation, hosts: hosts, backend: backend)

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { center.authorizationRequests == 1 }
        #expect(center.posted.isEmpty)

        center.answerAuthorization(true)
        await notifications.deliveries?.value

        #expect(center.dropped == 0)
        #expect(center.posted.map(\.title) == ["Finished on workstation"])
    }

    /// One alert, however many notifications follow it.
    @Test func authorizationIsAskedForOnceAndTheRestJustArrive() async {
        let workstation = machine()
        let backend = fake(for: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaults = scratchDefaults()
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        let center = FakeNotificationCenter()
        let notifications = MoldNotifications(
            landedPrints: landed, queue: QueueStore(hosts: hosts), hosts: hosts,
            library: LibraryStore(hosts: hosts), center: center, defaults: defaults,
            coalesceDelay: .milliseconds(5), executablePath: insideBundle)
        await connect(workstation, hosts: hosts, backend: backend)

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { center.posted.count == 1 }
        backend.emit(.gallery(.added(filename: "b.png", row: nil)))
        await settle { center.posted.count == 2 }
        await notifications.deliveries?.value

        #expect(center.authorizationRequests == 1)
        #expect(center.dropped == 0)
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
        let workstation = machine()
        let backend = fake(for: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let defaults = scratchDefaults()
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        await connect(workstation, hosts: hosts, backend: backend)
        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { landed.count == 1 }

        defaults.set(false, forKey: "badgeLandedPrints")

        await settle { landed.count == 0 }
        #expect(landed.count == 0)
    }
}
