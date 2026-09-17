import Foundation
import MoldClient
import Testing

@testable import Mold

/// The Dock badge's source of truth. `LandedPrints` never sees the badge or
/// the AppKit call that draws it -- these pin what it counts and when it
/// forgets, the way `HostStoreLifecycleTests` pins `HostStore+Events`.
@MainActor
struct LandedPrintsTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// A machine that answers, and advertises `/api/events`.
    private func fake(for host: MoldHost) -> FakeBackend {
        let fake = FakeBackend(host: host)
        fake.serverStatus = FakeFixtures.serverStatus()
        fake.capabilityBlock = FakeFixtures.capabilities(events: true)
        fake.exportBlock = FakeFixtures.exportOptions()
        return fake
    }

    /// Opens this machine's live stream and waits for it, the same steps
    /// every test below needs before it can hand the fake a frame.
    private func connect(_ host: MoldHost, hosts: HostStore, backend: FakeBackend) async {
        await hosts.refresh(host)
        hosts.reconcileEventStreams()
        await settle { backend.callCount("events") == 1 }
    }

    /// A scratch `UserDefaults` suite, cleared before use -- so a preference
    /// one test writes never leaks into another's default of "on".
    private func scratchDefaults(_ name: String = #function) -> UserDefaults {
        let suite = "landed-prints-tests-\(name)-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    /// **Fails today**: there is no `LandedPrints` type.
    @Test func aPrintThatLandsWhileInactiveCountsOnce() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let landed = LandedPrints(hosts: hosts, defaults: scratchDefaults())
        landed.isActive = false
        await connect(plato, hosts: hosts, backend: backend)

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))

        await settle { landed.count == 1 }
        #expect(landed.count == 1)
    }

    @Test func theSameFilenameTwiceCountsOnce() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let landed = LandedPrints(hosts: hosts, defaults: scratchDefaults())
        landed.isActive = false
        await connect(plato, hosts: hosts, backend: backend)

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { landed.count == 1 }
        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { landed.recent.count == 1 }

        #expect(landed.count == 1)
    }

    /// Two DIFFERENT prints on two DIFFERENT machines are two arrivals --
    /// dedup is by filename, not by how many machines mentioned it.
    @Test func printsOnTwoMachinesBothCount() async {
        let plato = machine("plato")
        let bender = machine("bender")
        let platoBackend = fake(for: plato)
        let benderBackend = fake(for: bender)
        let hosts = HostStore(hosts: [plato, bender]) { host in
            host.id == plato.id ? platoBackend : benderBackend
        }
        let landed = LandedPrints(hosts: hosts, defaults: scratchDefaults())
        landed.isActive = false
        await connect(plato, hosts: hosts, backend: platoBackend)
        await connect(bender, hosts: hosts, backend: benderBackend)

        platoBackend.emit(.gallery(.added(filename: "a.png", row: nil)))
        benderBackend.emit(.gallery(.added(filename: "b.png", row: nil)))

        await settle { landed.count == 2 }
        #expect(landed.count == 2)
    }

    @Test func aPrintThatLandsWhileActiveDoesNotCount() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let landed = LandedPrints(hosts: hosts, defaults: scratchDefaults())
        landed.isActive = true
        await connect(plato, hosts: hosts, backend: backend)

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        // Waits out `settle`'s full budget -- there is no event to catch, so
        // this is what gives a WOULD-have-counted frame time to land before
        // the negative assertion below.
        await settle { landed.count > 0 }

        #expect(landed.count == 0)
    }

    @Test func activatingClears() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let landed = LandedPrints(hosts: hosts, defaults: scratchDefaults())
        landed.isActive = false
        await connect(plato, hosts: hosts, backend: backend)
        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { landed.count == 1 }

        landed.isActive = true

        #expect(landed.count == 0)
    }

    @Test func aDisabledBadgeCountsNothing() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let defaults = scratchDefaults()
        defaults.set(false, forKey: "badgeLandedPrints")
        let landed = LandedPrints(hosts: hosts, defaults: defaults)
        landed.isActive = false
        await connect(plato, hosts: hosts, backend: backend)

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { landed.count > 0 }

        #expect(landed.count == 0)
    }

    /// `job(.ended)`/`job(.stateCommitted)` say a job in THIS app's own queue
    /// moved -- not that anything landed. `gallery_added` is the only signal
    /// that means a print arrived (design M6 decision 21).
    @Test func thisAppsOwnQueueIsNotABadge() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let landed = LandedPrints(hosts: hosts, defaults: scratchDefaults())
        landed.isActive = false
        await connect(plato, hosts: hosts, backend: backend)

        backend.emit(.job(.ended(id: "job-1")))
        backend.emit(.job(.stateCommitted(id: "job-1")))
        await settle { landed.count > 0 }

        #expect(landed.count == 0)
    }

    // MARK: - The badge itself

    /// Zero is NO badge, not a badge reading "0".
    @Test func anEmptyCountPaintsNoBadgeAtAll() {
        #expect(DockBadge.label(for: 0) == nil)
        #expect(DockBadge.label(for: 1) == "1")
        #expect(DockBadge.label(for: 12) == "12")
    }

    /// **Fails today**: the badge is an `.onChange(of:)` on `RootView`
    /// (`MoldApp.swift:88-90`), so it tracks the count only while that view
    /// is mounted -- and this app does not terminate when its last window
    /// closes. `LandedPrints` goes on counting either way, which is what
    /// makes the two disagree.
    @Test func theBadgeFollowsTheCountWithNoViewInvolved() async {
        let plato = machine()
        let backend = fake(for: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let landed = LandedPrints(hosts: hosts, defaults: scratchDefaults())
        landed.isActive = false
        // A sentinel for "no badge", so `.last` is a plain `String?` rather
        // than a `String??` where `nil` would also mean "never painted".
        var painted: [String] = []
        let badge = DockBadge { painted.append($0 ?? "none") }
        badge.follow(landed)
        await connect(plato, hosts: hosts, backend: backend)

        // Painted once on adoption: whatever is true now, not only what
        // changes later.
        #expect(painted == ["none"])

        backend.emit(.gallery(.added(filename: "a.png", row: nil)))
        await settle { painted.last == "1" }
        #expect(painted.last == "1")

        backend.emit(.gallery(.added(filename: "b.png", row: nil)))
        await settle { painted.last == "2" }
        #expect(painted.last == "2")

        // Coming back to the app clears what accumulated while away.
        landed.isActive = true
        await settle { painted.last == "none" }
        #expect(painted.last == "none")
    }
}
