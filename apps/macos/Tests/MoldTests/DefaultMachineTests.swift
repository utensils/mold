import Foundation
import MoldClient
import Testing

@testable import Mold

/// `HostStore.preferredHost` gains an explicit choice
/// (`HostStore+Default.swift`, design M7 S6): a machine set as default
/// outranks the "first one up" heuristic even when it is down, and a
/// REMOVED default falls back to that heuristic rather than to nothing.
///
/// `defaultMachine` reads and writes `AppStorageSuite.defaults` directly --
/// the same suite `Destination.launch` and `HostPersistence` already use, not
/// an injectable one of its own (design S6 declares it a plain property).
/// Under `xcodebuild test` that resolves to the named scratch suite the test
/// scheme's `MOLD_NATIVE_FRESH` selects (`Mold.xcscheme`'s `TestAction`), but
/// that suite is SHARED across every test in the run, unlike
/// `PendingBatchTests`'s own per-test scratch -- so every test here clears
/// the key first, and no earlier test's choice survives into this one.
@MainActor
struct DefaultMachineTests {
    private func machine(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func reset() {
        AppStorageSuite.defaults.removeObject(forKey: "defaultMachine")
    }

    /// `serverStatus` left unplanted is what `status()` throws off
    /// (`FakeBackend.swift:163-167`) -- a machine that never answers is a
    /// down one, with no separate flag to set.
    private func fake(_ host: MoldHost, up: Bool) -> FakeBackend {
        let fake = FakeBackend(host: host)
        if up { fake.serverStatus = FakeFixtures.serverStatus() }
        return fake
    }

    /// **Fails today**: `preferredHost` prefers any up machine over an
    /// explicit choice, so it lands on `hal9000` instead.
    @Test func aChosenDefaultIsWhereWorkGoesEvenWhenItIsDown() async {
        reset()
        let workstation = machine("workstation")
        let hal9000 = machine("hal9000")
        let hosts = HostStore(hosts: [workstation, hal9000]) { host in
            host.id == workstation.id ? self.fake(workstation, up: false) : self.fake(hal9000, up: true)
        }
        await hosts.refreshAll()

        hosts.setDefault(workstation)

        #expect(hosts.preferredHost?.id == workstation.id)
    }

    /// **Fails today**: nothing is stored to fall back from, so this only
    /// happens to pass by way of the fallback that already existed --
    /// pinned here so it keeps passing once a real default exists to lose.
    @Test func aDefaultThatHasBeenRemovedFallsBackToTheFirstOneUp() async {
        reset()
        let hal9000 = machine("hal9000")
        let hosts = HostStore(hosts: [hal9000]) { _ in self.fake(hal9000, up: true) }
        await hosts.refreshAll()

        // Named, but never added -- as good as a default whose machine was
        // since removed.
        hosts.setDefault(machine("workstation"))

        #expect(hosts.preferredHost?.id == hal9000.id)
    }

    /// The decision-10 bug: `selectedMachine` (`Sidebar.swift`) is a
    /// different key, and `machine(selected:)` -- the resolver both the
    /// sidebar and `MachinesPane` read it through -- must never write
    /// `defaultMachine` merely because it was asked to resolve one.
    @Test func lookingAtAMachineDoesNotMakeItTheDefault() {
        reset()
        let workstation = machine("workstation")
        let hosts = HostStore(hosts: [workstation]) { self.fake($0, up: true) }

        _ = hosts.machine(selected: workstation.id.uuidString)

        #expect(hosts.defaultMachine == nil)
    }

    /// UAT 2026-09-17 #4: Set as Default wrote the key, and nothing redrew --
    /// no Default badge on the card, the item still offered -- until the pane
    /// was left and re-entered, because a computed accessor over the suite
    /// is not observable state.
    ///
    /// **Fails today**: no observation is registered for a computed property,
    /// so `onChange` never fires.
    @Test func settingTheDefaultIsSomethingAViewCanWatch() {
        reset()
        let workstation = machine("workstation")
        let hosts = HostStore(hosts: [workstation]) { self.fake($0, up: true) }
        let fired = Flag()
        withObservationTracking { _ = hosts.defaultMachine } onChange: { fired.raise() }

        hosts.setDefault(workstation)

        #expect(fired.isRaised)
        #expect(AppStorageSuite.defaults.string(forKey: "defaultMachine") == workstation.id.uuidString)
    }

    /// A preferences reset clears the key straight from the suite, the way
    /// every `@AppStorage` view expects -- the store must follow, not keep a
    /// default the person just reset.
    ///
    /// **Fails today**: the observed copy is never told.
    @Test func aPreferencesResetClearsTheDefaultInTheStoreToo() {
        reset()
        let workstation = machine("workstation")
        let hosts = HostStore(hosts: [workstation]) { self.fake($0, up: true) }
        hosts.setDefault(workstation)

        PreferencesReset.reset(in: AppStorageSuite.defaults)

        #expect(hosts.defaultMachine == nil)
    }

    @Test func forgettingAMachineForgetsThatItWasTheDefault() {
        reset()
        let workstation = machine("workstation")
        let hosts = HostStore(hosts: [workstation]) { self.fake($0, up: true) }
        hosts.setDefault(workstation)

        hosts.remove(workstation)

        #expect(hosts.defaultMachine == nil)
    }
}
