import Foundation
import MoldClient
import Testing

@testable import Mold

/// `ConfigStore` -- one `GET /api/config` listing per machine, per-key
/// writes that re-read the whole thing (a single `expand.*` PUT rewrites
/// all eight, `config_sync.rs:674-688`), and what each machine refused,
/// kept per key rather than folded into the fleet-wide failure banner.
///
/// Refusal-specific cases live in `ConfigStoreTests+Refusals.swift`, which
/// shares `machine(_:)` below -- not `private`, so an extension in another
/// file can still call it.
@MainActor
struct ConfigStoreTests {
    func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    // MARK: - A write re-reads the whole listing

    /// **Fails today** -- there is no `set` at all. One `expand.*` PUT
    /// rewrites all eight rows (`config_sync.rs:674-688`), so applying only
    /// the answered row would leave seven fields showing stale numbers;
    /// the store re-reads instead of trusting the one row it changed.
    @Test func writingOneExpansionKeyRereadsTheWholeListing() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = FakeFixtures.configListing()
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)
        await store.refresh(on: workstation.id)
        #expect(backend.callCount("config") == 1)

        let ok = await store.set("expand.max_tokens", to: .number(500), on: workstation.id)

        #expect(ok == true)
        #expect(backend.callCount("config") == 2)
        #expect(backend.configWrites.contains { $0.0 == "expand.max_tokens" })
    }

    /// What lands after a successful write is what the fake's `config()`
    /// answers on the re-read -- the fake mutates its own listing on a
    /// write, the way a live server's next `GET` would.
    @Test func aSuccessfulWriteShowsWhatTheMachineAnsweredWith() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = ConfigListing(entries: [
            FakeFixtures.configEntry("expand.max_tokens", value: .number(300), source: "db"),
        ])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)
        await store.refresh(on: workstation.id)

        let ok = await store.set("expand.max_tokens", to: .number(500), on: workstation.id)

        #expect(ok == true)
        #expect(store.entry("expand.max_tokens", on: workstation.id)?.value == .number(500))
    }

    /// DELETE, then a re-read of the fallback the machine reports.
    @Test func resettingAKeyShowsTheFallbackTheMachineReturned() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = ConfigListing(entries: [
            FakeFixtures.configEntry("expand.max_tokens", value: .number(500), source: "db"),
        ])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)
        await store.refresh(on: workstation.id)

        let ok = await store.reset("expand.max_tokens", on: workstation.id)

        #expect(ok == true)
        #expect(backend.configResets == ["expand.max_tokens"])
        #expect(backend.callCount("config") == 2)
        #expect(store.entry("expand.max_tokens", on: workstation.id)?.value == .null)
    }

    // MARK: - A DB-off machine

    /// A 503 `CONFIG_UNAVAILABLE` marks the machine `unavailable`, exactly
    /// as `ModelDefaultsStore.refresh` already handles it -- never a
    /// per-key refusal, since no key was even considered.
    @Test func aMachineWithItsDatabaseOffSaysSoOnceAndAsksNothingElse() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.plantedErrors["setConfig"] =
            MoldClientError.http(status: 503, code: "CONFIG_UNAVAILABLE", message: "no metadata db")
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)

        let ok = await store.set("expand.max_tokens", to: .number(500), on: workstation.id)

        #expect(ok == false)
        #expect(store.unavailable.contains(workstation.id))
        #expect(store.refusal(for: "expand.max_tokens", on: workstation.id) == nil)
        #expect(hosts.failures.isEmpty)
    }

    // MARK: - Profiles

    @Test func profilesDecodeFromTheWire() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.profilesAnswer = ConfigProfiles(active: "default", profiles: ["default", "studio"])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)

        await store.refreshProfiles(on: workstation.id)

        #expect(store.profiles[workstation.id]?.active == "default")
        #expect(store.profiles[workstation.id]?.profiles == ["default", "studio"])
    }

    /// A host that predates `GET /api/config/profiles` answers 404 -- that
    /// is "never had the route", not a failure to report.
    @Test func anOlderHostWithNoProfilesRouteAnswersNil() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.plantedErrors["configProfiles"] =
            MoldClientError.http(status: 404, code: nil, message: "no such route")
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)

        await store.refreshProfiles(on: workstation.id)

        #expect(store.profiles[workstation.id] == nil)
        #expect(hosts.failures.isEmpty)
    }

    // MARK: - M3 regression guard

    /// The rename must not touch M3's own behaviour: per-model defaults
    /// still read from the same listing this store now serves for
    /// Advanced too.
    @Test func thePerModelDefaultsStillReadFromTheSameListing() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = ConfigListing(entries: [
            FakeFixtures.configEntry("models.flux-dev:q8.default_steps", value: .number(12), source: "db"),
        ])
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)
        await store.refresh(on: workstation.id)

        let defaults = store.defaults(for: "flux-dev:q8", on: workstation.id)

        #expect(defaults.steps == 12)
    }
}
