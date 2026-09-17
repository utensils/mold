import Foundation
import MoldClient
import Testing

@testable import Mold

/// The Advanced table (S3): a pure row builder over a listing, a pure
/// per-entry field resolver (the same idiom `DiscoverRow.resolve` uses), and
/// the pane's own whole-pane state -- all askable with no view, the same way
/// `ConfigStoreTests` asks the store with no view.
@MainActor
struct AdvancedTableTests {
    func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    // MARK: - Rows

    @Test func aSearchNarrowsByKeyAndSaysHowManyOfHowMany() {
        let listing = ConfigListing(entries: [
            FakeFixtures.configEntry("expand.max_tokens", value: .number(300), source: "db"),
            FakeFixtures.configEntry("expand.temperature", value: .number(0.7), source: "db"),
            FakeFixtures.configEntry("gallery.trash_retention_days", value: .number(30), source: "default"),
            FakeFixtures.configEntry("logging.level", value: .string("info"), source: "file"),
        ])

        let rows = AdvancedSettings.rows(listing, query: "expand", refusals: [:])

        #expect(rows.map(\.entry.key) == ["expand.max_tokens", "expand.temperature"])
        #expect(AdvancedSettings.subtitle(showing: rows.count, total: listing.entries.count) == "2 of 4")
    }

    @Test func aPerModelRowSortsAfterEveryEngineKey() {
        let listing = ConfigListing(entries: [
            FakeFixtures.configEntry("zzz.knob", value: .string("x"), source: "default"),
            FakeFixtures.configEntry("models.flux-dev:q8.default_steps", value: .number(20), source: "db"),
            FakeFixtures.configEntry("aaa.knob", value: .string("y"), source: "default"),
            FakeFixtures.configEntry("models.aaa:q8.default_steps", value: .number(20), source: "db"),
        ])

        let rows = AdvancedSettings.rows(listing, query: "", refusals: [:])

        #expect(rows.map(\.entry.key) == [
            "aaa.knob", "zzz.knob",
            "models.aaa:q8.default_steps", "models.flux-dev:q8.default_steps",
        ])
    }

    @Test func aRefusalSitsOnItsOwnRowAndNowhereElse() {
        let listing = ConfigListing(entries: [
            FakeFixtures.configEntry("expand.max_tokens", value: .number(300), source: "db"),
            FakeFixtures.configEntry("gallery.trash_retention_days", value: .number(30), source: "default"),
        ])

        let rows = AdvancedSettings.rows(
            listing, query: "", refusals: ["expand.max_tokens": "Must be between 1 and 8192."])

        let flagged = rows.first { $0.entry.key == "expand.max_tokens" }
        let clean = rows.first { $0.entry.key == "gallery.trash_retention_days" }
        #expect(flagged?.refusal == "Must be between 1 and 8192.")
        #expect(clean?.refusal == nil)
    }

    // MARK: - The pane's whole-pane state

    /// **Fails today** -- there is no `AdvancedSettings` type at all.
    @Test func aMachineWithNoDatabaseReplacesTheTableRatherThanEmptyingIt() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        backend.plantedErrors["config"] =
            MoldClientError.http(status: 503, code: "CONFIG_UNAVAILABLE", message: "no metadata db")
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let store = ConfigStore(hosts: hosts)
        await store.refresh(on: plato.id)

        #expect(AdvancedSettings.resolve(hosts: hosts, store: store, machine: plato.id) == .unavailable)
    }

    @Test func noMachinesShowsTheAddAMachineState() {
        let hosts = HostStore(hosts: []) { _ in FakeBackend(host: MoldHost(name: "x", baseURL: URL(string: "http://x")!)) }
        let store = ConfigStore(hosts: hosts)

        #expect(AdvancedSettings.resolve(hosts: hosts, store: store, machine: nil) == .noMachines)
    }

    @Test func aListingThatHasNotLoadedYetShowsProgress() {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let store = ConfigStore(hosts: hosts)

        #expect(AdvancedSettings.resolve(hosts: hosts, store: store, machine: plato.id) == .loading)
    }
}
