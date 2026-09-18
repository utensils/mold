import Foundation
import MoldClient
import Testing

@testable import Mold

/// What each machine refused, kept on the row it is about -- split out of
/// `ConfigStoreTests.swift` past the file-size advisory (`make lint`,
/// `lint-size`), sharing that file's `machine(_:)` helper.
@MainActor
extension ConfigStoreTests {
    // MARK: - Refusals stay on their row

    /// **Fails today** -- `ModelDefaultsStore` has no `set` and no
    /// `refusals` at all. `output_dir`'s 409 `RESTART_REQUIRED`
    /// (`routes_config.rs:180-191`) is the refusal every live server gives,
    /// whatever the value, and it must land on the row, never on
    /// `hosts.failures` -- that banner is for the MACHINE, not for a key it
    /// refused to change.
    @Test func aKeyTheMachineRefusesKeepsItsMessageOnThatRowAlone() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = FakeFixtures.configListing()
        backend.plantedErrors["setConfig"] = MoldClientError.http(
            status: 409, code: "RESTART_REQUIRED",
            message: "'output_dir' is fixed for the lifetime of mold serve")
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)
        await store.refresh(on: workstation.id)

        let ok = await store.set("output_dir", to: .string("/new/path"), on: workstation.id)

        #expect(ok == false)
        #expect(store.refusal(for: "output_dir", on: workstation.id)?.code == "RESTART_REQUIRED")
        #expect(hosts.failures.isEmpty)
    }

    /// An env-owned key's 403 `ENV_OVERRIDDEN` (`routes_config.rs:192-197`)
    /// is the same kind of refusal, naming the variable.
    @Test func anEnvOwnedKeysRefusalAlsoStaysOffTheBanner() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = FakeFixtures.configListing()
        backend.plantedErrors["setConfig"] = MoldClientError.http(
            status: 403, code: "ENV_OVERRIDDEN",
            message: "'models_dir' is set by MOLD_MODELS_DIR in the environment — unset it to edit")
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)
        await store.refresh(on: workstation.id)

        let ok = await store.set("models_dir", to: .string("/elsewhere"), on: workstation.id)

        #expect(ok == false)
        #expect(store.refusal(for: "models_dir", on: workstation.id)?.code == "ENV_OVERRIDDEN")
        #expect(hosts.failures.isEmpty)
    }

    /// A 422's prose carries the bounds -- they are literal arguments at
    /// each `set_value` arm (`config_keys.rs:652`) and exist nowhere else,
    /// so the row must show exactly what the machine said, not a generic
    /// "invalid value".
    @Test func aFourTwentyTwoKeepsItsOwnSentence() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = FakeFixtures.configListing()
        backend.plantedErrors["setConfig"] = MoldClientError.http(
            status: 422, code: nil, message: "must be between 1 and 100 (got 200)")
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)
        await store.refresh(on: workstation.id)

        let ok = await store.set("expand.max_tokens", to: .number(200), on: workstation.id)

        #expect(ok == false)
        #expect(store.refusal(for: "expand.max_tokens", on: workstation.id)?.sentence
            == "Must be between 1 and 100 (got 200)")
    }

    /// Editing the row again -- successfully this time -- drops the stale
    /// message rather than leaving it under a field that no longer needs
    /// it.
    @Test func editingARefusedRowAgainClearsItsMessage() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        backend.configListing = FakeFixtures.configListing()
        backend.plantedErrors["setConfig"] = MoldClientError.http(
            status: 422, code: nil, message: "must be between 1 and 100 (got 200)")
        let hosts = HostStore(hosts: [workstation]) { _ in backend }
        let store = ConfigStore(hosts: hosts)
        await store.refresh(on: workstation.id)
        await store.set("expand.max_tokens", to: .number(200), on: workstation.id)
        #expect(store.refusal(for: "expand.max_tokens", on: workstation.id) != nil)

        backend.plantedErrors["setConfig"] = nil
        let ok = await store.set("expand.max_tokens", to: .number(50), on: workstation.id)

        #expect(ok == true)
        #expect(store.refusal(for: "expand.max_tokens", on: workstation.id) == nil)
    }
}
