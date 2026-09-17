import Foundation
import MoldClient
import Testing

@testable import Mold

/// Settings ▸ Accounts (design M5 S7): the pure `AccountsRow` a provider's
/// state resolves to, the pure gate that hides the pane on a machine that
/// can't browse a catalog, and `CatalogStore`'s save/clear round trip.
@MainActor
struct AccountsTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    // MARK: - AccountsRow

    /// **Fails today**: `AccountsRow` does not exist.
    @Test func anEnvironmentTokenReadsAsFromTheEnvironmentAndOffersNoClear() {
        let status = FakeFixtures.credentialsFixture()
        let row = AccountsRow.resolve(status.hf)
        guard case let .environment(masked) = row else {
            Issue.record("expected .environment, got \(row)")
            return
        }
        #expect(masked == "hf_••••hhml")
        #expect(row.offersClear == false)
    }

    @Test func aStoredTokenOffersClear() {
        let status = FakeFixtures.credentialStatus(
            hfConfigured: true, hfSource: "server", hfMasked: "hf_••••9999")
        let row = AccountsRow.resolve(status.hf)
        guard case let .stored(masked) = row else {
            Issue.record("expected .stored, got \(row)")
            return
        }
        #expect(masked == "hf_••••9999")
        #expect(row.offersClear)
    }

    @Test func anUnconfiguredProviderOffersNoClearEither() {
        let status = FakeFixtures.credentialsFixture()
        #expect(AccountsRow.resolve(status.civitai) == .unset)
        #expect(AccountsRow.resolve(status.civitai).offersClear == false)
    }

    /// The server 400s an empty token (`catalog_credentials.rs:293-295`), so
    /// Save is inert on a blank field rather than asking and getting refused.
    @Test func aBlankTokenCannotBeSaved() {
        #expect(AccountsRow.canSave(token: "") == false)
        #expect(AccountsRow.canSave(token: "   ") == false)
        #expect(AccountsRow.canSave(token: "hf_x") == true)
    }

    // MARK: - Availability

    /// **Fails today**: `AccountsSettings` does not exist.
    @Test func aMachineThatCannotBrowseShowsTheSentence() {
        #expect(AccountsSettings.showsProviders(capabilities: nil) == false)
        #expect(AccountsSettings.showsProviders(capabilities: FakeFixtures.capabilities(catalog: false)) == false)
        #expect(AccountsSettings.showsProviders(capabilities: FakeFixtures.capabilities(catalog: true)))
        #expect(AccountsSettings.noCatalogSentence == "This machine doesn't browse a catalog.")
    }

    // MARK: - CatalogStore round trip

    /// The masked value is the only form of the token this app ever reads
    /// back, from the same fixture plato itself answers with.
    @Test func theMaskedValueIsTheOnlyTokenFormEverRead() {
        let status = FakeFixtures.credentialsFixture()
        #expect(status.hf.masked == "hf_••••hhml")
        #expect(status.hf.configured)
        #expect(status.hf.isFromEnvironment)
        #expect(status.civitai.configured == false)
        #expect(status.civitai.masked == nil)
    }

    @Test func savingWritesTheProviderAndTokenOnceAndRereadsTheAnswer() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.credentialStatus = FakeFixtures.credentialStatus(
            hfConfigured: true, hfSource: "server", hfMasked: "hf_••••9999")
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        let ok = await catalog.saveCredential("hf", token: "hf_supersecret9999", on: plato.id)

        #expect(ok)
        #expect(fake.credentialWrites.count == 1)
        #expect(fake.credentialWrites.first?.provider == "hf")
        #expect(fake.credentialWrites.first?.token == "hf_supersecret9999")
        // Re-read from the answer, never a second fetch.
        #expect(fake.calls.filter { $0 == "catalogCredentials" }.isEmpty)
        #expect(catalog.credentials(on: plato.id)?.hf.masked == "hf_••••9999")
    }

    /// Two machines, one write -- the picker names the target, not a global.
    @Test func savingOnOneMachineDoesNotTouchAnother() async {
        let plato = machine("plato")
        let hal = machine("hal9000")
        let fakePlato = FakeBackend(host: plato)
        fakePlato.credentialStatus = FakeFixtures.credentialStatus(
            hfConfigured: true, hfSource: "server", hfMasked: "hf_••••1111")
        let fakeHal = FakeBackend(host: hal)
        let hosts = HostStore(hosts: [plato, hal]) { host in host.id == plato.id ? fakePlato : fakeHal }
        let catalog = CatalogStore(hosts: hosts)

        await catalog.saveCredential("hf", token: "hf_1111", on: plato.id)

        #expect(fakePlato.credentialWrites.count == 1)
        #expect(fakeHal.credentialWrites.isEmpty)
    }

    @Test func clearingDeletesAndRereadsTheFallback() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.credentialStatus = FakeFixtures.credentialsFixture() // falls back to the environment
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        await catalog.clearCredential("hf", on: plato.id)

        #expect(fake.credentialClears == ["hf"])
        #expect(catalog.credentials(on: plato.id)?.hf.isFromEnvironment == true)
    }

    @Test func aFailedSaveReportsAndLeavesTheStoredStateAlone() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.refuses = ["setCatalogCredential"]
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let catalog = CatalogStore(hosts: hosts)

        let ok = await catalog.saveCredential("hf", token: "hf_x", on: plato.id)

        #expect(ok == false)
        #expect(catalog.credentials(on: plato.id) == nil)
        #expect(hosts.failures.contains { $0.host == plato.id && $0.verb == "save the Hugging Face token" })
    }
}
