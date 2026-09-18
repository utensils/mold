import Foundation
import MoldClient
import Testing

@testable import Mold

/// M7 S5: pairing, for a keyed machine the app holds an operator key for
/// (design fact 7, decision 12). `PairingSection.resolve` and
/// `PairingSheet.resolve`/`.Countdown.resolve` are pure -- `AccountsSettings`'s
/// own idiom (`AccountsSettings.swift:19-24`) -- so the four section states
/// and the sheet's own two branches are pinned with no rendered view at all.
///
/// The sheet's own tests live in `PairingTests+Sheet.swift`, which shares
/// `machine(_:)` below -- not `private`, the same reason
/// `ConfigStoreTests+Refusals.swift` needs its sibling's helper.
@MainActor
struct PairingTests {
    func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    // MARK: - Section states

    /// **Fails today** -- `PairingSection` does not exist. `pairing_available`
    /// is `true` even on a keyless host (`routes.rs:9678-9685`), so this
    /// plants that exact shape: the gate a naive read would fall for is
    /// right here and still resolves to nothing to show.
    @Test func aKeylessMachineDrawsNoPairingSectionAtAll() {
        let answer = PairedClients(authRequired: false, pairingAvailable: true, clients: [])
        #expect(PairingSection.resolve(answer, authority: nil) == .absent)
    }

    /// A 403 `PAIRING_OPERATOR_REQUIRED` means this app's own key is a
    /// paired one, not an operator's -- caught by CODE in `PairingStore`,
    /// never by status alone.
    @Test func aPairedKeySaysItCannotManageOtherDevices() {
        #expect(PairingSection.resolve(nil, authority: .paired) == .needsOperator)
    }

    /// `pairing_available == false` on an otherwise-keyed host is the
    /// metadata database being off, not a refusal -- `list_paired_clients`
    /// (`routes.rs:9686-9692`) answers 200 with the flag lowered, it never
    /// throws.
    @Test func aMachineWithItsDatabaseOffSaysWhichSwitchToThrow() {
        let answer = PairedClients(authRequired: true, pairingAvailable: false, clients: [])
        #expect(PairingSection.resolve(answer, authority: nil) == .databaseOff)
    }

    /// A stale `.paired` marker from a key that has since been swapped for
    /// an operator one must not survive a listing that actually succeeded.
    @Test func aSuccessfulListingShowsTheClientsThemselves() {
        let client = PairedClient(
            id: "client-1", name: "James's iPhone", clientKind: "mobile", createdAtMs: 0,
            lastUsedAtMs: nil)
        let answer = PairedClients(authRequired: true, pairingAvailable: true, clients: [client])
        #expect(PairingSection.resolve(answer, authority: nil) == .clients([client]))
    }

    // MARK: - Revoke re-reads

    /// **Fails today** -- `PairingStore` does not exist. A revoke is
    /// followed by a re-read, the same "never trust the one row you asked
    /// for" rule `ConfigStore.set` already follows for `expand.*`.
    @Test func aRevokedClientLeavesTheListAndTheMachineWasAsked() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let client = PairedClient(
            id: "client-1", name: "James's iPhone", clientKind: "mobile",
            createdAtMs: 1_700_000_000_000, lastUsedAtMs: nil)
        fake.pairedClientsAnswer = PairedClients(
            authRequired: true, pairingAvailable: true, clients: [client])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let store = PairingStore(hosts: hosts)
        await store.refresh(on: workstation.id)
        #expect(store.byHost[workstation.id]?.clients.count == 1)

        await store.revoke(client, on: workstation.id)

        #expect(fake.revokedClients == ["client-1"])
        #expect(store.byHost[workstation.id]?.clients.isEmpty == true)
        #expect(fake.callCount("pairedClients") == 2)
    }

    /// The 403 path never reaches `hosts.failures` -- `needsOperator` is a
    /// state the section draws, not a banner over a machine that answered
    /// fine.
    @Test func aFourZeroThreeSetsAuthorityWithNoFailureBanner() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.plantedErrors["pairedClients"] = MoldClientError.http(
            status: 403, code: "PAIRING_OPERATOR_REQUIRED", message: "operator key required")
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let store = PairingStore(hosts: hosts)

        await store.refresh(on: workstation.id)

        #expect(store.authority[workstation.id] == .paired)
        #expect(hosts.failures.isEmpty)
    }

    // MARK: - Fixture

    /// **Fails today** -- `PairingStore.Fixture` does not exist. Every
    /// mutating call reports through the same funnel a real refusal would,
    /// the same class of hook as `QueueStore`'s (design M6 decision 27,
    /// here decision 25).
    @Test func theFixtureHookRefusesEveryWrite() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let store = PairingStore(hosts: hosts)
        let client = PairedClient(
            id: "client-1", name: "iPhone", clientKind: "mobile", createdAtMs: 0, lastUsedAtMs: nil)
        let fixture = PairingStore.Fixture(hosts: [
            "workstation": PairingStore.HostFixture(
                clients: PairedClients(authRequired: true, pairingAvailable: true, clients: [client]))
        ])

        store.seed(from: fixture)
        #expect(store.byHost[workstation.id]?.clients.map(\.id) == ["client-1"])
        #expect(store.isSeeded)

        await store.createSession(on: workstation.id)
        await store.revoke(client, on: workstation.id)
        await store.refresh(on: workstation.id)

        #expect(fake.calls.isEmpty)
        #expect(hosts.failures.contains { $0.sentence.contains("fixture") })
        #expect(store.byHost[workstation.id]?.clients.map(\.id) == ["client-1"])
    }

    /// The orchestrator's own addition to S7: a fixture can seed an
    /// in-flight session directly, so `PairingSheet` has a code and a
    /// countdown to draw for a UAT screenshot with no live host at all.
    /// `createSession` still refuses -- seeding is not the same door as a
    /// real request, and the seeded session must survive that refusal
    /// untouched.
    @Test func aFixtureCanSeedAnInFlightSession() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let store = PairingStore(hosts: hosts)
        let session = PairingSession(
            token: "tok", expiresAt: 4_102_444_800, authRequired: true,
            instanceId: "instance-1", hostname: "workstation")
        let fixture = PairingStore.Fixture(hosts: [
            "workstation": PairingStore.HostFixture(
                clients: PairedClients(authRequired: true, pairingAvailable: true, clients: []),
                session: session)
        ])

        store.seed(from: fixture)

        #expect(store.session == session)
        #expect(store.sessionHost == workstation.id)

        await store.createSession(on: workstation.id)

        #expect(fake.calls.isEmpty)
        #expect(store.session == session)
    }

    /// `operator_required: true` seeds the 403 state directly, so the
    /// "this app's key can't manage this machine" screenshot needs no live
    /// host either.
    @Test func aFixtureCanSeedOperatorRequired() {
        let workstation = machine()
        let hosts = HostStore(hosts: [workstation]) { _ in FakeBackend(host: workstation) }
        let store = PairingStore(hosts: hosts)
        let fixture = PairingStore.Fixture(hosts: [
            "workstation": PairingStore.HostFixture(
                clients: PairedClients(authRequired: true, pairingAvailable: true, clients: []),
                operatorRequired: true)
        ])

        store.seed(from: fixture)

        #expect(store.authority[workstation.id] == .paired)
    }
}
