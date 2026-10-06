import Foundation
import MoldClient
import Testing

@testable import Mold

@MainActor
struct RemotePairingStoreTests {
    @Test func preparationEnrollsBeforeShowingOnlyTheVerifiedPublicCode() async throws {
        let harness = RemotePairingHarness()
        let store = harness.store()
        let phoneHost = try await store.prepare(harness.host)
        #expect(phoneHost.id == harness.host.id)
        #expect(phoneHost.baseURL == harness.owner.publicUrl)
        #expect(harness.calls == ["enroll", "save", "stop", "start", "advertise", "verify"])
        #expect(store.state == .ready(harness.owner.publicUrl))
        #expect(store.pairing.session?.token == "phone-code")
        #expect(store.pairing.session?.token != harness.owner.token)
        #expect(store.enabled)
    }

    @Test func repeatedPreparationReusesTheTunnelAndUnexpiredCode() async throws {
        let harness = RemotePairingHarness()
        let store = harness.store()
        _ = try await store.prepare(harness.host)
        harness.calls = []
        _ = try await store.prepare(harness.host)
        #expect(!harness.calls.contains("enroll"))
        #expect(!harness.calls.contains("start"))
        #expect(!harness.calls.contains("stop"))
        #expect(harness.fake.callCount("pairingSession") == 1)
    }

    @Test func aFailedReadinessProofWithdrawsTheTunnelAndOffersAnError() async {
        let harness = RemotePairingHarness()
        harness.verificationFailure = ManagedRelayFailure.notReady
        let store = harness.store()
        await #expect(throws: ManagedRelayFailure.self) { try await store.prepare(harness.host) }
        #expect(!harness.alive)
        #expect(harness.advertised == nil)
        if case .failed = store.state {} else { Issue.record("No retryable error") }
    }

    @Test func savedEnrollmentIsRenewedAndReusedAfterRelaunch() async throws {
        let harness = RemotePairingHarness()
        harness.saved = harness.owner
        let store = harness.store()
        #expect(store.enabled)
        _ = try await store.prepare(harness.host)
        #expect(harness.previous == harness.owner)
        #expect(store.enrollment?.hostId == harness.owner.hostId)
    }

    @Test func disablingWithdrawsBeforeRevocationAndClearsOwnerMaterial() async throws {
        let harness = RemotePairingHarness()
        let store = harness.store()
        _ = try await store.prepare(harness.host)
        harness.calls = []
        await store.disable()
        #expect(harness.calls == ["stop", "withdraw", "clear", "revoke"])
        #expect(!store.enabled)
        #expect(harness.saved == nil)
        #expect(store.state == .off)
    }

    @Test func quittingStopsTransportButKeepsEnrollmentForPairedPhones() async throws {
        let harness = RemotePairingHarness()
        let store = harness.store()
        _ = try await store.prepare(harness.host)
        harness.calls = []
        store.shutdown()
        #expect(harness.calls == ["stop", "withdraw"])
        #expect(harness.saved == harness.owner)
    }

    @Test func failedSecretCleanupStillRevokesAndCannotEnableAccessOnRelaunch() async throws {
        let harness = RemotePairingHarness()
        let store = harness.store()
        _ = try await store.prepare(harness.host)
        harness.refuseClear = true
        harness.calls = []
        await store.disable()
        #expect(harness.calls.contains("revoke"))
        #expect(!store.enabled)
        #expect(harness.saved != nil)
        #expect(!harness.store().enabled)
        #expect(store.canStopRemoteAccess)
    }

    @Test func anUnsavedNewEnrollmentIsRevokedBeforeReportingFailure() async {
        let harness = RemotePairingHarness()
        harness.refuseSave = true
        let store = harness.store()
        await #expect(throws: ManagedRelayFailure.self) { try await store.prepare(harness.host) }
        #expect(harness.calls.contains("revoke"))
        #expect(harness.cleanupWasCancelled == [false])
        #expect(!harness.calls.contains("start"))
        #expect(!store.enabled)
    }

    @Test func ownerSecretRoundTripsWithoutChangingRelayIdentity() throws {
        let harness = RemotePairingHarness()
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let secrets = SecretStore(directory: directory)
        try ManagedRelayEnrollment.save(harness.owner, secrets: secrets)
        #expect(try ManagedRelayEnrollment.load(secrets: secrets) == harness.owner)
        try ManagedRelayEnrollment.save(nil, secrets: secrets)
        #expect(try ManagedRelayEnrollment.load(secrets: secrets) == nil)
    }

    @Test func concurrentClicksShareOneEnrollmentAndConnector() async throws {
        let harness = RemotePairingHarness()
        harness.holdEnrollment = true
        let store = harness.store()
        let first = Task { try await store.prepare(harness.host) }
        while harness.gate == nil { await Task.yield() }
        let second = Task { try await store.prepare(harness.host) }
        await Task.yield()
        harness.gate?.resume(returning: harness.owner)
        _ = try await first.value
        _ = try await second.value
        #expect(harness.calls.filter { $0 == "enroll" }.count == 1)
        #expect(harness.calls.filter { $0 == "start" }.count == 1)
    }

    @Test func disablingDuringEnrollmentCannotEnableAccessLater() async throws {
        let harness = RemotePairingHarness()
        harness.holdEnrollment = true
        let store = harness.store()
        let first = Task { try await store.prepare(harness.host) }
        while harness.gate == nil { await Task.yield() }
        await store.disable()
        harness.gate?.resume(returning: harness.owner)
        await #expect(throws: CancellationError.self) { try await first.value }
        #expect(!harness.calls.contains("start"))
        #expect(!store.enabled)
        #expect(store.state == .off)
        #expect(harness.calls.contains("revoke"))
        #expect(harness.cleanupWasCancelled == [false])
    }

    @Test func aDeadConnectorReconnectsBeforeItsDailyLeaseRenewal() async throws {
        let harness = RemotePairingHarness()
        let store = harness.store()
        _ = try await store.prepare(harness.host)
        #expect(!store.pendingOperationNeeded(lastRenewal: .now))
        harness.alive = false
        #expect(store.pendingOperationNeeded(lastRenewal: .now))
        harness.calls = []
        _ = try await store.prepare(harness.host)
        #expect(harness.calls.contains("start"))
        #expect(store.state == .ready(harness.owner.publicUrl))
    }

    @Test func enrollmentRejectsForeignOriginsAndOperatorKeys() throws {
        let harness = RemotePairingHarness()
        let owner = harness.owner
        try owner.validate()
        let foreign = ManagedRelayEnrollment(hostId: owner.hostId, token: owner.token,
            publicUrl: URL(string: "https://foreign.invalid")!, relayUrl: owner.relayUrl, expiresAt: owner.expiresAt)
        #expect(throws: ManagedRelayFailure.self) { try foreign.validate() }
        let wrongKey = ManagedRelayEnrollment(hostId: owner.hostId, token: "operator-key",
            publicUrl: owner.publicUrl, relayUrl: owner.relayUrl, expiresAt: owner.expiresAt)
        #expect(throws: ManagedRelayFailure.self) { try wrongKey.validate() }
    }

    @Test func pairingCannotRaceAnUnfinishedStop() async throws {
        let harness = RemotePairingHarness()
        let store = harness.store()
        _ = try await store.prepare(harness.host)
        harness.holdRevocation = true
        let stop = Task { await store.disable() }
        while harness.revocationGate == nil { await Task.yield() }
        await #expect(throws: ManagedRelayFailure.self) { try await store.prepare(harness.host) }
        #expect(!harness.alive)
        harness.revocationGate?.resume()
        await stop.value
        #expect(store.enrollment == nil)
        #expect(!store.enabled)
        harness.holdRevocation = false
        _ = try await store.prepare(harness.host)
        #expect(harness.alive)
        #expect(store.enrollment == harness.owner)
    }
}
