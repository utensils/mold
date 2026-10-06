import Foundation
import MoldClient

@testable import Mold

@MainActor final class RemotePairingHarness {
    let host = MoldHost(id: MoldEngine.localHostID, name: "This Mac",
                        baseURL: URL(string: "http://127.0.0.1:7680")!, apiKey: "operator-test")
    let owner = ManagedRelayEnrollment(hostId: String(repeating: "a", count: 32),
        token: String(repeating: "x", count: 43),
        publicUrl: URL(string: "https://\(String(repeating: "a", count: 32)).mold-link.urandom.io")!,
        relayUrl: URL(string: "wss://gateway.example/production")!, expiresAt: 4_102_444_800)
    lazy var fake: FakeBackend = {
        let fake = FakeBackend(host: host)
        fake.pairingSessions = [PairingSession(token: "phone-code", expiresAt: 4_102_444_800,
            authRequired: true, instanceId: "instance", hostname: "This Mac")]
        return fake
    }()
    var calls: [String] = []
    var saved: ManagedRelayEnrollment?
    var optedIn: Bool?
    var refuseClear = false
    var refuseSave = false
    var previous: ManagedRelayEnrollment?
    var alive = false
    var advertised: URL?
    var verificationFailure: (any Error)?
    var cleanupWasCancelled: [Bool] = []
    var holdEnrollment = false
    var gate: CheckedContinuation<ManagedRelayEnrollment, Never>?

    func store() -> RemotePairingStore {
        let hosts = HostStore(hosts: [host]) { [self] _ in fake }
        let pairing = PairingStore(hosts: hosts)
        return RemotePairingStore(pairing: pairing, dependencies: .init(
            load: { [self] in saved },
            loadEnabled: { [self] in optedIn ?? (saved != nil) },
            saveEnabled: { [self] in optedIn = $0 },
            save: { [self] value in
                calls.append(value == nil ? "clear" : "save")
                if value == nil, refuseClear { throw ManagedRelayFailure.unavailable }
                if value != nil, refuseSave { throw ManagedRelayFailure.unavailable }
                saved = value
            },
            enroll: { [self] old in
                calls.append("enroll"); previous = old
                if holdEnrollment { return await withCheckedContinuation { gate = $0 } }
                return owner
            }, revoke: { [self] _ in calls.append("revoke"); cleanupWasCancelled.append(Task.isCancelled) },
            start: { [self] _, _ in calls.append("start"); alive = true },
            stop: { [self] in calls.append("stop"); alive = false },
            isAlive: { [self] in alive },
            advertise: { [self] origin in calls.append(origin == nil ? "withdraw" : "advertise"); advertised = origin },
            verify: { [self] _, _ in calls.append("verify"); if let verificationFailure { throw verificationFailure } }))
    }
}
