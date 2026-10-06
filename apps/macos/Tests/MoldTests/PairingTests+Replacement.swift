import Foundation
import MoldClient
import Testing

@testable import Mold

extension PairingTests {
    @Test func failedReplacementNeverShowsThePreviousCode() async {
        let host = machine()
        let fake = FakeBackend(host: host)
        fake.pairingSessions = [PairingSession(token: "old-token", expiresAt: 4_102_444_800,
                                             authRequired: true, instanceId: "instance", hostname: host.name)]
        let hosts = HostStore(hosts: [host]) { _ in fake }
        let store = PairingStore(hosts: hosts)
        await store.createSession(on: host.id)
        #expect(store.session?.token == "old-token")
        fake.plantedErrors["pairingSession"] = URLError(.cannotConnectToHost)
        await store.createSession(on: host.id)
        #expect(store.session == nil)
        #expect(store.sessionHost == nil)
        #expect(store.sessionFailure[host.id] != nil)
    }
}
