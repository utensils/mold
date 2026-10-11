import Foundation
import MoldClient
import MoldClientTesting
import Testing
@testable import MoldCompanion

@MainActor
struct ConnectionEventFencingTests {
    @Test(arguments: [false, true])
    func aLateOldRouteStatusCannotOverwriteAReplacementOrRestoreARemovedHost(removed: Bool) async throws {
        let fake = FakeBackend()
        let file = HostListFile(url: FileManager.default.temporaryDirectory.appending(path: "event-route-\(UUID()).json"))
        let hosts = HostStore(list: file, credentials: HostStoreTests.MemoryCredentials(), makeBackend: { _ in fake })
        let old = MoldHost(name: "halcyon", baseURL: URL(string: "https://relay.example")!)
        var replacement = old
        replacement.baseURL = URL(string: "http://192.168.1.142:7680")!
        let current = replacement
        let fresh = try MoldJSON.decoder.decode(ServerStatus.self, from: Data(#"{"version":"fresh","busy":false,"uptime_secs":10,"instance_id":"current"}"#.utf8))
        let stale = try MoldJSON.decoder.decode(ServerStatus.self, from: Data(#"{"version":"stale","busy":false,"uptime_secs":1,"instance_id":"old"}"#.utf8))
        hosts.setHosts([old])
        fake.stub("status()") { _ in
            await MainActor.run {
                if removed { hosts.remove(old.id) }
                else {
                    hosts.setHosts([current])
                    hosts.setReachability(.up(fresh), for: old.id)
                }
            }
            return stale
        }
        await hosts.refreshStatus(old)
        #expect(hosts.lastAnswered[old.id] == nil)
        if removed { #expect(hosts.reachability[old.id] == nil) }
        else { #expect(hosts.reachability[old.id] == .up(fresh)) }
    }
}
