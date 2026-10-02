import Foundation
import MoldClient
import MoldClientTesting
import Testing
@testable import MoldCompanion

@MainActor
struct ConnectionPersistenceTests {
    @Test func roamingRetainsOneMachineAndItsKeyAcrossLaunches() throws {
        let file = HostListFile(url: FileManager.default.temporaryDirectory.appending(path: "routes-\(UUID()).json"))
        let credentials = HostStoreTests.MemoryCredentials()
        let host = MoldHost(name: "HAL9000", baseURL: URL(string: "http://192.168.1.2:7680")!, apiKey: "test-key")
        try credentials.setAPIKey("test-key", for: host.id)
        let hosts = HostStore(list: file, credentials: credentials, makeBackend: { _ in FakeBackend() })
        hosts.setHosts([host])
        var routed = host
        routed.baseURL = URL(string: "https://relay.example")!
        routed.connectionInstanceID = "hal-identity"
        routed.connectionEndpoints = [ConnectionEndpoint(url: routed.baseURL.absoluteString, kind: .relay)]
        hosts.applyConnection(routed, expectedURL: host.baseURL)
        let restored = HostStore(list: file, credentials: credentials, makeBackend: { _ in FakeBackend() })
        #expect(restored.hosts.count == 1)
        #expect(restored.host(host.id)?.apiKey == "test-key")
        #expect(restored.host(host.id)?.connectionEndpoints == routed.connectionEndpoints)
        #expect(restored.host(host.id)?.connectionInstanceID == "hal-identity")
        #expect(!String(decoding: try Data(contentsOf: file.url), as: UTF8.self).contains("test-key"))
    }
}
