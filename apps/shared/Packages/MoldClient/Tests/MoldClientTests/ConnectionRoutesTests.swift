import Foundation
import Testing
@testable import MoldClient

struct ConnectionRoutesTests {
    @Test func localOnlyAddressesAreRejected() {
        for host in ["localhost", "test.localhost", "127.0.0.2", "0.0.0.0", "[::]", "[::1]", "169.254.1.2", "[fe80::1]"] {
            #expect(ConnectionRoutes.sanitized([ConnectionEndpoint(url: "http://\(host):7680", kind: .lan)]).isEmpty)
        }
    }

    @Test func advertisedOriginsAreBoundedAndCredentialFree() {
        let endpoints = [
            ConnectionEndpoint(url: "http://192.168.1.2:7680", kind: .lan),
            ConnectionEndpoint(url: "https://relay.example", kind: .relay),
            ConnectionEndpoint(url: "https://secret@wrong.example", kind: .relay),
            ConnectionEndpoint(url: "https://wrong.example/path?key=x", kind: .relay)]
        #expect(ConnectionRoutes.sanitized(endpoints).count == 2)
    }
    @Test func ordinaryHostnamesAreNotMistakenForLinkLocalIPv6() {
        #expect(ConnectionRoutes.sanitized([ConnectionEndpoint(url: "https://features.example", kind: .relay)]).count == 1)
    }

    @Test func oversizedOriginsAreRejectedConsistently() {
        let url = "https://" + String(repeating: "x", count: 2050) + ".example"
        #expect(ConnectionRoutes.sanitized([ConnectionEndpoint(url: url, kind: .relay)]).isEmpty)
    }

    @Test func savedHostRetainsRoutesWithoutCredentials() throws {
        let host = MoldHost(name: "Machine", baseURL: URL(string: "http://192.168.1.2:7680")!, apiKey: "secret",
                            connectionEndpoints: [ConnectionEndpoint(url: "https://relay.example", kind: .relay)],
                            connectionInstanceID: "machine")
        let data = try MoldJSON.localEncoder.encode(StoredHost(host))
        #expect(!String(decoding: data, as: UTF8.self).contains("secret"))
        let restored = try MoldJSON.localDecoder.decode(StoredHost.self, from: data).host(apiKey: "secret")
        #expect(restored.connectionEndpoints == host.connectionEndpoints)
        #expect(restored.connectionInstanceID == "machine")
    }
}

struct SecureRouteRaceTests {
    @Test func fastLANSkipsCloudAndCancelsSameTierHang() async throws {
        let calls = RouteCalls()
        let endpoints: [ConnectionEndpoint] = [.init(url: "fast", kind: .lan), .init(url: "hung", kind: .lan), .init(url: "cloud", kind: .relay)]
        let (index, _) = try await ConnectionRoutes.select(endpoints, connect: { endpoint in
            await calls.record(endpoint.url)
            if endpoint.url == "hung" { try await Task.sleep(for: .seconds(30)) }
            return endpoint.url
        }, close: { _ in })
        #expect(index == 0)
        #expect(!(await calls.values).contains("cloud"))
    }
    @Test func slowerLANStillBeatsReadyCloud() async throws {
        let calls = RouteCalls()
        let endpoints: [ConnectionEndpoint] = [.init(url: "lan", kind: .lan), .init(url: "cloud", kind: .relay)]
        let (index, _) = try await ConnectionRoutes.select(endpoints, connect: { endpoint in
            if endpoint.kind == .lan { try await Task.sleep(for: .milliseconds(400)) }
            return endpoint.url
        }, close: { await calls.record($0) })
        #expect(index == 0)
        #expect(await calls.values == ["cloud"])
    }
    @Test func unreachableLANFallsBackWithoutReplayingARequest() async throws {
        let (index, _) = try await ConnectionRoutes.select([.init(url: "lan", kind: .lan), .init(url: "cloud", kind: .relay)], connect: { endpoint in
            if endpoint.kind == .lan { throw URLError(.cannotConnectToHost) }
            return endpoint.url
        }, close: { _ in })
        #expect(index == 1)
    }
}
private actor RouteCalls {
    var values: [String] = []
    func record(_ value: String) { values.append(value) }
}
