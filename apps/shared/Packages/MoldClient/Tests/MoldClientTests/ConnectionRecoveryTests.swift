import Foundation
import Testing
@testable import MoldClient

struct ConnectionRecoveryTests {
    @Test func originalApprovedHostnameSurvivesPersistenceAndFailedProbes() async throws {
        let original = URL(string: "http://workstation.local:7680")!
        let host = MoldHost(name: "Machine", baseURL: URL(string: "https://relay.example")!, apiKey: "mold_pair_" + String(repeating: "a", count: 43),
                            connectionEndpoints: [ConnectionEndpoint(url: "http://192.168.1.2:7680", kind: .lan)],
                            connectionInstanceID: "machine", connectionOriginalURL: original)
        let data = try MoldJSON.localEncoder.encode(StoredHost(host))
        let restored = try MoldJSON.localDecoder.decode(StoredHost.self, from: data).host(apiKey: "mold_pair_" + String(repeating: "a", count: 43))
        #expect(restored.connectionOriginalURL == original)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FailedRouteProtocol.self]
        let recovered = try await HTTPBackend(host: restored, session: URLSession(configuration: configuration)).resolvedConnection()
        #expect(recovered?.baseURL == original)
        #expect(recovered?.connectionOriginalURL == original)
    }
    @Test func operatorKeysNeverProbeAndRetainOriginalOrigin() async throws {
        let original = URL(string: "http://workstation.local:7680")!
        let host = MoldHost(name: "Machine", baseURL: URL(string: "https://relay.example")!, apiKey: "operator",
                            connectionEndpoints: [ConnectionEndpoint(url: "https://relay.example", kind: .relay)],
                            connectionInstanceID: "machine", connectionOriginalURL: original)
        let recovered = try await HTTPBackend(host: host).resolvedConnection()
        #expect(recovered?.baseURL == original)
    }
    @Test func cancellationNeverFallsBack() async throws {
        let host = MoldHost(name: "Machine", baseURL: URL(string: "http://workstation.local:7680")!, apiKey: "mold_pair_" + String(repeating: "a", count: 43),
                            connectionEndpoints: [ConnectionEndpoint(url: "http://192.168.1.2:7680", kind: .lan)],
                            connectionInstanceID: "machine")
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FailedRouteProtocol.self]
        let task = Task { try await HTTPBackend(host: host, session: URLSession(configuration: configuration)).resolvedConnection() }
        task.cancel()
        do { _ = try await task.value; Issue.record("cancelled resolution must not return a fallback") } catch {}
    }
}
private final class FailedRouteProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() { client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost)) }
    override func stopLoading() {}
}
