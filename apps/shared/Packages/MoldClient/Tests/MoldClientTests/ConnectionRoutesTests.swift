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

    @Test func proofIsBoundToKindNonceAndIdentity() {
        let nonce = String(repeating: "a", count: 64)
        let proof = ConnectionRoutes.proof(secret: "mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG", kind: "api", nonce: nonce, instanceID: "machine")
        #expect(ConnectionRoutes.verifies(proof, secret: "mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG", kind: "api", nonce: nonce, instanceID: "machine"))
        #expect(!ConnectionRoutes.verifies(proof, secret: "mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG", kind: "pairing", nonce: nonce, instanceID: "machine"))
        #expect(!ConnectionRoutes.verifies(proof, secret: "mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG", kind: "api", nonce: nonce, instanceID: "wrong"))
    }
    @Test func proofMatchesTheSharedProtocolVector() {
        let nonce = String(repeating: "0123456789abcdef", count: 4)
        #expect(ConnectionRoutes.tag("test-only-route-secret") == "e4c8e720b2b762b8")
        #expect(ConnectionRoutes.proof(secret: "test-only-route-secret", kind: "api", nonce: nonce, instanceID: "fixture-machine")
            == "09588691253e789f49c73ec7c6bbe10c6373119985a328df283f2bb610ad6979")
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

@Suite(.serialized)
struct ConnectionProbeTests {
    private func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ConnectionProbeProtocol.self]
        configuration.httpAdditionalHeaders = ["X-Api-Key": "must-not-leak"]
        return URLSession(configuration: configuration)
    }

    @Test func operatorKeysNeverProbeLearnedAddresses() async {
        await #expect(throws: MoldClientError.self) {
            _ = try await ConnectionRoutes.select(endpoints: [ConnectionEndpoint(url: "https://relay.example", kind: .relay)],
                                                  secret: "password", kind: "api", instanceID: "machine", session: session())
        }
    }

    @Test func failedLANFallsBackToRelayWithoutForwardingCredentials() async throws {
        ConnectionProbeProtocol.failLAN = true
        ConnectionProbeProtocol.badProof = false
        let url = try await ConnectionRoutes.select(endpoints: [
            ConnectionEndpoint(url: "http://192.168.1.2:7680", kind: .lan),
            ConnectionEndpoint(url: "https://relay.example", kind: .relay)],
            secret: "mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG", kind: "api", instanceID: "machine", session: session())
        #expect(url.absoluteString == "https://relay.example")
    }

    @Test func pairingPrefersLANWhenAllAdvertisedRoutesAnswer() async throws {
        ConnectionProbeProtocol.failLAN = false
        ConnectionProbeProtocol.failTailscale = false
        ConnectionProbeProtocol.badProof = false
        let route = try await selectPairingRoute(relay: "https://lan-preferred.example")
        #expect(route.absoluteString == "http://192.168.1.2:7680")
    }

    @Test func pairingUsesTailscaleWhenLANIsUnavailable() async throws {
        ConnectionProbeProtocol.failLAN = true
        ConnectionProbeProtocol.failTailscale = false
        ConnectionProbeProtocol.badProof = false
        let route = try await selectPairingRoute(relay: "https://tailscale-preferred.example")
        #expect(route.absoluteString == "http://100.64.1.2:7680")
    }

    @Test func pairingUsesProxyWhenBothDirectRoutesAreUnavailable() async throws {
        ConnectionProbeProtocol.failLAN = true
        ConnectionProbeProtocol.failTailscale = true
        ConnectionProbeProtocol.badProof = false
        defer { ConnectionProbeProtocol.failTailscale = false }
        let route = try await selectPairingRoute(relay: "https://proxy-fallback.example")
        #expect(route.absoluteString == "https://proxy-fallback.example")
    }

    private func selectPairingRoute(relay: String) async throws -> URL {
        // Deliberately advertise proxy first: selection must use route kind,
        // not QR payload order. Each test gets a separate route-memory key.
        try await ConnectionRoutes.select(endpoints: [
            ConnectionEndpoint(url: relay, kind: .relay),
            ConnectionEndpoint(url: "http://100.64.1.2:7680", kind: .tailscale),
            ConnectionEndpoint(url: "http://192.168.1.2:7680", kind: .lan)],
            secret: "mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG",
            kind: "pairing", instanceID: "machine", session: session())
    }

    @Test func directRouteWinsAndWrongProofIsNeverAccepted() async throws {
        ConnectionProbeProtocol.failLAN = false
        ConnectionProbeProtocol.badProof = false
        let routes = [ConnectionEndpoint(url: "https://relay.example", kind: .relay),
                      ConnectionEndpoint(url: "http://192.168.1.2:7680", kind: .lan)]
        let url = try await ConnectionRoutes.select(endpoints: routes, secret: "mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG", kind: "api",
                                                     instanceID: "machine", session: session())
        #expect(url.host == "192.168.1.2")
        ConnectionProbeProtocol.badProof = true
        await #expect(throws: (any Error).self) {
            _ = try await ConnectionRoutes.select(endpoints: routes, secret: "mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG", kind: "api",
                                                   instanceID: "machine", session: session())
        }
    }
}

private final class ConnectionProbeProtocol: URLProtocol {
    nonisolated(unsafe) static var failLAN = false
    nonisolated(unsafe) static var failTailscale = false
    nonisolated(unsafe) static var badProof = false
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}
    override func startLoading() {
        #expect(request.value(forHTTPHeaderField: "X-Api-Key") == nil)
        #expect(request.value(forHTTPHeaderField: "Cookie") == nil)
        #expect(request.url?.path == "/api/connection-probe")
        if (Self.failLAN && request.url?.host == "192.168.1.2")
            || (Self.failTailscale && request.url?.host == "100.64.1.2") {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
            return
        }
        let data = request.httpBody ?? request.httpBodyStream.map { stream in
            stream.open(); defer { stream.close() }
            var bytes = [UInt8](repeating: 0, count: 1024)
            let count = stream.read(&bytes, maxLength: bytes.count)
            return Data(bytes.prefix(max(0, count)))
        } ?? Data()
        let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: String] ?? [:]
        #expect(body["key_tag"] == ConnectionRoutes.tag("mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"))
        #expect(!String(decoding: data, as: UTF8.self).contains("mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"))
        let proof = Self.badProof ? "wrong" : ConnectionRoutes.proof(secret: "mold_pair_abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG", kind: body["kind"] ?? "",
                                                                       nonce: body["nonce"] ?? "", instanceID: "machine")
        let answer = try! JSONSerialization.data(withJSONObject: ["instance_id": "machine", "proof": proof, "version": 1])
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: answer)
        client?.urlProtocolDidFinishLoading(self)
    }
}
