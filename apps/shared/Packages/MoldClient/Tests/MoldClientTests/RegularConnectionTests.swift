import Foundation
import Testing
@testable import MoldClient

struct RegularConnectionTests {
    static let token = String(repeating: "a", count: 64)
    static let regularCode = """
    {"type":"mold.mobile-pairing","version":2,"mode":"regular",
     "base_url":"https://gpu.example.com:8443","instance_id":"machine","name":"GPU",
     "pairing_token":"\(token)","expires_at":4000000000}
    """

    @Test func explicitRegularDirectWANCodeParsesAndRoundTrips() throws {
        let payload = try MobilePairingPayload.parse(Self.regularCode)
        #expect(payload.secureInvitation == nil)
        #expect(payload.token == Self.token)
        #expect(payload.baseURL == "https://gpu.example.com:8443")
        let link = try #require(payload.url)
        #expect(link.query == nil)
        #expect(link.fragment?.contains("mode=regular") == true)
        #expect(try MobilePairingPayload.parse(link.absoluteString) == payload)
    }

    @Test func existingDirectMachineKeepsItsRegularCredential() throws {
        let data = Data(#"{"id":"00000000-0000-0000-0000-000000000001","name":"GPU","base_url":"http://gpu.example.com:7680"}"#.utf8)
        let saved = try MoldJSON.localDecoder.decode(StoredHost.self, from: data)
        let host = saved.host(apiKey: "existing-operator-key")
        #expect(!host.requiresSecurePairing)
        #expect(host.apiKey == "existing-operator-key")
    }

    @Test func regularCodeNeverCarriesCloudOrSecureAuthority() throws {
        for extra in ["\"websocket_url\":\"wss://relay.example.com/prod\"",
                      "\"host_key\":\"\(String(repeating: "0", count: 64))\"",
                      "\"endpoints\":[{\"kind\":\"relay\",\"url\":\"https://relay.example.com\"}]"] {
            let code = Self.regularCode.dropLast() + "," + extra + "}"
            #expect(throws: MobilePairingPayload.ParseError.unsupported) {
                try MobilePairingPayload.parse(code)
            }
        }
    }
    @Test func regularTransportSupportsDirectWANAndExistingPathPrefixes() async throws {
        let host = MoldHost(name: "GPU", baseURL: URL(string: "https://gpu.example.com/mold")!,
                            apiKey: "regular-device-key", connectionMode: .regular)
        let backend = HTTPBackend(host: host, session: Self.session())
        let result: Message = try await backend.get("/message")
        #expect(result.value == "regular")
        #expect(try await backend.resolvedConnection() == nil)
        _ = try await backend.playableMedia(for: "movie.mp4")
    }

    @Test func regularClaimUsesOnlyTheOneUseTokenAndChecksReturnedMachine() async throws {
        let payload = try MobilePairingPayload.parse(Self.regularCode)
        let answer = try await HTTPBackend.claimPairing(payload, clientName: "Phone", clientKind: "iphone", session: Self.session())
        #expect(answer.apiKey == "regular-device-key")
        #expect(answer.secure == nil)
        #expect(answer.resolvedBaseURL?.absoluteString == payload.baseURL)
        var object = try #require(JSONSerialization.jsonObject(with: Data(Self.regularCode.utf8)) as? [String: Any])
        object["instance_id"] = "other-machine"
        let wrong = try MobilePairingPayload.parse(String(decoding: JSONSerialization.data(withJSONObject: object), as: UTF8.self))
        await #expect(throws: PairingClaimError.wrongMachine) {
            try await HTTPBackend.claimPairing(wrong, clientName: "Phone", clientKind: "iphone", session: Self.session())
        }
    }

    @Test func regularCannotSendSecureCredentialsOrUseMoldProxy() async throws {
        let secure = try #require(PairingLinkContractTests.fixtures.links.first).expected.secureInvitation
        let pins = try #require(secure)
        let hostPins = SecureHost(binding: pins.binding, namespace: pins.namespace, websocketURL: pins.websocketURL,
                                  grant: .init(id: String(repeating: "a", count: 32), expiresAt: 4_000_000_000, epoch: 0, signature: String(repeating: "a", count: 128)))
        for host in [MoldHost(name: "Proxy", baseURL: URL(string: "https://abc.mold-link.urandom.io")!, apiKey: "never-send", connectionMode: .regular),
                     MoldHost(name: "Mixed", baseURL: URL(string: "https://gpu.example.com")!, apiKey: "never-send", secure: hostPins, connectionMode: .regular)] {
            let backend = HTTPBackend(host: host, session: Self.session())
            await #expect(throws: SecureConnectionError.pairingRequired) { try await backend.status() }
            await #expect(throws: SecureConnectionError.pairingRequired) { try await backend.resolvedConnection() }
            await #expect(throws: SecureConnectionError.pairingRequired) { try await backend.playableMedia(for: "movie.mp4") }
        }
    }

    @Test func regularSessionDecodesCanonicalBaseURL() throws {
        let session = try MoldJSON.decoder.decode(PairingSession.self, from: Data("""
        {"mode":"regular","pairing_token":"\(Self.token)","expires_at":4000000000,
         "auth_required":true,"instance_id":"machine","hostname":"GPU","base_url":"https://gpu.example.com:8443"}
        """.utf8))
        let payload = MobilePairingPayload(session: session, baseURL: URL(string: "http://127.0.0.1:7680")!, name: "GPU")
        #expect(payload?.baseURL == "https://gpu.example.com:8443")
        #expect(payload?.mode == .regular)
    }

    @Test func regularRequestCannotMoveItsCredentialToAnotherOrigin() async throws {
        let host = MoldHost(name: "GPU", baseURL: URL(string: "https://gpu.example.com")!, apiKey: "regular-key", connectionMode: .regular)
        let backend = HTTPBackend(host: host, session: Self.session())
        var request = URLRequest(url: URL(string: "https://different.example/api/status")!)
        request.setValue("regular-key", forHTTPHeaderField: "X-Api-Key")
        await #expect(throws: SecureConnectionError.pairingRequired) { try await backend.response(request) }
    }

    private struct Message: Decodable { let value: String }
    private static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [RegularProtocol.self]
        return URLSession(configuration: config)
    }

}

private final class RegularProtocol: StubTransport {
    override func startLoading() {
        var status = 200
        let body: String
        if request.url?.path == "/api/pairing/regular/claim" {
            if request.value(forHTTPHeaderField: "X-Api-Key") != nil { status = 400 }
            body = #"{"mode":"regular","api_key":"regular-device-key","instance_id":"machine","hostname":"GPU","base_url":"https://gpu.example.com:8443"}"#
        } else if request.url?.path == "/mold/message", request.value(forHTTPHeaderField: "X-Api-Key") == "regular-device-key" {
            body = #"{"value":"regular"}"#
        } else { status = 400; body = "{}" }
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
}
