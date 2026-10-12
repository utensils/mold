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

    @Test func aDecodedHostWithoutModeDoesNotOptIntoRegular() throws {
        let original = MoldHost(name: "GPU", baseURL: URL(string: "http://gpu.example.com:7680")!, apiKey: "old-key")
        let encoded = try MoldJSON.localEncoder.encode(original)
        var object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "connectionMode")
        let decoded = try MoldJSON.localDecoder.decode(MoldHost.self, from: JSONSerialization.data(withJSONObject: object))
        #expect(decoded.connectionMode == .secure)
        #expect(decoded.requiresSecurePairing)
        #expect(!decoded.permitsRegularDirect)
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

    @Test func knownSecureCredentialCannotCrossAnAliasInRegularHeaders() async throws {
        let invitation = try #require(PairingLinkContractTests.fixtures.links.first?.expected.secureInvitation)
        let pins = SecureHost(binding: invitation.binding, namespace: invitation.namespace, websocketURL: invitation.websocketURL,
                              grant: .init(id: String(repeating: "a", count: 32), expiresAt: 4_000_000_000, epoch: 0, signature: String(repeating: "a", count: 128)))
        let secret = "secure-" + UUID().uuidString
        _ = HTTPBackend(host: MoldHost(name: "Secure", baseURL: URL(string: "https://secure.example")!, apiKey: secret, secure: pins))
        let regular = MoldHost(name: "Alias", baseURL: URL(string: "https://gpu.example.com")!, apiKey: "regular-key", connectionMode: .regular)
        let backend = HTTPBackend(host: regular, session: Self.session())
        for (header, value) in [("X-API-Key", secret), ("Authorization", "bEaReR  " + secret),
                                ("Authorization", "Bearer harmless, Bearer " + secret),
                                ("X-API-Key", "harmless, " + secret)] {
            var request = URLRequest(url: regular.baseURL.appending(path: "api/status"))
            request.setValue(value, forHTTPHeaderField: header)
            await #expect(throws: SecureConnectionError.pairingRequired) { try await backend.response(request) }
        }
        var reused = regular; reused.apiKey = secret
        await #expect(throws: SecureConnectionError.pairingRequired) {
            try await HTTPBackend(host: reused, session: Self.session()).status()
        }
    }

    @Test func secureCredentialDenialHistorySurvivesReplacementAndRelaunchWithoutSecrets() throws {
        let suite = "regular-history-" + UUID().uuidString
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let invitation = try #require(PairingLinkContractTests.fixtures.links.first?.expected.secureInvitation)
        let pins = SecureHost(binding: invitation.binding, namespace: invitation.namespace, websocketURL: invitation.websocketURL,
                              grant: .init(id: String(repeating: "a", count: 32), expiresAt: 4_000_000_000, epoch: 0, signature: String(repeating: "a", count: 128)))
        let secret = "secure-private-key"
        var host = MoldHost(name: "Secure", baseURL: URL(string: "https://secure.example")!, apiKey: secret, secure: pins)
        SecureCredentialHistory(defaults: defaults).remember([host])
        host.secure = nil; host.connectionMode = .regular; host.baseURL = URL(string: "http://alias.example")!
        let restored = SecureCredentialHistory(defaults: defaults)
        restored.remember([host])
        #expect(throws: SecureConnectionError.pairingRequired) {
            try restored.refuseKnownSecureCredentials(host: host, request: URLRequest(url: host.baseURL), session: Self.session())
        }
        host.apiKey = "regular-device-key"
        try restored.refuseKnownSecureCredentials(host: host, request: URLRequest(url: host.baseURL), session: Self.session())
        for header in ["Cookie", "X-Mold-Secure-Peer", "X-Mold-Secure-Control"] {
            let config = URLSessionConfiguration.ephemeral
            config.httpAdditionalHeaders = [header: "private-session-value"]
            #expect(throws: SecureConnectionError.pairingRequired) {
                try restored.refuseKnownSecureCredentials(host: host, request: URLRequest(url: host.baseURL), session: URLSession(configuration: config))
            }
        }
        let stored = try #require(defaults.stringArray(forKey: SecureCredentialHistory.storageKey))
        #expect(stored.count == 1)
        #expect(stored[0].count == 64)
        #expect(!String(describing: defaults.dictionaryRepresentation()).contains(secret))
        let config = URLSessionConfiguration.ephemeral
        config.httpAdditionalHeaders = ["Authorization": "Bearer " + secret]
        #expect(throws: SecureConnectionError.pairingRequired) {
            try restored.refuseKnownSecureCredentials(host: host, request: URLRequest(url: host.baseURL), session: URLSession(configuration: config))
        }
    }

    @Test func regularRequestsStripCookieAndSecureProofHeaders() async throws {
        let host = MoldHost(name: "GPU", baseURL: URL(string: "https://gpu.example.com/mold")!, apiKey: "regular-device-key", connectionMode: .regular)
        let backend = HTTPBackend(host: host, session: Self.session())
        var request = backend.request("/message")
        for header in ["Cookie", "X-Mold-Secure-Peer", "X-Mold-Secure-Control"] { request.setValue("private-value", forHTTPHeaderField: header) }
        let (_, response) = try await backend.response(request)
        #expect(response.statusCode == 200)
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

    @Test func regularServerSessionProducesNativePairingCode() throws {
        // Match the owner-authenticated server response, without synthesizing
        // Secure-only fields in the native fixture.
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appending(path: "Fixtures/regular-pairing-session.json")
        let session = try MoldJSON.decoder.decode(PairingSession.self, from: Data(contentsOf: fixture))
        #expect(session.authRequired)
        let payload = try #require(MobilePairingPayload(session: session, baseURL: URL(string: "http://127.0.0.1:7680")!, name: "GPU"))
        #expect(payload.mode == .regular)
        #expect(payload.baseURL == "http://192.168.1.142:7680")
        #expect(payload.url != nil)
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
            if ["Cookie", "X-Mold-Secure-Peer", "X-Mold-Secure-Control"].contains(where: { request.value(forHTTPHeaderField: $0) != nil }) { status = 400 }
            body = #"{"value":"regular"}"#
        } else { status = 400; body = "{}" }
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
}
