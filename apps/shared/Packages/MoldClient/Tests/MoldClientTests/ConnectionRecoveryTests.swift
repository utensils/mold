import Foundation
import Testing
@testable import MoldClient

struct ConnectionRecoveryTests {
    @Test func originalOriginSurvivesButCannotAuthorizeLegacyHTTPFallback() async throws {
        let original = URL(string: "http://workstation.local:7680")!
        let host = MoldHost(name: "Machine", baseURL: URL(string: "https://relay.example")!, apiKey: "old-operator-key",
                            connectionEndpoints: [ConnectionEndpoint(url: "http://192.168.1.2:7680", kind: .lan)],
                            connectionInstanceID: "machine", connectionOriginalURL: original)
        let data = try MoldJSON.localEncoder.encode(StoredHost(host))
        let restored = try MoldJSON.localDecoder.decode(StoredHost.self, from: data).host(apiKey: "old-operator-key")
        #expect(restored.connectionOriginalURL == original)
        await #expect(throws: SecureConnectionError.pairingRequired) { try await HTTPBackend(host: restored).resolvedConnection() }
    }
    @Test func pinnedHostMetadataSurvivesWithoutPrivateDeviceIdentity() throws {
        let invitation = try #require(PairingLinkContractTests.fixtures.links.first?.expected.secureInvitation)
        let secure = SecureHost(binding: invitation.binding, namespace: invitation.namespace, websocketURL: invitation.websocketURL,
                                grant: .init(id: "signed-grant", expiresAt: 4_000_000_000, epoch: 0, signature: "signature"))
        let host = MoldHost(name: "Pinned", baseURL: URL(string: "https://paired.example")!, apiKey: "private-api-key", secure: secure)
        let data = try MoldJSON.localEncoder.encode(StoredHost(host))
        #expect(!String(decoding: data, as: UTF8.self).contains("private-api-key"))
        #expect(!String(decoding: data, as: UTF8.self).contains(invitation.seed))
        #expect(try MoldJSON.localDecoder.decode(StoredHost.self, from: data).host(apiKey: "private-api-key") == host)
    }
}
