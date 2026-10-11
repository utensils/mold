import Foundation
import Testing
@testable import MoldClient

struct SecureBoundaryTests {
    @Test func legacyRemoteFailsBeforeSendingAnHTTPRequest() async {
        let backend = HTTPBackend(host: MoldHost(name: "Legacy", baseURL: URL(string: "https://untrusted.invalid")!, apiKey: "must-not-leave"))
        await #expect(throws: SecureConnectionError.pairingRequired) { try await backend.status() }
        await #expect(throws: SecureConnectionError.pairingRequired) { try await backend.resolvedConnection() }
        await #expect(throws: SecureConnectionError.pairingRequired) { try await backend.playableMedia(for: "movie.mp4") }
    }
    @Test func embeddedExceptionIsExactNumericLoopbackOnly() {
        for host in ["127.0.0.1", "[::1]"] { #expect(HTTPBackend.isEmbeddedLoopback(URL(string: "http://\(host):7680")!)) }
        for host in ["localhost", "127.0.0.2", "192.168.1.2", "127.0.0.1.example"] { #expect(!HTTPBackend.isEmbeddedLoopback(URL(string: "http://\(host):7680")!)) }
        #expect(!HTTPBackend.isEmbeddedLoopback(URL(string: "http://credential@127.0.0.1:7680")!))
    }
    @Test func canonicalAdmissionMatchesTheSharedCloudFixture() throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appending(path: "../../../../../../relay/aws/test/fixtures/admission-v3.json").standardizedFileURL
        let object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? [String: Any])
        let challenge = try #require(object["challenge"] as? [String: Any]), value = try #require(challenge["value"] as? [String: Any])
        let bytes = try SecureAdmission.join(namespace: value["namespace"] as! String, role: 2,
            connectionID: value["connectionId"] as! String, nonce: value["nonce"] as! String,
            device: .init(signingKey: value["signingKey"] as! String, noiseKey: "", signature: ""), expiresAt: 1_900_000_000)
        #expect(SecureHex.encode(bytes) == challenge["bytes"] as? String)
        try SecureCrypto.verify(key: value["signingKey"] as! String, message: bytes, signature: challenge["signature"] as! String)
    }
}
