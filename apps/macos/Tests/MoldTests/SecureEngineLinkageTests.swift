import Foundation
import Testing
import CMoldSecure
@testable import MoldClient

/// This target links the shipping engine as well as the standalone crypto archive.
/// Package-only tests cannot detect duplicate runtimes or cross-library ownership.
struct SecureEngineLinkageTests {
    @Test func combinedEngineAndClientPerformNoiseAndDestroyTheirOwnHandles() throws {
        for _ in 0..<32 {
            let host = try SecureIdentity(), device = try SecureIdentity()
            let hostKey = try host.binding(), deviceKey = try device.binding()
            let namespace = try SecureAdmission.namespace(hostKey)
            let client = try SecureNoiseSession(identity: device, namespace: namespace, host: hostKey, sid: "linked-engine", rid: "device")
            let server = try SecureNoiseSession(identity: host, namespace: namespace, host: hostKey, sid: "linked-engine", rid: "device", initiator: false, device: deviceKey)
            try server.readHandshake(client.writeHandshake())
            try client.readHandshake(server.writeHandshake())
            try server.readHandshake(client.writeHandshake())
            let record = SecureHTTPRecord(.data, 1, Data(repeating: 42, count: 16_320))
            #expect(try server.open(client.seal(record)) == record)
            #expect(try client.handshakeHash() == server.handshakeHash())
            mold_secure_release(client.handle)
            #expect(throws: SecureConnectionError.authenticationFailed) { try client.seal(record) }
            // The Swift owner subsequently releases the same handle during deinit.
        }
    }
    @Test func malformedInputDoesNotPoisonTheStandaloneRuntime() throws {
        for _ in 0..<32 {
            let identity = try SecureIdentity()
            let pin = try identity.binding()
            let session = try SecureNoiseSession(identity: identity, namespace: SecureAdmission.namespace(pin), host: pin, sid: "bad", rid: "bad")
            #expect(throws: SecureConnectionError.authenticationFailed) { try session.readHandshake(Data([0, 1, 2])) }
            try SecureCrypto.verify(SecureIdentity().binding())
        }
    }
}
