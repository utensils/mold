import Foundation
import Testing
@testable import MoldClient

struct SecureCryptoTests {
    @Test func sameRustABIAuthenticatesBothPeersAndEncryptsRecords() throws {
        let host = try SecureIdentity(), device = try SecureIdentity()
        let hostBinding = try host.binding(), deviceBinding = try device.binding()
        try SecureCrypto.verify(hostBinding)
        let namespace = String(TransportDigest.sha256(try SecureHex.decode(hostBinding.signingKey, count: 32)).prefix(32))
        let client = try SecureNoiseSession(identity: device, namespace: namespace, host: hostBinding, sid: "epoch", rid: "peer")
        let server = try SecureNoiseSession(identity: host, namespace: namespace, host: hostBinding, sid: "epoch", rid: "peer", initiator: false, device: deviceBinding)
        try server.readHandshake(client.writeHandshake())
        try client.readHandshake(server.writeHandshake())
        try server.readHandshake(client.writeHandshake())
        #expect(client.ready && server.ready)
        let request = SecureHTTPRecord(.data, 1, Data("private payload".utf8))
        let sealed = try client.seal(request)
        #expect(sealed.range(of: request.payload) == nil)
        #expect(try server.open(sealed) == request)
        #expect(throws: SecureConnectionError.authenticationFailed) { try server.open(sealed) }
    }

    @Test func wrongPinnedKeyStopsBeforeAnyApplicationBytes() throws {
        let host = try SecureIdentity(), attacker = try SecureIdentity(), device = try SecureIdentity()
        let pin = try host.binding(), actual = try attacker.binding(), clientBinding = try device.binding()
        let namespace = String(TransportDigest.sha256(try SecureHex.decode(pin.signingKey, count: 32)).prefix(32))
        let client = try SecureNoiseSession(identity: device, namespace: namespace, host: pin, sid: "epoch", rid: "peer")
        // The responder cannot even construct a session claiming a host identity it does not own.
        #expect(actual != pin)
        #expect(throws: SecureConnectionError.authenticationFailed) {
            try SecureNoiseSession(identity: attacker, namespace: namespace, host: pin, sid: "epoch", rid: "peer", initiator: false, device: clientBinding)
        }
        #expect(!client.ready)
        #expect(throws: SecureConnectionError.authenticationFailed) { try client.seal(SecureHTTPRecord(.end, 1)) }
    }
}
