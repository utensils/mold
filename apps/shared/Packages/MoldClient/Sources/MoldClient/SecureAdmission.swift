import Foundation
import CryptoKit

enum SecureAdmission {
    static func namespace(_ binding: SecureBinding) throws -> String {
        SecureHex.encode(Data(SHA256.hash(data: try SecureHex.decode(binding.signingKey, count: 32))).prefix(16))
    }
    static func validate(host: SecureHost, device: SecureBinding, now: UInt64 = UInt64(Date().timeIntervalSince1970)) throws {
        try SecureCrypto.verify(host.binding)
        try validateCloudURL(host.websocketURL)
        guard try namespace(host.binding) == host.namespace, host.grant.expiresAt > now else { throw SecureConnectionError.authenticationFailed }
        var bytes = Data("MOLD3-GRANT\0".utf8)
        bytes.append(try SecureHex.decode(host.namespace, count: 16))
        bytes.append(try SecureHex.decode(device.signingKey, count: 32))
        bytes.append(try SecureHex.decode(device.noiseKey, count: 32))
        bytes.append(try SecureHex.decode(host.grant.id, count: 16))
        bytes.append(SecureHex.u64(host.grant.expiresAt)); bytes.append(SecureHex.u64(host.grant.epoch))
        try SecureCrypto.verify(key: host.binding.signingKey, message: bytes, signature: host.grant.signature)
    }
    static func validateCloudURL(_ url: URL?) throws {
        guard let url else { return }
        guard url.scheme == "wss", url.user == nil, url.password == nil,
              url.fragment == nil, url.query == nil, url.host?.isEmpty == false,
              url.absoluteString.utf8.count <= 2048,
              url.port.map({ (1...65535).contains($0) }) ?? true else { throw SecureConnectionError.authenticationFailed }
    }
    static func join(namespace: String, role: UInt8, connectionID: String, nonce: String, device: SecureBinding, expiresAt: UInt64) throws -> Data {
        guard (1...3).contains(role), !connectionID.isEmpty, connectionID.utf8.count <= 128,
              connectionID.utf8.allSatisfy({ (33...126).contains($0) }), expiresAt <= 9_007_199_254_740_991
        else { throw SecureConnectionError.protocolViolation }
        var bytes = Data("MOLD3-JOIN\0".utf8); bytes.append(3)
        bytes.append(try SecureHex.decode(namespace, count: 16)); bytes.append(role)
        bytes.append(UInt8(connectionID.utf8.count >> 8)); bytes.append(UInt8(connectionID.utf8.count & 255))
        bytes.append(Data(connectionID.utf8)); bytes.append(try SecureHex.decode(nonce, count: 32))
        bytes.append(try SecureHex.decode(device.signingKey, count: 32)); bytes.append(SecureHex.u64(expiresAt))
        return bytes
    }
    static func object(_ value: some Encodable) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any]
        else { throw SecureConnectionError.protocolViolation }
        return object
    }
}
