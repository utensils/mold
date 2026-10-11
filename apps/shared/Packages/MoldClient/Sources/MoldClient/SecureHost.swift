import Foundation

/// Public pins and signed admission metadata. Device private keys never belong in a host list.
public struct SecureHost: Codable, Hashable, Sendable {
    public let binding: SecureBinding
    public let namespace: String
    public let websocketURL: URL
    public let grant: SecureDeviceGrant
    public init(binding: SecureBinding, namespace: String, websocketURL: URL, grant: SecureDeviceGrant) {
        self.binding = binding; self.namespace = namespace; self.websocketURL = websocketURL; self.grant = grant
    }
}

public struct SecureBinding: Codable, Hashable, Sendable {
    public let signingKey: String
    public let noiseKey: String
    public let signature: String
    public init(signingKey: String, noiseKey: String, signature: String) {
        self.signingKey = signingKey; self.noiseKey = noiseKey; self.signature = signature
    }
}

public struct SecureDeviceGrant: Codable, Hashable, Sendable {
    public let id: String
    public let expiresAt: UInt64
    public let epoch: UInt64
    public let signature: String
    public init(id: String, expiresAt: UInt64, epoch: UInt64, signature: String) {
        self.id = id; self.expiresAt = expiresAt; self.epoch = epoch; self.signature = signature
    }
}

public struct SecureInvite: Codable, Hashable, Sendable {
    public let id: String
    public let signingKey: String
    public let expiresAt: UInt64
    public let epoch: UInt64
    public let signature: String
    public init(id: String, signingKey: String, expiresAt: UInt64, epoch: UInt64, signature: String) {
        self.id = id; self.signingKey = signingKey; self.expiresAt = expiresAt; self.epoch = epoch; self.signature = signature
    }
}

enum SecureHex {
    static func decode(_ text: String, count: Int) throws -> Data {
        let bytes = Array(text.utf8)
        guard bytes.count == count * 2, bytes.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
        else { throw SecureConnectionError.authenticationFailed }
        func nibble(_ byte: UInt8) -> UInt8 { byte < 58 ? byte - 48 : byte - 87 }
        return Data(stride(from: 0, to: bytes.count, by: 2).map { nibble(bytes[$0]) << 4 | nibble(bytes[$0 + 1]) })
    }
    static func encode(_ bytes: Data) -> String { bytes.map { String(format: "%02x", $0) }.joined() }
    static func u64(_ value: UInt64) -> Data {
        Data((0..<8).reversed().map { UInt8(truncatingIfNeeded: value >> ($0 * 8)) })
    }
}
