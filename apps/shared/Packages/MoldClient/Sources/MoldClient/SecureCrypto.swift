import Foundation
import CMoldSecure

/// Ownership only: all key generation, signing, binding checks and Noise operations
/// call the same audited Rust implementation used by the host and other clients.
final class SecureIdentity: Sendable {
    let handle: UInt64
    init(secret: Data? = nil) throws {
        if let secret {
            guard secret.count == 64 else { throw SecureConnectionError.authenticationFailed }
            handle = secret.withUnsafeBytes { mold_secure_identity_restore($0.bindMemory(to: UInt8.self).baseAddress, secret.count) }
        } else { handle = mold_secure_identity_new() }
        guard handle != 0 else { throw SecureConnectionError.authenticationFailed }
    }
    deinit { mold_secure_release(handle) }
    func secret() throws -> Data { try SecureCrypto.output(capacity: 64) { mold_secure_identity_export(handle, $0, $1) } }
    func binding() throws -> SecureBinding {
        let bytes = try SecureCrypto.output(capacity: 128) { mold_secure_identity_binding(handle, $0, $1) }
        guard bytes.count == 128 else { throw SecureConnectionError.authenticationFailed }
        return SecureBinding(signingKey: SecureHex.encode(bytes.prefix(32)), noiseKey: SecureHex.encode(bytes.dropFirst(32).prefix(32)),
                             signature: SecureHex.encode(bytes.suffix(64)))
    }
    func sign(_ data: Data) throws -> Data {
        try data.withUnsafeBytes { input in
            try SecureCrypto.output(capacity: 64) { mold_secure_identity_sign(handle, input.bindMemory(to: UInt8.self).baseAddress, data.count, $0, $1) }
        }
    }
}

enum SecureCrypto {
    static func output(capacity: Int = 65_535, _ operation: (UnsafeMutablePointer<UInt8>, Int) -> Int32) throws -> Data {
        var bytes = [UInt8](repeating: 0, count: capacity)
        let count = bytes.withUnsafeMutableBufferPointer { operation($0.baseAddress!, capacity) }
        guard count >= 0, count <= capacity else { throw SecureConnectionError.authenticationFailed }
        return Data(bytes.prefix(Int(count)))
    }
    static func verify(key: String, message: Data, signature: String) throws {
        let key = try SecureHex.decode(key, count: 32), signature = try SecureHex.decode(signature, count: 64)
        let result = key.withUnsafeBytes { key in message.withUnsafeBytes { messageBytes in signature.withUnsafeBytes { signature in
            mold_secure_verify_signature(key.bindMemory(to: UInt8.self).baseAddress,
                messageBytes.bindMemory(to: UInt8.self).baseAddress, message.count,
                signature.bindMemory(to: UInt8.self).baseAddress)
        } } }
        guard result == 1 else { throw SecureConnectionError.authenticationFailed }
    }
    static func verify(_ binding: SecureBinding) throws {
        var message = Data("MOLD3-BIND\0".utf8)
        message.append(try SecureHex.decode(binding.signingKey, count: 32))
        message.append(try SecureHex.decode(binding.noiseKey, count: 32))
        try verify(key: binding.signingKey, message: message, signature: binding.signature)
    }
}

/// Only a connection actor may order calls on one session. Rust also serializes the
/// registry, so cancellation/release can never race a raw pointer or free active memory.
final class SecureNoiseSession: Sendable {
    let handle: UInt64
    init(identity: SecureIdentity, namespace: String, host: SecureBinding, sid: String, rid: String,
         initiator: Bool = true, device: SecureBinding? = nil) throws {
        let device = try device ?? identity.binding()
        let data = try JSONSerialization.data(withJSONObject: ["initiator": initiator, "namespace": namespace,
            "host_key": host.signingKey, "host_noise_key": host.noiseKey, "device_key": device.signingKey,
            "device_noise_key": device.noiseKey, "sid": sid, "rid": rid])
        handle = data.withUnsafeBytes { mold_secure_session_new(identity.handle, $0.bindMemory(to: UInt8.self).baseAddress, data.count) }
        guard handle != 0 else { throw SecureConnectionError.authenticationFailed }
    }
    deinit { mold_secure_release(handle) }
    var ready: Bool { mold_secure_session_ready(handle) == 1 }
    func handshakeHash() throws -> Data { try SecureCrypto.output(capacity: 32) { mold_secure_session_handshake_hash(handle, $0, $1) } }
    func writeHandshake() throws -> Data { try SecureCrypto.output { mold_secure_session_handshake_write(handle, $0, $1) } }
    func readHandshake(_ data: Data) throws {
        let code = data.withUnsafeBytes { mold_secure_session_handshake_read(handle, $0.bindMemory(to: UInt8.self).baseAddress, data.count) }
        guard code == 0 else { throw SecureConnectionError.authenticationFailed }
    }
    func seal(_ record: SecureHTTPRecord) throws -> Data {
        let bytes = try record.encoded()
        return try bytes.withUnsafeBytes { input in
            try SecureCrypto.output { mold_secure_session_seal(handle, input.bindMemory(to: UInt8.self).baseAddress, bytes.count, $0, $1) }
        }
    }
    func open(_ bytes: Data) throws -> SecureHTTPRecord {
        let plain = try bytes.withUnsafeBytes { input in
            try SecureCrypto.output { mold_secure_session_open(handle, input.bindMemory(to: UInt8.self).baseAddress, bytes.count, $0, $1) }
        }
        return try SecureHTTPRecord(encoded: plain)
    }
}
