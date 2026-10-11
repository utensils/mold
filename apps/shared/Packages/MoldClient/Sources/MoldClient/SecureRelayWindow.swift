import Foundation
import CryptoKit

/// Ciphertext ordering is checked before Noise sees bytes. Sixty-four frames and the last
/// 32 duplicate digests are the entire reordering state; gaps are never skipped.
struct SecureRelayWindow: Sendable {
    static let capacity = 64
    private(set) var next: UInt64 = 0
    private var pending: [UInt64: Data] = [:]
    private var seen: [UInt64: Data] = [:]
    var credit: Int { Self.capacity - pending.count }
    var hasGap: Bool { !pending.isEmpty && pending[next] == nil }

    mutating func insert(sequence: UInt64, data: Data) throws -> Bool {
        guard !data.isEmpty, data.count <= 16_384 else { throw SecureConnectionError.protocolViolation }
        if sequence < next {
            // A very late at-least-once delivery may outlive our bounded digest
            // history. It cannot reach Noise again; acknowledge and discard it.
            if let digest = seen[sequence], digest != Data(SHA256.hash(data: data)) {
                throw SecureConnectionError.protocolViolation
            }
            return true
        }
        guard sequence < next + UInt64(Self.capacity) else { throw SecureConnectionError.protocolViolation }
        if let old = pending[sequence] {
            guard old == data else { throw SecureConnectionError.protocolViolation }
            return true
        }
        pending[sequence] = data
        return false
    }
    mutating func drain(capacity: Int) -> Data {
        var bytes = Data()
        while let value = pending[next], bytes.count + value.count <= capacity {
            pending.removeValue(forKey: next)
            seen[next] = Data(SHA256.hash(data: value))
            if next >= UInt64(Self.capacity * 2) { seen.removeValue(forKey: next - UInt64(Self.capacity * 2)) }
            next += 1
            bytes.append(value)
        }
        return bytes
    }
}
