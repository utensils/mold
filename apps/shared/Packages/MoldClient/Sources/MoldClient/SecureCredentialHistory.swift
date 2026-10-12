import Foundation
import CMoldSecure

/// Remembers only which credentials must never cross a Regular connection.
/// This history grants no access and survives removal or replacement of a host.
public final class SecureCredentialHistory: @unchecked Sendable {
    public static let shared = SecureCredentialHistory(defaults: .standard)
    public static let storageKey = "secureCredentialDigests.v1"
    private let defaults: UserDefaults
    private let lock = NSLock()

    public init(defaults: UserDefaults) { self.defaults = defaults }

    public func remember(_ hosts: [MoldHost]) {
        lock.lock(); defer { lock.unlock() }
        let previous = Set(defaults.stringArray(forKey: Self.storageKey) ?? [])
        var digests = previous
        for host in hosts where host.secure != nil {
            if let key = host.apiKey, let digest = Self.digest(key) { digests.insert(digest) }
        }
        if digests != previous { defaults.set(digests.sorted(), forKey: Self.storageKey) }
    }

    func refuseKnownSecureCredentials(host: MoldHost, request: URLRequest, session: URLSession) throws {
        var credentials = [host.apiKey].compactMap { $0 }
        for headers in [request.allHTTPHeaderFields ?? [:],
                        (session.configuration.httpAdditionalHeaders ?? [:]).reduce(into: [String: String]()) { result, entry in
                            if let name = entry.key as? String, let value = entry.value as? String { result[name] = value }
                        }] {
            for (name, value) in headers {
                if name.caseInsensitiveCompare("X-API-Key") == .orderedSame { credentials.append(value) }
                if name.caseInsensitiveCompare("Authorization") == .orderedSame {
                    let parts = value.split(maxSplits: 1, whereSeparator: { $0.isWhitespace })
                    if parts.count == 2, parts[0].lowercased() == "bearer" { credentials.append(String(parts[1])) }
                }
            }
        }
        lock.lock(); defer { lock.unlock() }
        let known = Set(defaults.stringArray(forKey: Self.storageKey) ?? [])
        for credential in credentials where !credential.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            guard let digest = Self.digest(credential), !known.contains(digest) else {
                throw SecureConnectionError.pairingRequired
            }
        }
    }

    private static func digest(_ credential: String) -> String? {
        let bytes = Data(credential.trimmingCharacters(in: .whitespacesAndNewlines).utf8)
        guard !bytes.isEmpty, bytes.count <= 4096 else { return nil }
        var output = [UInt8](repeating: 0, count: 32)
        let count = bytes.withUnsafeBytes { bytes in
            output.withUnsafeMutableBufferPointer { output in
                mold_secure_credential_digest(bytes.bindMemory(to: UInt8.self).baseAddress, bytes.count,
                                              output.baseAddress, output.count)
            }
        }
        guard count == 32 else { return nil }
        return output.map { String(format: "%02x", $0) }.joined()
    }
}
