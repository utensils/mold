import Foundation
import CMoldSecure

/// Host bootstrap only. Rust owns atomic creation, permission validation and
/// concurrent-writer convergence, shared with Tauri and headless hosts.
enum SharedOperatorKey {
    static func explicit(_ raw: String) throws -> String {
        let input = Data(raw.utf8)
        var output = [UInt8](repeating: 0, count: 4096)
        defer { _ = output.withUnsafeMutableBytes { $0.initializeMemory(as: UInt8.self, repeating: 0) } }
        let count = input.withUnsafeBytes { input in
            output.withUnsafeMutableBufferPointer { output in
                mold_secure_operator_override(input.bindMemory(to: UInt8.self).baseAddress, input.count,
                                              output.baseAddress, output.count)
            }
        }
        guard count > 0, count <= output.count,
              let key = String(bytes: output.prefix(Int(count)), encoding: .utf8) else {
            throw SecretStoreError.unreadable(path: "MOLD_API_KEY",
                                              reason: "The explicit owner key configuration is unavailable or invalid.")
        }
        return key
    }

    static func resolve(home: URL, legacyCandidate: String?) throws -> String {
        let path = Data(home.path(percentEncoded: false).utf8)
        let candidate = Data((legacyCandidate ?? "").utf8)
        var output = [UInt8](repeating: 0, count: 4096)
        defer { _ = output.withUnsafeMutableBytes { $0.initializeMemory(as: UInt8.self, repeating: 0) } }
        let count = path.withUnsafeBytes { path in candidate.withUnsafeBytes { candidate in
            output.withUnsafeMutableBufferPointer { output in
                mold_secure_operator_key(path.bindMemory(to: UInt8.self).baseAddress, path.count,
                                         candidate.bindMemory(to: UInt8.self).baseAddress, candidate.count,
                                         output.baseAddress, output.count)
            }
        } }
        guard count > 0, count <= output.count,
              let key = String(bytes: output.prefix(Int(count)), encoding: .utf8) else {
            throw SecretStoreError.unreadable(path: home.appending(path: "secure/operator.key").path,
                                              reason: "The shared host owner key is unavailable or is not private.")
        }
        return key
    }
}
