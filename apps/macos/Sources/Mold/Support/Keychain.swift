import Foundation
import Security

/// API keys, kept out of preferences.
///
/// A host's URL is ordinary configuration and lives in UserDefaults; its key
/// is a credential and does not. Writing one into a plist would put it in
/// backups, in Time Machine, and in any sync that copies the domain.
enum Keychain {
    private static let service = "io.utensils.mold.native"

    static func apiKey(for hostID: UUID) -> String? {
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query(hostID, returningData: true) as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func setAPIKey(_ key: String?, for hostID: UUID) {
        SecItemDelete(query(hostID) as CFDictionary)
        guard let key, !key.isEmpty, let data = key.data(using: .utf8) else { return }
        var attributes = query(hostID)
        attributes[kSecValueData as String] = data
        // The app reaches hosts on a timer, so it must be able to read this
        // while the screen is locked -- but it is never worth syncing.
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(attributes as CFDictionary, nil)
    }

    private static func query(_ hostID: UUID, returningData: Bool = false) -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: hostID.uuidString,
        ]
        if returningData {
            query[kSecReturnData as String] = true
            query[kSecMatchLimit as String] = kSecMatchLimitOne
        }
        return query
    }
}
