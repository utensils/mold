#if os(iOS)
import Foundation
import Security

/// The iPhone's `CredentialStore`: one generic-password item per machine, in
/// the app's own Keychain access group (`.claude/rules/mobile.md`: API keys are
/// Keychain-only on the phone). Never on the Mac -- the Mac keeps keys in
/// `SecretStore`'s file and must not answer that question two ways.
///
/// Every `OSStatus` other than "no such item" is thrown, never read as "no
/// key": a locked Keychain returning nil is how a later save once deleted
/// every key (review 05-H5). Items are `AfterFirstUnlockThisDeviceOnly` so a
/// background refresh after the first unlock can still reach its machines,
/// and a key never leaves this device in a backup.
public struct KeychainCredentialStore: CredentialStore {
    public static let defaultService = "io.utensils.mold.companion.remote-api-key"

    public let service: String

    public init(service: String = Self.defaultService) {
        self.service = service
    }

    public func apiKey(for host: UUID) throws -> String? {
        var query = base(host)
        query[kSecReturnData] = true
        query[kSecMatchLimit] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        switch status {
        case errSecSuccess:
            guard let data = result as? Data, let key = String(data: data, encoding: .utf8) else {
                throw KeychainError.unreadable
            }
            return key
        case errSecItemNotFound:
            return nil
        default:
            throw KeychainError.status(status)
        }
    }

    public func setAPIKey(_ key: String, for host: UUID) throws {
        guard !key.isEmpty else { return try clearAPIKey(for: host) }
        let data = Data(key.utf8)
        let update = [kSecValueData: data, kSecAttrAccessible: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly]
            as [CFString: Any]
        let status = SecItemUpdate(base(host) as CFDictionary, update as CFDictionary)
        switch status {
        case errSecSuccess:
            return
        case errSecItemNotFound:
            var add = base(host)
            add.merge(update) { _, new in new }
            let added = SecItemAdd(add as CFDictionary, nil)
            guard added == errSecSuccess else { throw KeychainError.status(added) }
        default:
            throw KeychainError.status(status)
        }
    }

    public func clearAPIKey(for host: UUID) throws {
        let status = SecItemDelete(base(host) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.status(status)
        }
    }

    private func base(_ host: UUID) -> [CFString: Any] {
        [kSecClass: kSecClassGenericPassword,
         kSecAttrService: service,
         kSecAttrAccount: host.uuidString]
    }
}

public enum KeychainError: Error, Equatable {
    /// The Keychain refused: locked, missing entitlement, or worse. Never "no key".
    case status(OSStatus)
    /// An item existed but was not a UTF-8 key.
    case unreadable
}
#endif
