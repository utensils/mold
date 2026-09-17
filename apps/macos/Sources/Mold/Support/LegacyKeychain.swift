import Foundation
import MoldClient
import Security

/// Where host keys USED to live, and the one-time move out.
///
/// Legacy on purpose: nothing may read or write a key through this type any
/// more. `SecretStore` is the store (`.claude/rules/desktop.md`), and this
/// exists only so an install that already has Keychain items keeps its keys.
/// It reads each item, writes it to the file, deletes the item, and records
/// that it is done -- but ONLY when every machine was accounted for, so a
/// locked keychain is retried on the next launch instead of silently
/// abandoning the keys it could not read that once.
enum LegacyKeychain {
    static let migratedKey = "keychainKeysMigrated"
    private static let service = "io.utensils.mold.native"

    /// What one read found. `absent` and `unreadable` are deliberately
    /// different answers: the first means this machine simply never had a
    /// stored key, the second means the Keychain would not say -- and only the
    /// second is a reason to come back next launch.
    enum Item: Equatable {
        case found(String)
        case absent
        case unreadable
    }

    /// The Keychain, or a stand-in. Injected so the migration's own rules are
    /// testable without a real keychain -- and so nothing in the app can reach
    /// the live one by accident.
    struct Source {
        var read: @MainActor (UUID) -> Item
        var delete: @MainActor (UUID) -> Void

        static let keychain = Source(read: readItem, delete: deleteItem)
    }

    static func migrateIfNeeded(_ hosts: [StoredHost], into secrets: SecretStore,
                                defaults: UserDefaults, from source: Source = .keychain) {
        guard !defaults.bool(forKey: migratedKey) else { return }
        defaults.set(migrate(hosts, into: secrets, from: source), forKey: migratedKey)
    }

    /// Moves what it can and answers whether the move is COMPLETE. One host
    /// failing -- an unreadable item, a file that would not write -- never
    /// stops the others: each is its own read, write and delete.
    static func migrate(_ hosts: [StoredHost], into secrets: SecretStore,
                        from source: Source) -> Bool {
        var complete = true
        for host in hosts {
            switch source.read(host.id) {
            case .absent:
                continue
            case .unreadable:
                complete = false
            case let .found(key):
                do {
                    try secrets.set(key, for: SecretStore.remoteAPIKeyName(for: host.id))
                    source.delete(host.id)
                } catch {
                    complete = false
                }
            }
        }
        return complete
    }

    private static func readItem(_ hostID: UUID) -> Item {
        var result: CFTypeRef?
        var query = self.query(hostID)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        switch SecItemCopyMatching(query as CFDictionary, &result) {
        case errSecSuccess:
            guard let data = result as? Data, let key = String(data: data, encoding: .utf8)
            else { return .unreadable }
            return .found(key)
        case errSecItemNotFound:
            return .absent
        default:
            return .unreadable
        }
    }

    private static func deleteItem(_ hostID: UUID) {
        SecItemDelete(query(hostID) as CFDictionary)
    }

    private static func query(_ hostID: UUID) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: hostID.uuidString,
        ]
    }
}
