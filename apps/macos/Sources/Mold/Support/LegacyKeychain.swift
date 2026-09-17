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

    /// A UAT run does not touch the real Keychain at all.
    ///
    /// `SecretStore` swaps to a throwaway DIRECTORY under `MOLD_NATIVE_FRESH`,
    /// so its claim that a fresh run "can never read -- or delete --
    /// anybody's real keys" was true of the store and not of this: the
    /// migration reads and `SecItemDelete`s against the real service whenever
    /// the fresh prefs suite happens to hold a saved host list. Seeded hosts
    /// get fresh UUIDs that match no real item, so nothing was destroyed in
    /// practice -- but an overclaim in a comment about credentials is one
    /// `guard` away from being true.
    static func migrateIfNeeded(_ hosts: [StoredHost], into secrets: SecretStore,
                                defaults: UserDefaults, from source: Source = .keychain,
                                isFresh: Bool = NativeUAT.fresh.isSet()) {
        guard !isFresh else { return }
        guard !defaults.bool(forKey: migratedKey) else { return }
        defaults.set(migrate(hosts, into: secrets, from: source), forKey: migratedKey)
    }

    /// Moves what it can and answers whether the move is COMPLETE. One host
    /// failing -- an unreadable item, a file that would not write -- never
    /// stops the others: each is its own read, write and delete.
    ///
    /// **A value already in the file WINS.** Because one unreadable item
    /// leaves the whole move unfinished, this runs again next launch -- and in
    /// between, the person whose machine was 401ing will have opened Edit
    /// Machine and typed the current key. The file is newer by construction:
    /// nothing but a deliberate save puts a value there, while the Keychain
    /// item is whatever was last written before this build. Overwriting was
    /// the same "resurrect a key the person has since changed" hazard
    /// `PreferencesReset.kept` guards for the reset path (review E1). The
    /// stale item is still deleted -- leaving it would only make this happen
    /// again.
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
                if !move(key, of: host, into: secrets, from: source) { complete = false }
            }
        }
        return complete
    }

    /// One host's key: keep what the file already holds, otherwise write, and
    /// delete the item only once the FILE says so -- read back from disk, not
    /// from the cache that would answer with what we meant to write. The
    /// Keychain item is the only other copy there is.
    private static func move(_ key: String, of host: StoredHost, into secrets: SecretStore,
                             from source: Source) -> Bool {
        let name = SecretStore.remoteAPIKeyName(for: host.id)
        do {
            if try secrets.persistedValue(for: name)?.isEmpty != false {
                try secrets.set(key, for: name)
                guard try secrets.persistedValue(for: name) == key else { return false }
            }
            source.delete(host.id)
            return true
        } catch {
            return false
        }
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
