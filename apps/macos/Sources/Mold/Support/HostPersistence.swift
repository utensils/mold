import Foundation
import MoldClient

/// The machine list, across launches.
///
/// Only the name and URL are stored in preferences. The API key travels
/// separately through `SecretStore`'s owner-only file and is rejoined when the
/// list is read, so a preferences file that leaks carries no credential.
///
/// **Writing a key is always explicit.** `save(_:)` persists the LIST and
/// nothing else. It used to loop `setAPIKey(host.apiKey, …)` over every host,
/// which mirrored an absent in-memory key into the store as a delete: one
/// failed read at launch -- a locked keychain, a re-signed dev build -- and the
/// next add, edit or removal of any machine destroyed every stored key
/// (review 05-H5).
enum HostPersistence {
    private static let key = "hosts"

    static func load(from defaults: UserDefaults = AppStorageSuite.defaults,
                     secrets: SecretStore = .shared) -> [MoldHost]? {
        guard let data = defaults.data(forKey: key) else { return nil }
        guard let stored = try? MoldJSON.localDecoder.decode([StoredHost].self, from: data),
              !stored.isEmpty
        else { return nil }
        LegacyKeychain.migrateIfNeeded(stored, into: secrets, defaults: defaults)
        return stored.map { $0.host(apiKey: apiKey(for: $0.id, in: secrets)) }
    }

    static func save(_ hosts: [MoldHost], to defaults: UserDefaults = AppStorageSuite.defaults) {
        defaults.set(try? MoldJSON.localEncoder.encode(hosts.map(StoredHost.init)), forKey: key)
    }

    /// One machine's key, because somebody typed one -- or emptied the field,
    /// which the editor shows pre-filled and is therefore an explicit clear.
    /// Called from `add` and `update`, never from a list write.
    static func setAPIKey(_ apiKey: String?, for host: MoldHost.ID,
                          in secrets: SecretStore = .shared) throws {
        let name = SecretStore.remoteAPIKeyName(for: host)
        if let apiKey, !apiKey.isEmpty {
            try secrets.set(apiKey, for: name)
        } else {
            try secrets.clear(name)
        }
    }

    /// The machine is gone, so its key goes with it.
    static func forget(_ host: MoldHost, in secrets: SecretStore = .shared) throws {
        try secrets.clear(SecretStore.remoteAPIKeyName(for: host.id))
    }

    /// `value(for:)` throws only on a name the store does not keep, and a
    /// host UUID never is one -- an unreadable FILE is already an empty store
    /// by design, exactly as the Rust reference has it.
    private static func apiKey(for host: MoldHost.ID, in secrets: SecretStore) -> String? {
        try? secrets.value(for: SecretStore.remoteAPIKeyName(for: host))
    }
}
