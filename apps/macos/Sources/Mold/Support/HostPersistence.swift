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
    /// Preferences that would not decode, kept rather than left to be
    /// overwritten by the next save -- the rule `SecretStore` follows for
    /// `secrets.json.corrupt` and the Tauri app for `settings.json.invalid`.
    static let unreadableKey = "hosts.unreadable"

    static func load(from defaults: UserDefaults = AppStorageSuite.defaults,
                     secrets: SecretStore = .shared) -> [MoldHost]? {
        guard let data = defaults.data(forKey: key),
              let stored = decode(data, in: defaults)
        else { return nil }
        LegacyKeychain.migrateIfNeeded(stored, into: secrets, defaults: defaults)
        return stored.map { $0.host(apiKey: apiKey(for: $0.id, in: secrets)) }
    }

    /// A SAVED but empty list is `[]`, never `nil`.
    ///
    /// `HostStore.seededHosts` reads `nil` as "never saved" and seeds
    /// `MOLD_NATIVE_HOSTS` over it, so answering `nil` for an emptied list
    /// resurrected on every launch exactly the machines somebody had just
    /// removed -- the resurrection its own comment says it prevents
    /// (review 05-L5).
    ///
    /// A document that does not decode is parked before anything can overwrite
    /// it, and every element that DOES read is kept: one malformed machine
    /// must not forget the others.
    private static func decode(_ data: Data, in defaults: UserDefaults) -> [StoredHost]? {
        if let stored = try? MoldJSON.localDecoder.decode([StoredHost].self, from: data) {
            return stored
        }
        defaults.set(data, forKey: unreadableKey)
        let partial = (try? MoldJSON.localDecoder.decode([Readable].self, from: data)) ?? []
        let hosts = partial.compactMap(\.host)
        return hosts.isEmpty ? nil : hosts
    }

    /// One element of the saved list, or nothing. Decoding the array through
    /// this keeps the machines that read when one of them does not.
    private struct Readable: Decodable {
        let host: StoredHost?

        init(from decoder: Decoder) throws {
            host = try? StoredHost(from: decoder)
        }
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

    /// A key this launch could not read reads as no key, because the machine
    /// list has to load either way.
    ///
    /// That is safe only because the STORE refuses to write while it cannot
    /// read (`SecretStore.loaded`, review E2): the editor then shows a blank
    /// field, but Save and Remove both throw rather than turning a blank into
    /// a clear, and `HostStore` reports the refusal on that machine. Without
    /// that refusal this line would be H5 arriving by a different door.
    private static func apiKey(for host: MoldHost.ID, in secrets: SecretStore) -> String? {
        try? secrets.value(for: SecretStore.remoteAPIKeyName(for: host))
    }
}
