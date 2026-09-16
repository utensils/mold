import Foundation
import MoldClient

/// The machine list, across launches.
///
/// Only the name and URL are stored here. The API key travels separately
/// through the Keychain and is rejoined when the list is read, so a
/// preferences file that leaks carries no credential.
enum HostPersistence {
    private static let key = "hosts"

    static func load(from defaults: UserDefaults = AppStorageSuite.defaults) -> [MoldHost]? {
        guard let data = defaults.data(forKey: key),
              let stored = try? MoldJSON.localDecoder.decode([StoredHost].self, from: data),
              !stored.isEmpty
        else { return nil }
        return stored.map { $0.host(apiKey: Keychain.apiKey(for: $0.id)) }
    }

    static func save(_ hosts: [MoldHost], to defaults: UserDefaults = AppStorageSuite.defaults) {
        let stored = hosts.map(StoredHost.init)
        defaults.set(try? MoldJSON.localEncoder.encode(stored), forKey: key)
        for host in hosts {
            Keychain.setAPIKey(host.apiKey, for: host.id)
        }
    }

    static func forget(_ host: MoldHost) {
        Keychain.setAPIKey(nil, for: host.id)
    }
}
