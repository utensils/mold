import Foundation
import MoldClient

/// The machine list, across launches.
///
/// Only the name and URL are stored here. The API key travels separately
/// through the Keychain and is rejoined when the list is read, so a
/// preferences file that leaks carries no credential.
enum HostPersistence {
    private static let key = "hosts"

    struct Stored: Codable {
        let id: UUID
        var name: String
        var baseURL: URL
    }

    static func load(from defaults: UserDefaults = .standard) -> [MoldHost]? {
        guard let data = defaults.data(forKey: key),
              let stored = try? MoldJSON.decoder.decode([Stored].self, from: data),
              !stored.isEmpty
        else { return nil }
        return stored.map {
            MoldHost(id: $0.id, name: $0.name, baseURL: $0.baseURL,
                     apiKey: Keychain.apiKey(for: $0.id))
        }
    }

    static func save(_ hosts: [MoldHost], to defaults: UserDefaults = .standard) {
        let stored = hosts.map { Stored(id: $0.id, name: $0.name, baseURL: $0.baseURL) }
        defaults.set(try? MoldJSON.encoder.encode(stored), forKey: key)
        for host in hosts {
            Keychain.setAPIKey(host.apiKey, for: host.id)
        }
    }

    static func forget(_ host: MoldHost) {
        Keychain.setAPIKey(nil, for: host.id)
    }
}
