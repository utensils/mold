import Foundation
import MoldClient
import Testing

@testable import Mold

/// 05-L5: what a saved machine list means when it is empty, and when part of
/// it will not decode.
@MainActor
struct HostPersistenceTests {
    private func scratch() throws -> (UserDefaults, SecretStore) {
        let name = "io.utensils.mold.native.tests.hosts.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return (defaults, SecretStore(directory: directory))
    }

    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name):7680")!)
    }

    /// **Fails today**: `load` answered `nil` for a decoded-empty array, and
    /// `HostStore.seededHosts` reads `nil` as "never saved" and seeds
    /// `MOLD_NATIVE_HOSTS` over it -- so a dev run resurrected on every launch
    /// exactly the machines that had just been removed.
    @Test func anEmptiedMachineListStaysEmpty() throws {
        let (defaults, secrets) = try scratch()
        HostPersistence.save([host("workstation")], to: defaults)
        HostPersistence.save([], to: defaults)

        #expect(HostPersistence.load(from: defaults, secrets: secrets) == [])
    }

    /// Never saved at all is still `nil` -- that is what makes seeding a
    /// first run possible.
    @Test func aListNobodyEverSavedIsStillAbsent() throws {
        let (defaults, secrets) = try scratch()
        #expect(HostPersistence.load(from: defaults, secrets: secrets) == nil)
    }

    /// One malformed machine must not forget the others, and the bytes that
    /// would not read are parked rather than left for the next save to
    /// overwrite.
    @Test func oneUnreadableMachineNeverForgetsTheRest() throws {
        let (defaults, secrets) = try scratch()
        let good = StoredHost(host("workstation"))
        let json = try MoldJSON.localEncoder.encode([good])
        var array = try #require(
            try JSONSerialization.jsonObject(with: json) as? [[String: Any]])
        array.append(["name": "broken"]) // no id, no base_url
        let data = try JSONSerialization.data(withJSONObject: array)
        defaults.set(data, forKey: "hosts")

        let loaded = try #require(HostPersistence.load(from: defaults, secrets: secrets))
        #expect(loaded.map(\.name) == ["workstation"])
        #expect(defaults.data(forKey: HostPersistence.unreadableKey) == data)
    }
}
