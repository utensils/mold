import Foundation
import MoldClient
import Testing

@testable import Mold

/// 05-H5: writing the machine list must never imply anything about a key, and
/// the one-time move out of the Keychain must not be able to lose one.
@MainActor
struct HostSecretsTests {
    private func scratch() throws -> (UserDefaults, SecretStore) {
        let name = "io.utensils.mold.native.tests.secrets.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appending(path: name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return (defaults, SecretStore(directory: directory))
    }

    private func host(_ name: String, apiKey: String? = nil) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name):7680")!, apiKey: apiKey)
    }

    // MARK: - 05-H5

    /// **Fails today**: `save(_:)` looped `Keychain.setAPIKey(host.apiKey, …)`
    /// over every host, and `setAPIKey(nil, …)` deletes. One failed read at
    /// launch left every host with `apiKey == nil`, and the next persist --
    /// adding, editing or removing ANY machine -- destroyed every stored key.
    @Test func savingTheMachineListNeverTouchesAStoredKey() throws {
        let (defaults, secrets) = try scratch()
        let workstation = host("workstation")
        try HostPersistence.setAPIKey("k-workstation", for: workstation.id, in: secrets)

        // The in-memory copy has no key -- exactly the state a failed read
        // leaves behind -- and is saved alongside a machine being added.
        HostPersistence.save([workstation, host("hal9000")], to: defaults)

        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: workstation.id)) == "k-workstation")
    }

    @Test func aSavedMachineComesBackWithItsKey() throws {
        let (defaults, secrets) = try scratch()
        let workstation = host("workstation")
        HostPersistence.save([workstation], to: defaults)
        try HostPersistence.setAPIKey("k-workstation", for: workstation.id, in: secrets)

        let loaded = try #require(HostPersistence.load(from: defaults, secrets: secrets))
        #expect(loaded.map(\.apiKey) == ["k-workstation"])
    }

    /// Clearing is its own call, and removing a machine takes its key with it.
    @Test func aKeyIsClearedOnlyWhenSomebodyAsks() throws {
        let (_, secrets) = try scratch()
        let workstation = host("workstation")
        let name = SecretStore.remoteAPIKeyName(for: workstation.id)
        try HostPersistence.setAPIKey("k-workstation", for: workstation.id, in: secrets)

        try HostPersistence.setAPIKey("", for: workstation.id, in: secrets)
        #expect(try secrets.value(for: name) == nil)

        try HostPersistence.setAPIKey("k-again", for: workstation.id, in: secrets)
        try HostPersistence.forget(workstation, in: secrets)
        #expect(try secrets.value(for: name) == nil)
    }

    // MARK: - The one-time move out of the Keychain

    /// A global-actor-isolated closure is implicitly `@Sendable`, so what the
    /// stand-in deleted is recorded in a reference somebody holds rather than
    /// in a captured local.
    @MainActor final class Deletions {
        var ids: Set<UUID> = []
    }

    private func source(_ items: [UUID: LegacyKeychain.Item],
                        into deletions: Deletions = Deletions()) -> LegacyKeychain.Source {
        LegacyKeychain.Source(read: { items[$0] ?? .absent },
                              delete: { deletions.ids.insert($0) })
    }

    /// Every migration test below is an ORDINARY launch. The test scheme sets
    /// `MOLD_NATIVE_FRESH`, under which the real migration deliberately does
    /// nothing at all -- which `aFreshRunNeverTouchesTheRealKeychain` is the
    /// test for.
    private func migrate(_ hosts: [StoredHost], into secrets: SecretStore,
                         defaults: UserDefaults, from source: LegacyKeychain.Source) {
        LegacyKeychain.migrateIfNeeded(hosts, into: secrets, defaults: defaults,
                                       from: source, isFresh: false)
    }

    @Test func everyKeychainKeyMovesToTheFileAndTheItemGoes() throws {
        let (defaults, secrets) = try scratch()
        let workstation = StoredHost(host("workstation"))
        let hal = StoredHost(host("hal9000"))
        let deleted = Deletions()
        let source = source([workstation.id: .found("k-workstation"), hal.id: .found("k-hal")], into: deleted)

        migrate([workstation, hal], into: secrets, defaults: defaults, from: source)

        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: workstation.id)) == "k-workstation")
        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: hal.id)) == "k-hal")
        #expect(deleted.ids == [workstation.id, hal.id])
        #expect(defaults.bool(forKey: LegacyKeychain.migratedKey))
    }

    /// Idempotent: a second launch reads no Keychain item at all, so a key
    /// the person later cleared can never come back.
    @Test func theMoveHappensOnce() throws {
        let (defaults, secrets) = try scratch()
        let workstation = StoredHost(host("workstation"))
        migrate([workstation], into: secrets, defaults: defaults,
                                       from: source([workstation.id: .found("k-workstation")]))
        try secrets.clear(SecretStore.remoteAPIKeyName(for: workstation.id))

        migrate([workstation], into: secrets, defaults: defaults,
                                       from: source([workstation.id: .found("k-workstation")]))
        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: workstation.id)) == nil)
    }

    /// **Fails today**: `.found` wrote the Keychain value unconditionally, and
    /// one `.unreadable` host leaves the done-flag off so the WHOLE migration
    /// re-runs next launch. So: launch 1 the keychain is locked and `workstation`
    /// loads keyless; the user opens Edit Machine and types the current key;
    /// launch 2 the item reads and the OLD, rotated value silently replaces
    /// what they just typed, and the item is deleted. The file is newer by
    /// construction -- nothing but a deliberate save puts a value there -- so
    /// a value already in it wins, and the stale item still goes.
    @Test func aRetriedMigrationKeepsTheKeyTheUserJustTyped() throws {
        let (defaults, secrets) = try scratch()
        let workstation = StoredHost(host("workstation"))
        let name = SecretStore.remoteAPIKeyName(for: workstation.id)

        // Launch 1: the item will not read, so the move stays unfinished.
        migrate([workstation], into: secrets, defaults: defaults,
                                       from: source([workstation.id: .unreadable]))
        #expect(!defaults.bool(forKey: LegacyKeychain.migratedKey))

        // The user retypes the current key while the old one sits in the item.
        try HostPersistence.setAPIKey("k-typed", for: workstation.id, in: secrets)

        // Launch 2: the item reads at last -- and holds the rotated key.
        let deleted = Deletions()
        migrate([workstation], into: secrets, defaults: defaults,
                                       from: source([workstation.id: .found("k-rotated")], into: deleted))

        #expect(try secrets.value(for: name) == "k-typed")
        #expect(deleted.ids == [workstation.id], "the stale item still goes")
        #expect(defaults.bool(forKey: LegacyKeychain.migratedKey))
    }

    /// **Fails today**: `SecretStore` swaps to a throwaway DIRECTORY under
    /// `MOLD_NATIVE_FRESH`, so its comment claims a fresh run "can never read
    /// -- or delete -- anybody's real keys". The migration had no such gate:
    /// it read and `SecItemDelete`d against the REAL service whenever the
    /// fresh prefs suite happened to hold a saved host list. Seeded hosts get
    /// fresh UUIDs that match no real item, so nothing was destroyed in
    /// practice -- but a comment about credentials should not be an overclaim.
    @Test func aFreshRunNeverTouchesTheRealKeychain() throws {
        let (defaults, secrets) = try scratch()
        let workstation = StoredHost(host("workstation"))
        let deleted = Deletions()

        LegacyKeychain.migrateIfNeeded(
            [workstation], into: secrets, defaults: defaults,
            from: source([workstation.id: .found("k-real")], into: deleted), isFresh: true)

        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: workstation.id)) == nil)
        #expect(deleted.ids.isEmpty)
        #expect(!defaults.bool(forKey: LegacyKeychain.migratedKey),
                "and it is not recorded as done, so a real launch still moves them")
    }

    /// The Keychain item is the only other copy, so it is deleted only once
    /// the FILE says it holds the key -- read back from disk, not from the
    /// cache that would answer with what we meant to write.
    @Test func anItemGoesOnlyAfterTheFileProvesItHasTheKey() throws {
        let (_, secrets) = try scratch()
        let workstation = StoredHost(host("workstation"))
        let name = SecretStore.remoteAPIKeyName(for: workstation.id)

        try secrets.set("k-workstation", for: name)
        #expect(try secrets.persistedValue(for: name) == "k-workstation")

        // A write another instance made behind this one's warm cache is what
        // `persistedValue` has to see.
        try SecretStore(directory: secrets.directory).set("k-elsewhere", for: name)
        #expect(try secrets.persistedValue(for: name) == "k-elsewhere")
    }

    /// One item the Keychain will not give up must not cost the others theirs,
    /// and must not be written off: the move stays unfinished so the next
    /// launch tries again.
    @Test func oneUnreadableItemLosesNeitherTheOthersNorItself() throws {
        let (defaults, secrets) = try scratch()
        let locked = StoredHost(host("workstation"))
        let readable = StoredHost(host("hal9000"))

        migrate([locked, readable], into: secrets, defaults: defaults,
                                       from: source([locked.id: .unreadable,
                                                     readable.id: .found("k-hal")]))

        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: readable.id)) == "k-hal")
        #expect(!defaults.bool(forKey: LegacyKeychain.migratedKey))

        migrate([locked, readable], into: secrets, defaults: defaults,
                                       from: source([locked.id: .found("k-workstation"),
                                                     readable.id: .absent]))
        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: locked.id)) == "k-workstation")
        #expect(defaults.bool(forKey: LegacyKeychain.migratedKey))
    }
}
