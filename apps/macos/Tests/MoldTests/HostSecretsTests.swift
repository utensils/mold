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
        let plato = host("plato")
        try HostPersistence.setAPIKey("k-plato", for: plato.id, in: secrets)

        // The in-memory copy has no key -- exactly the state a failed read
        // leaves behind -- and is saved alongside a machine being added.
        HostPersistence.save([plato, host("hal9000")], to: defaults)

        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: plato.id)) == "k-plato")
    }

    @Test func aSavedMachineComesBackWithItsKey() throws {
        let (defaults, secrets) = try scratch()
        let plato = host("plato")
        HostPersistence.save([plato], to: defaults)
        try HostPersistence.setAPIKey("k-plato", for: plato.id, in: secrets)

        let loaded = try #require(HostPersistence.load(from: defaults, secrets: secrets))
        #expect(loaded.map(\.apiKey) == ["k-plato"])
    }

    /// Clearing is its own call, and removing a machine takes its key with it.
    @Test func aKeyIsClearedOnlyWhenSomebodyAsks() throws {
        let (_, secrets) = try scratch()
        let plato = host("plato")
        let name = SecretStore.remoteAPIKeyName(for: plato.id)
        try HostPersistence.setAPIKey("k-plato", for: plato.id, in: secrets)

        try HostPersistence.setAPIKey("", for: plato.id, in: secrets)
        #expect(try secrets.value(for: name) == nil)

        try HostPersistence.setAPIKey("k-again", for: plato.id, in: secrets)
        try HostPersistence.forget(plato, in: secrets)
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

    @Test func everyKeychainKeyMovesToTheFileAndTheItemGoes() throws {
        let (defaults, secrets) = try scratch()
        let plato = StoredHost(host("plato"))
        let hal = StoredHost(host("hal9000"))
        let deleted = Deletions()
        let source = source([plato.id: .found("k-plato"), hal.id: .found("k-hal")], into: deleted)

        LegacyKeychain.migrateIfNeeded([plato, hal], into: secrets, defaults: defaults, from: source)

        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: plato.id)) == "k-plato")
        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: hal.id)) == "k-hal")
        #expect(deleted.ids == [plato.id, hal.id])
        #expect(defaults.bool(forKey: LegacyKeychain.migratedKey))
    }

    /// Idempotent: a second launch reads no Keychain item at all, so a key
    /// the person later cleared can never come back.
    @Test func theMoveHappensOnce() throws {
        let (defaults, secrets) = try scratch()
        let plato = StoredHost(host("plato"))
        LegacyKeychain.migrateIfNeeded([plato], into: secrets, defaults: defaults,
                                       from: source([plato.id: .found("k-plato")]))
        try secrets.clear(SecretStore.remoteAPIKeyName(for: plato.id))

        LegacyKeychain.migrateIfNeeded([plato], into: secrets, defaults: defaults,
                                       from: source([plato.id: .found("k-plato")]))
        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: plato.id)) == nil)
    }

    /// **Fails today**: `.found` wrote the Keychain value unconditionally, and
    /// one `.unreadable` host leaves the done-flag off so the WHOLE migration
    /// re-runs next launch. So: launch 1 the keychain is locked and `plato`
    /// loads keyless; the user opens Edit Machine and types the current key;
    /// launch 2 the item reads and the OLD, rotated value silently replaces
    /// what they just typed, and the item is deleted. The file is newer by
    /// construction -- nothing but a deliberate save puts a value there -- so
    /// a value already in it wins, and the stale item still goes.
    @Test func aRetriedMigrationKeepsTheKeyTheUserJustTyped() throws {
        let (defaults, secrets) = try scratch()
        let plato = StoredHost(host("plato"))
        let name = SecretStore.remoteAPIKeyName(for: plato.id)

        // Launch 1: the item will not read, so the move stays unfinished.
        LegacyKeychain.migrateIfNeeded([plato], into: secrets, defaults: defaults,
                                       from: source([plato.id: .unreadable]))
        #expect(!defaults.bool(forKey: LegacyKeychain.migratedKey))

        // The user retypes the current key while the old one sits in the item.
        try HostPersistence.setAPIKey("k-typed", for: plato.id, in: secrets)

        // Launch 2: the item reads at last -- and holds the rotated key.
        let deleted = Deletions()
        LegacyKeychain.migrateIfNeeded([plato], into: secrets, defaults: defaults,
                                       from: source([plato.id: .found("k-rotated")], into: deleted))

        #expect(try secrets.value(for: name) == "k-typed")
        #expect(deleted.ids == [plato.id], "the stale item still goes")
        #expect(defaults.bool(forKey: LegacyKeychain.migratedKey))
    }

    /// The Keychain item is the only other copy, so it is deleted only once
    /// the FILE says it holds the key -- read back from disk, not from the
    /// cache that would answer with what we meant to write.
    @Test func anItemGoesOnlyAfterTheFileProvesItHasTheKey() throws {
        let (_, secrets) = try scratch()
        let plato = StoredHost(host("plato"))
        let name = SecretStore.remoteAPIKeyName(for: plato.id)

        try secrets.set("k-plato", for: name)
        #expect(try secrets.persistedValue(for: name) == "k-plato")

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
        let locked = StoredHost(host("plato"))
        let readable = StoredHost(host("hal9000"))

        LegacyKeychain.migrateIfNeeded([locked, readable], into: secrets, defaults: defaults,
                                       from: source([locked.id: .unreadable,
                                                     readable.id: .found("k-hal")]))

        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: readable.id)) == "k-hal")
        #expect(!defaults.bool(forKey: LegacyKeychain.migratedKey))

        LegacyKeychain.migrateIfNeeded([locked, readable], into: secrets, defaults: defaults,
                                       from: source([locked.id: .found("k-plato"),
                                                     readable.id: .absent]))
        #expect(try secrets.value(for: SecretStore.remoteAPIKeyName(for: locked.id)) == "k-plato")
        #expect(defaults.bool(forKey: LegacyKeychain.migratedKey))
    }
}
