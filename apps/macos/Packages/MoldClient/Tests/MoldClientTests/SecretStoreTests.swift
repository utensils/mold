import Foundation
import Testing

@testable import MoldClient

/// A port of `desktop/src-tauri/src/secrets.rs`'s own suite (`:172-311`),
/// including `secrets_file_is_owner_only`. The Rust store is the reference
/// implementation for this file's format, its name rule, its corrupt-file
/// parking and its 0600 mode, so its tests are the ones that say whether the
/// port is faithful.
struct SecretStoreTests {
    private func scratch() throws -> (SecretStore, URL) {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "mold-secret-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return (SecretStore(directory: dir), dir)
    }

    private func hostName() -> String {
        SecretStore.remoteAPIKeyName(for: UUID())
    }

    @Test func roundTripsAndClears() throws {
        let (store, _) = try scratch()
        let name = hostName()
        #expect(try store.value(for: name) == nil)
        try store.set("k1", for: name)
        #expect(try store.value(for: name) == "k1")
        try store.set("k2", for: name)
        #expect(try store.value(for: name) == "k2")
        try store.clear(name)
        #expect(try store.value(for: name) == nil)
    }

    @Test func rejectsUnknownNames() throws {
        let (store, _) = try scratch()
        #expect(throws: SecretStoreError.self) { try store.value(for: "ssh-private-key") }
        #expect(throws: SecretStoreError.self) { try store.set("x", for: "anything") }
        // The desktop's other slots are not this app's: it has no catalog
        // token on this Mac at all (they live on the machine), and no RunPod.
        #expect(throws: SecretStoreError.self) { try store.set("x", for: "hf-token") }
    }

    @Test func rejectsMalformedPerHostNames() throws {
        let (store, _) = try scratch()
        #expect(throws: SecretStoreError.self) { try store.set("x", for: "remote-api-key.") }
        #expect(throws: SecretStoreError.self) { try store.set("x", for: "remote-api-key./etc/passwd") }
        #expect(throws: SecretStoreError.self) { try store.set("x", for: "local-engine-api-key.evil") }
    }

    @Test func persistsAcrossInstances() throws {
        let (store, dir) = try scratch()
        let name = hostName()
        try store.set("cv_1", for: name)
        #expect(try SecretStore(directory: dir).value(for: name) == "cv_1")
    }

    /// The file is `{"name": "value"}` and nothing else -- the same document
    /// the Tauri app writes, so neither build has to learn a second shape.
    @Test func theFileIsAFlatNameToValueMap() throws {
        let (store, dir) = try scratch()
        try store.set("k1", for: SecretStore.localEngineKeyName)
        let raw = try Data(contentsOf: dir.appending(path: "secrets.json"))
        let map = try MoldJSON.localDecoder.decode([String: String].self, from: raw)
        #expect(map == [SecretStore.localEngineKeyName: "k1"])
    }

    @Test func corruptFileIsPreservedNotClobbered() throws {
        let (store, dir) = try scratch()
        let path = dir.appending(path: "secrets.json")
        try Data("not json {".utf8).write(to: path)
        // Reads degrade to empty…
        #expect(try store.value(for: SecretStore.localEngineKeyName) == nil)
        // …and the first write moves the original aside instead of erasing it.
        try store.set("new", for: SecretStore.localEngineKeyName)
        let parked = try String(contentsOf: dir.appending(path: "secrets.json.corrupt"), encoding: .utf8)
        #expect(parked == "not json {")
        #expect(try store.value(for: SecretStore.localEngineKeyName) == "new")
    }

    /// **Fails today**: the temporary file was made by `Data.write(to:)`,
    /// which creates at the process umask -- 0644 -- and was chmodded only on
    /// the NEXT statement. A complete copy of every credential existed
    /// world-readable for that window, and if the write itself threw
    /// (ENOSPC, EDQUOT, EIO) it was never chmodded and never removed, because
    /// it was created OUTSIDE the `do` that cleans up (review E3).
    @Test func aTemporaryFileIsOwnerOnlyFromItsFirstByte() throws {
        let (_, dir) = try scratch()
        let path = dir.appending(path: "probe.tmp")

        try SecretStore.writeOwnerOnly(Data("secret".utf8), to: path)

        let attributes = try FileManager.default
            .attributesOfItem(atPath: path.path(percentEncoded: false))
        #expect((attributes[.posixPermissions] as? NSNumber)?.int16Value ?? 0 & 0o777 == 0o600)
        #expect(try Data(contentsOf: path) == Data("secret".utf8))
    }

    /// A write that cannot land leaves nothing readable behind and changes
    /// nothing that was already there.
    @Test func aFailedWriteLeavesNoPlaintextBehind() throws {
        let (store, dir) = try scratch()
        let name = hostName()
        try store.set("kept", for: name)
        let before = try Data(contentsOf: dir.appending(path: "secrets.json"))

        try FileManager.default.setAttributes([.posixPermissions: 0o500],
                                              ofItemAtPath: dir.path(percentEncoded: false))
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o700],
                                                   ofItemAtPath: dir.path(percentEncoded: false))
        }

        #expect(throws: (any Error).self) { try store.set("new", for: name) }
        let left = try FileManager.default.contentsOfDirectory(atPath: dir.path(percentEncoded: false))
        #expect(!left.contains { $0.hasSuffix(".tmp") })
        #expect(try Data(contentsOf: dir.appending(path: "secrets.json")) == before)
    }

    /// **Fails today**: `try? Data(contentsOf:)` made ENOENT and EACCES the
    /// same answer, and only a PARSE failure set `corrupt`. A file that exists
    /// but will not read -- the wrong owner after a `sudo` launch, an ACL, a
    /// transient I/O error -- became an empty store, and the very next `set`
    /// renamed a one-entry document over every other machine's key, with no
    /// parked copy and nothing said (review E2).
    ///
    /// Parking is not enough either: whatever stopped the read will stop the
    /// rename. The store refuses to read AND refuses to write, and the error
    /// is the one `HostStore.report` puts in front of the person.
    @Test func anUnreadableFileIsNeverWrittenOver() throws {
        let (store, dir) = try scratch()
        let name = hostName()
        try store.set("kept", for: name)
        let path = dir.appending(path: "secrets.json")
        let before = try Data(contentsOf: path)

        // A fresh store, so nothing is cached from the write above.
        let cold = SecretStore(directory: dir)
        try FileManager.default.setAttributes([.posixPermissions: 0o000],
                                              ofItemAtPath: path.path(percentEncoded: false))
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o600],
                                                   ofItemAtPath: path.path(percentEncoded: false))
        }

        #expect(throws: SecretStoreError.self) { try cold.value(for: name) }
        #expect(throws: SecretStoreError.self) { try cold.set("new", for: name) }
        #expect(throws: SecretStoreError.self) { try cold.clear(name) }

        try FileManager.default.setAttributes([.posixPermissions: 0o600],
                                              ofItemAtPath: path.path(percentEncoded: false))
        #expect(try Data(contentsOf: path) == before)
        // Nothing was cached either: the moment it reads again, it reads.
        #expect(try cold.value(for: name) == "kept")
    }

    @Test func secretsFileIsOwnerOnly() throws {
        let (store, dir) = try scratch()
        try store.set("k1", for: hostName())
        let attributes = try FileManager.default
            .attributesOfItem(atPath: dir.appending(path: "secrets.json").path(percentEncoded: false))
        let mode = (attributes[.posixPermissions] as? NSNumber)?.int16Value ?? 0
        #expect(mode & 0o777 == 0o600)
    }

    @Test func concurrentWritersDoNotLoseUpdates() async throws {
        let (store, _) = try scratch()
        let names = (0..<8).map { _ in hostName() }
        await withTaskGroup(of: Void.self) { group in
            for (index, name) in names.enumerated() {
                group.addTask { try? store.set("k\(index)", for: name) }
            }
        }
        for (index, name) in names.enumerated() {
            #expect(try store.value(for: name) == "k\(index)")
        }
    }

    // MARK: - The engine's own key

    /// `SecretStore::local_server_api_key`'s precedence, exactly
    /// (`secrets.rs:103-119`): a non-empty `MOLD_API_KEY` is the operator's
    /// override, then whatever this install already minted, then a fresh UUID
    /// that is stored before it is returned.
    @Test func theEngineKeyIsMintedOnceAndPersists() throws {
        let (store, dir) = try scratch()
        let first = try store.localEngineAPIKey(environment: [:])
        #expect(!first.isEmpty)
        #expect(try store.localEngineAPIKey(environment: [:]) == first)
        #expect(try SecretStore(directory: dir).localEngineAPIKey(environment: [:]) == first)
    }

    @Test func theEnvironmentOverridesTheStoredEngineKey() throws {
        let (store, _) = try scratch()
        try store.set("stored", for: SecretStore.localEngineKeyName)
        #expect(try store.localEngineAPIKey(environment: ["MOLD_API_KEY": "operator"]) == "operator")
        // Empty is not an override -- it is an unset variable spelled out.
        #expect(try store.localEngineAPIKey(environment: ["MOLD_API_KEY": ""]) == "stored")
    }

    // MARK: - Where the file lives

    /// A UAT run must never be able to reach the real keys, which is the same
    /// promise `AppStorageSuite` makes for preferences.
    @Test func aFreshRunUsesAThrowawayDirectory() throws {
        let real = SecretStore.applicationSupport(environment: [:])
        let fresh = SecretStore.applicationSupport(environment: ["MOLD_NATIVE_FRESH": "1"])
        #expect(real.lastPathComponent == SecretStore.directoryName)
        #expect(fresh.lastPathComponent == SecretStore.freshDirectoryName)
        #expect(real != fresh)
    }
}
