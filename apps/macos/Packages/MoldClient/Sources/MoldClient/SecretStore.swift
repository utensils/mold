import Foundation
import Synchronization

/// This Mac's credentials, in an owner-only file.
///
/// A port of `desktop/src-tauri/src/secrets.rs`, and deliberately NOT the
/// Keychain. `.claude/rules/desktop.md` already states the rule for the Tauri
/// app -- "owner-only `secrets.json` under app data … deliberately NOT the
/// macOS Keychain, whose prompts users found obnoxious; don't reintroduce
/// `keyring`" -- and one mold on one Mac must not answer that question two
/// different ways. The Keychain also swallowed every `OSStatus`, so a locked
/// keychain or a re-signed dev build read back as "this machine has no key"
/// and the next save deleted every one of them (review 05-H5).
///
/// The document is `{"<name>": "<value>"}` and nothing else. Names are
/// constrained to the two this app has, so the store can never be pointed at
/// another file or grow a slot nobody declared.
public final class SecretStore: Sendable {
    /// The key the in-process engine is started with, and that "This Mac"
    /// then presents back to it.
    public static let localEngineKeyName = "local-engine-api-key"
    /// One machine's key. The suffix is that host's UUID -- the same identity
    /// `StoredHost` writes to preferences, so the two files join on it.
    public static func remoteAPIKeyName(for host: UUID) -> String {
        perHostPrefix + host.uuidString
    }

    private static let perHostPrefix = "remote-api-key."

    /// `AppStorageSuite`'s two domains, spelled here because this package may
    /// not reach into the app. A UAT run under `MOLD_NATIVE_FRESH` gets the
    /// throwaway twin, so it can exercise a first launch without ever being
    /// able to read -- or delete -- anybody's real keys.
    public static let directoryName = "io.utensils.mold.native"
    public static let freshDirectoryName = "io.utensils.mold.native.fresh"

    struct Loaded {
        var map: [String: String]
        /// The file existed and did not parse. It is moved aside ONCE, on the
        /// next write, rather than clobbered: a transient parse failure must
        /// never silently destroy credentials (`secrets.rs:36-42`).
        var corrupt: Bool
    }

    let url: URL
    /// One lock around the whole read-modify-write cycle. An unserialised
    /// load-modify-save loses whichever write lands first (`secrets.rs:44-50`).
    let state = Mutex<Loaded?>(nil)

    public init(directory: URL) {
        url = directory.appending(path: "secrets.json")
    }

    /// Where the document lives -- for a diagnostic, and for a second instance
    /// on the same home.
    public var directory: URL { url.deletingLastPathComponent() }

    /// `~/Library/Application Support/io.utensils.mold.native`. Deliberately
    /// does not throw and does not create anything: resolving where the file
    /// GOES cannot fail, and the directory is made by the first write.
    public static func applicationSupport(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        in fileManager: FileManager = .default
    ) -> URL {
        let root = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.homeDirectoryForCurrentUser.appending(path: "Library/Application Support")
        // The app's `NativeUAT` gate, spelled again here because MoldClient
        // cannot import the app: a Release build reads no UAT hook, so it can
        // never be pointed at the throwaway directory.
        #if DEBUG
        let fresh = environment["MOLD_NATIVE_FRESH"] != nil
        #else
        let fresh = false
        #endif
        return root.appending(path: fresh ? freshDirectoryName : directoryName)
    }

    /// The store every surface in the app shares. Lazy, so it reads the
    /// environment once this process is running rather than at load.
    public static let shared = SecretStore(directory: applicationSupport())

    // MARK: - Reading and writing one name

    public func value(for name: String) throws -> String? {
        try Self.check(name)
        return try state.withLock { slot in
            try loaded(&slot).map[name]
        }
    }

    public func set(_ value: String, for name: String) throws {
        try Self.check(name)
        try write(name) { $0[name] = value }
    }

    public func clear(_ name: String) throws {
        try Self.check(name)
        try write(name) { $0.removeValue(forKey: name) }
    }

    /// The value on DISK, whatever this process has cached.
    ///
    /// `LegacyKeychain` deletes an item only once this says the file holds the
    /// key: the Keychain was the only other copy, and the in-process cache
    /// would answer with what we MEANT to write (review E1).
    public func persistedValue(for name: String) throws -> String? {
        try Self.check(name)
        return try state.withLock { _ in
            var cold: Loaded?
            return try loaded(&cold).map[name]
        }
    }

    /// One read-modify-write, under the process mutex AND the cross-process
    /// lock, re-reading the document inside both -- so a second live writer's
    /// key is never lost to this one's cache.
    private func write(_ name: String, _ change: (inout [String: String]) -> Void) throws {
        try state.withLock { slot in
            try locked {
                slot = nil
                var current = try loaded(&slot)
                change(&current.map)
                try save(&current)
                slot = current
            }
        }
    }

    /// The allowlist, as `secrets.rs:60-75` spells it: a known constant, or
    /// the per-host prefix with a suffix of ASCII letters, digits, `-`, `_`
    /// and `.` -- which is what keeps a name from ever being a path.
    static func check(_ name: String) throws {
        if name == localEngineKeyName { return }
        guard name.hasPrefix(perHostPrefix) else {
            throw SecretStoreError.unknownName(name)
        }
        let suffix = name.dropFirst(perHostPrefix.count)
        let allowed = suffix.utf8.allSatisfy { byte in
            (byte >= 0x30 && byte <= 0x39) || (byte >= 0x41 && byte <= 0x5A)
                || (byte >= 0x61 && byte <= 0x7A) || byte == 0x2D || byte == 0x5F || byte == 0x2E
        }
        guard !suffix.isEmpty, allowed else { throw SecretStoreError.unknownName(name) }
    }
}

