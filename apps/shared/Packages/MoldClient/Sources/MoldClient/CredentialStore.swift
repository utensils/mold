import Foundation

/// Where a machine's API key is kept, whichever app is asking.
///
/// The Mac keeps keys in `SecretStore`'s owner-only file and never the Keychain
/// (`.claude/rules/desktop.md`); the iPhone keeps them ONLY in the Keychain
/// (`.claude/rules/mobile.md`). Both answer this, keyed by the machine's UUID --
/// the identity each app's stored host list joins on -- so code above the
/// store never learns which one it has.
///
/// Every call throws rather than returning nil for a failure: a locked store
/// read back as "no key" is how a later save once deleted every key (review
/// 05-H5). `nil` means the machine has no key, and nothing else.
public protocol CredentialStore: Sendable {
    func apiKey(for host: UUID) throws -> String?
    func setAPIKey(_ key: String, for host: UUID) throws
    func clearAPIKey(for host: UUID) throws
}

extension SecretStore: CredentialStore {
    public func apiKey(for host: UUID) throws -> String? {
        try value(for: Self.remoteAPIKeyName(for: host))
    }

    public func setAPIKey(_ key: String, for host: UUID) throws {
        try set(key, for: Self.remoteAPIKeyName(for: host))
    }

    public func clearAPIKey(for host: UUID) throws {
        try clear(Self.remoteAPIKeyName(for: host))
    }
}
