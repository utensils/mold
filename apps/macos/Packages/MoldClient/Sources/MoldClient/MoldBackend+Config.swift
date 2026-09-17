import Foundation

/// Config rows, profiles, and mobile pairing -- Settings' whole wire surface.
public protocol MoldConfigBackend: Sendable {
    /// Every model's config surface, as `/api/config` reports it. An absent
    /// per-model key means "never configured"; a present key with a null
    /// value means the same thing.
    func config() async throws -> ConfigListing
    /// Sets one key. `models.<name>.<field>` CREATES the model's row.
    @discardableResult
    func setConfig(_ key: String, to value: ConfigScalar) async throws -> ConfigEntry
    /// Drops the DB row so the key falls back to file/env/default.
    @discardableResult
    func resetConfig(_ key: String) async throws -> ConfigEntry
    /// The active profile and the rest. Read-only here: switching one leaves
    /// the running server's loaded config untouched.
    func configProfiles() async throws -> ConfigProfiles
    /// Starts a two-minute, one-use mobile pairing handoff.
    func pairingSession() async throws -> PairingSession
    /// Every client currently holding paired access to this machine.
    func pairedClients() async throws -> PairedClients
    /// Revokes one paired client's access.
    func revokePairedClient(_ id: String) async throws
}
