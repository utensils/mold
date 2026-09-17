import Foundation

// Config rows, profiles, and mobile pairing -- Settings' whole wire surface.
// `config()`, `setConfig` and `resetConfig` moved here from `+Create.swift`
// (M7 S1 decision 2): config was always an odd fit there, and profiles and
// pairing belong beside it rather than opening a fourteenth concern.
public extension HTTPBackend {
    /// The whole listing, never a per-key GET: a model with no row 404s
    /// (`config_keys.rs:586-593`), so a never-configured model and a
    /// configured-but-unset one are only told apart by reading everything.
    func config() async throws -> ConfigListing { try await get("/api/config") }

    /// One key. `models.<name>.<field>` CREATES the model's config row
    /// (`config_keys.rs:759-762`), which is why nothing has to be configured
    /// first.
    @discardableResult
    func setConfig(_ key: String, to value: ConfigScalar) async throws -> ConfigEntry {
        try await send("/api/config/\(escaped(key))", method: "PUT", body: ConfigSet(value: value))
    }

    /// Drops the DB row so the key falls back to file/env/default. The body
    /// carries the fallback value.
    @discardableResult
    func resetConfig(_ key: String) async throws -> ConfigEntry {
        try await send("/api/config/\(escaped(key))", method: "DELETE", body: EmptyBody())
    }

    /// `GET /api/config/profiles`. Read-only here (design decision 9):
    /// switching writes the active-profile row without touching the running
    /// server's loaded config (`routes_config.rs:380-392`), so a table that
    /// could switch would show one profile's values while an edit landed in
    /// another's.
    func configProfiles() async throws -> ConfigProfiles { try await get("/api/config/profiles") }

    /// `POST /api/pairing/sessions`: starts a two-minute, one-use mobile
    /// pairing handoff. A keyless host answers with `token: nil` -- there is
    /// no key to hand over (`routes.rs:9557-9598`).
    func pairingSession() async throws -> PairingSession {
        try await post("/api/pairing/sessions", body: EmptyBody())
    }

    /// `GET /api/pairing/clients`. `pairingAvailable` is `true` even on a
    /// keyless host (`routes.rs:9678-9685`) -- read `PairedClients.canPair`,
    /// never this alone.
    func pairedClients() async throws -> PairedClients { try await get("/api/pairing/clients") }

    /// `DELETE /api/pairing/clients/:id`, 204. `escaped(id)` -- the same rule
    /// the catalog id bug taught (`HTTPBackendURLTests`).
    func revokePairedClient(_ id: String) async throws {
        try await delete("/api/pairing/clients/\(escaped(id))")
    }
}

/// The `PUT /api/config/:key` body. `routes_config.rs`'s `ConfigSetRequest`.
struct ConfigSet: Encodable {
    let value: ConfigScalar
}
