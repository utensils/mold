import Foundation

/// `GET /api/catalog/credentials` -- both providers this host can hold a
/// token for. Credentials live on the MACHINE, not on this Mac
/// (`catalog_credentials.rs:25`), so Settings ▸ Accounts is keyed on
/// whichever host is selected, exactly like every other per-machine pane.
public struct CatalogCredentialStatus: Codable, Hashable, Sendable {
    public let hf: CatalogCredentialState
    public let civitai: CatalogCredentialState
}

/// One provider's credential state. The token itself never leaves the
/// machine -- `masked` is the only form of it a client ever sees
/// (`catalog_credentials.rs:171-207`).
public struct CatalogCredentialState: Codable, Hashable, Sendable {
    public let configured: Bool
    /// `"server"` for a token stored on this host, `"environment"` for one
    /// read from `HF_TOKEN` / `CIVITAI_TOKEN`. `nil` when unconfigured.
    public let source: String?
    public let masked: String?

    /// A STORED token wins over the environment one on this machine
    /// (`catalog_credentials.rs:194-202`), so setting a value here changes
    /// what actually gets used even though the environment variable is still
    /// set and unchanged.
    public var isFromEnvironment: Bool { source == "environment" }
}
