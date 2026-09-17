import Foundation

/// Browsing the live HF/Civitai catalog, installing from it, and this
/// machine's stored provider credentials. New in M5 S1b.
public protocol MoldCatalogBackend: Sendable {
    /// Installs a `cv:`/`hf:` row. 202 with a possibly-null primary job id --
    /// a null primary with non-empty companion jobs means only companions
    /// were missing, not a failure.
    func installCatalogEntry(id: String) async throws -> CatalogInstall
    func searchCatalog(_ query: CatalogQuery) async throws -> CatalogListing
    func catalogEntry(id: String) async throws -> CatalogEntry
    func catalogCredentials() async throws -> CatalogCredentialStatus
    @discardableResult
    func setCatalogCredential(_ provider: String, token: String) async throws -> CatalogCredentialStatus
    @discardableResult
    func clearCatalogCredential(_ provider: String) async throws -> CatalogCredentialStatus
}
