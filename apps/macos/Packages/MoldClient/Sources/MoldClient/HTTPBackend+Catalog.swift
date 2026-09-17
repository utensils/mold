import Foundation

// Catalog search, install, and per-machine credentials.
public extension HTTPBackend {
    /// Installs a `cv:`/`hf:` row. NO BODY -- consent for a gated built-in
    /// reached this way goes through `acceptLicenses` and a retry
    /// (`catalog_api.rs:513-516`). 202 with a possibly-null primary job id.
    func installCatalogEntry(id: String) async throws -> CatalogInstall {
        let data = try await bytes(for: request(catalogDownloadPath(id), method: "POST"))
        do {
            return try MoldJSON.decoder.decode(CatalogInstall.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
    }

    func searchCatalog(_ query: CatalogQuery) async throws -> CatalogListing {
        let qs = query.queryString
        return try await get(qs.isEmpty ? "/api/catalog/search" : "/api/catalog/search?\(qs)")
    }

    func catalogEntry(id: String) async throws -> CatalogEntry {
        try await get(catalogEntryPath(id))
    }

    func catalogCredentials() async throws -> CatalogCredentialStatus {
        try await get("/api/catalog/credentials")
    }

    @discardableResult
    func setCatalogCredential(_ provider: String, token: String) async throws -> CatalogCredentialStatus {
        struct Body: Encodable { let token: String }
        return try await send(
            "/api/catalog/credentials/\(escaped(provider))", method: "PUT", body: Body(token: token))
    }

    @discardableResult
    func clearCatalogCredential(_ provider: String) async throws -> CatalogCredentialStatus {
        let data = try await bytes(
            for: request("/api/catalog/credentials/\(escaped(provider))", method: "DELETE"))
        do {
            return try MoldJSON.decoder.decode(CatalogCredentialStatus.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
    }
}

extension HTTPBackend {
    /// `/api/catalog/*id` is a WILDCARD route: unlike `escaped(_:)`, a
    /// literal `/` in the id must survive here, because `hf:owner/repo` is
    /// two segments the wildcard is built to split, not one component to
    /// protect from being split. `.urlPathAllowed` already allows `/`.
    ///
    /// Encoded as the WHOLE path, not the bare id: `addingPercentEncoding`
    /// treats a colon before the first slash as a possible URI SCHEME (RFC
    /// 3986's `path-noscheme` ambiguity) and escapes it regardless of the
    /// allowed set -- `"hf:owner/repo"` alone becomes `"hf%3Aowner/repo"`,
    /// but the same id after `/api/catalog/` is no longer a leading segment
    /// and survives untouched.
    private func catalogPath(_ path: String) -> String {
        path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? path
    }

    /// Split out so the URL can be pinned without a network call, the same
    /// precedent as `loraPath(model:)`.
    func catalogEntryPath(_ id: String) -> String { catalogPath("/api/catalog/\(id)") }
    func catalogDownloadPath(_ id: String) -> String { catalogPath("/api/catalog/\(id)/download") }
}
