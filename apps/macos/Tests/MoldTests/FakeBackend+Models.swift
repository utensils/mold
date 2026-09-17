import Foundation
import MoldClient
@testable import Mold

// M5 S1b: models, catalog and downloads. Stored witnesses live in
// `FakeBackend.swift` itself -- an extension cannot declare stored
// properties -- this file is the route implementations that read them.
extension FakeBackend {
    // MARK: - Models

    @discardableResult
    func deleteModel(_ model: String) async throws -> ModelRemoval {
        try record("deleteModel")
        deletedModels.append(model)
        guard let removal = removalAnswers[model] else { throw notPlanted() }
        return removal
    }

    func modelComponents(_ model: String) async throws -> ModelComponentsResponse {
        try record("modelComponents")
        guard let rows = componentRows[model] else { throw notPlanted() }
        return rows
    }

    func loadModel(_ model: String, gpu: Int?) async throws {
        try record("loadModel")
        loadedModels.append((model, gpu))
    }

    func unloadModel(model: String?, gpu: Int?) async throws {
        try record("unloadModel")
        unloadedModels.append((model, gpu))
    }

    func downloads() async throws -> DownloadsListing {
        try record("downloads")
        guard let downloadsListing else { throw notPlanted() }
        return downloadsListing
    }

    // MARK: - Catalog

    func installCatalogEntry(id: String) async throws -> CatalogInstall {
        try record("installCatalogEntry")
        catalogInstalls.append(id)
        guard let catalogInstallAnswer else { throw notPlanted() }
        return catalogInstallAnswer
    }

    func searchCatalog(_ query: CatalogQuery) async throws -> CatalogListing {
        try record("searchCatalog")
        guard let listing = catalogPages[query.queryString] else { throw notPlanted() }
        return listing
    }

    func catalogEntry(id: String) async throws -> CatalogEntry {
        try record("catalogEntry")
        throw notPlanted()
    }

    func catalogCredentials() async throws -> CatalogCredentialStatus {
        try record("catalogCredentials")
        guard let credentialStatus else { throw notPlanted() }
        return credentialStatus
    }

    @discardableResult
    func setCatalogCredential(_ provider: String, token: String) async throws -> CatalogCredentialStatus {
        try record("setCatalogCredential")
        credentialWrites.append((provider, token))
        guard let credentialStatus else { throw notPlanted() }
        return credentialStatus
    }

    @discardableResult
    func clearCatalogCredential(_ provider: String) async throws -> CatalogCredentialStatus {
        try record("clearCatalogCredential")
        credentialClears.append(provider)
        guard let credentialStatus else { throw notPlanted() }
        return credentialStatus
    }
}
