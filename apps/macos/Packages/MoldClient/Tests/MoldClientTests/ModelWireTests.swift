import Foundation
import Testing

@testable import MoldClient

// M5 S1b: the models/catalog/downloads wire. Fixtures captured by GET
// against plato (`100.105.134.43:7680`) on 2026-09-17, plus one synthetic
// listing for the one state plato itself cannot show.

private func platoModels() throws -> [Model] {
    try MoldJSON.decoder.decode([Model].self, from: RepoFixtures.fixture("models-plato.json"))
}

private func repairModels() throws -> [Model] {
    try MoldJSON.decoder.decode([Model].self, from: RepoFixtures.fixture("models-repair.json"))
}

/// Design fact 6: repair is `downloaded && remaining > 0` and nothing else.
/// A not-yet-installed model also carries a positive remainder -- its WHOLE
/// size -- so reading the remainder alone calls every available model
/// broken. Fails today: there is no `installState`.
@Test func aDownloadedModelWithBytesOutstandingNeedsRepair() throws {
    let flux = try #require(repairModels().first { $0.name == "flux-dev:q4" })
    guard case let .needsRepair(bytes) = flux.installState else {
        Issue.record("expected .needsRepair, got \(flux.installState)")
        return
    }
    #expect(bytes == 998_877_665)
}

@Test func aModelNobodyHasStartedIsAvailableNotBroken() throws {
    // Not downloaded on plato, with its whole size outstanding.
    let schnell = try #require(platoModels().first { $0.name == "flux-schnell:bf16" })
    guard case let .available(bytes) = schnell.installState else {
        Issue.record("expected .available, got \(schnell.installState)")
        return
    }
    #expect(bytes == 23_782_506_688)
}

@Test func anInstalledRowReportsItsOwnBytesAndAnAvailableOneReportsNone() throws {
    let models = try platoModels()
    let installed = try #require(models.first { $0.name == "flux-schnell:q8" })
    #expect(installed.diskUsageBytes == 23_061_759_552)
    #expect(installed.installState == .installed)

    let available = try #require(models.first { $0.name == "flux-schnell:bf16" })
    #expect(available.diskUsageBytes == nil)
}

/// `ModelsPane+Grouping`'s `installedOnly` scope reads `ModelStore.ready`,
/// which is `generators ∩ isReady` -- a non-generator installed row like an
/// upscaler or the prompt-expansion LLM never reaches it. MoldClient itself
/// must decode and keep every row regardless (design fact 5, M5): the
/// Installed table is a management surface, not a generator picker.
@Test func everyInstalledFamilySurvivesTheListing() throws {
    let models = try platoModels()
    let upscaler = try #require(models.first { $0.name == "real-esrgan-x4plus:fp16" })
    #expect(upscaler.installState == .installed)
    #expect(upscaler.isUpscaler)

    let expander = try #require(models.first { $0.name == "qwen3-expand:q8" })
    #expect(expander.installState == .installed)
    #expect(expander.isUtility)
}

@Test func aCatalogModelTakesTheCatalogRouteAndAManifestNameDoesNot() {
    #expect(Model.isCatalogName("cv:252914"))
    #expect(Model.isCatalogName("hf:owner/repo"))
    #expect(!Model.isCatalogName("flux-dev:q4"))
    #expect(!Model.isCatalogName("flux-dev"))
}

/// Design fact 4: `options` is a candidate list for a
/// `models.<name>.<component>_path` override, not provenance -- decoded so
/// the type round-trips, and read by nobody. This test says so in its name.
@Test func aComponentsAnswerKeepsItsHundredOptionsAndNamesItsRepairModel() throws {
    let response = try MoldJSON.decoder.decode(
        ModelComponentsResponse.self, from: RepoFixtures.fixture("components-flux-schnell.json"))
    #expect(response.model == "flux-schnell:q8")
    let transformer = try #require(response.components.first { $0.kind == "transformer" })
    #expect((transformer.options ?? []).count > 100)
    #expect(transformer.repairModel == "flux-schnell:q8")
    #expect(transformer.present)
}

/// Design fact 18: every row of a real three-row search had `license: null`
/// and all three flags null. All-null is the ORDINARY case and means NO
/// INFORMATION -- rendering it as "commercial: no" would be a refusal
/// nobody made.
@Test func aCatalogRowWithNoLicenceSaysNothing() throws {
    let listing = try MoldJSON.decoder.decode(
        CatalogListing.self, from: RepoFixtures.fixture("catalog-dreamshaper.json"))
    #expect(listing.entries.count == 3)
    #expect(listing.entries.allSatisfy { $0.licenseFlags.isEmpty })
    #expect(listing.total == 32)
    let first = try #require(listing.entries.first)
    #expect(first.id == "cv:128713")
    #expect(first.companionDetails.count == 2)
}

@Test func aSearchQueryEscapesItsTextAndOmitsWhatWasNotAsked() {
    let query = CatalogQuery(text: "a b/c#d", family: "sd15", pageSize: 3)
    let qs = query.queryString
    #expect(qs.contains("q=a%20b%2Fc%23d"))
    #expect(qs.contains("family=sd15"))
    #expect(qs.contains("page_size=3"))
    // Nothing asked for -- kind, source, sort, page, includeNSFW -- appears.
    #expect(!qs.contains("kind="))
    #expect(!qs.contains("source="))
    #expect(!qs.contains("sort="))
    #expect(!qs.contains("&page="))
    #expect(!qs.contains("include_nsfw"))

    #expect(CatalogQuery().queryString.isEmpty)
}

/// `catalog_api.rs:545-561`: a companion-only install has no primary job (no
/// primary file was missing) but is still real, queued work.
@Test func aCompanionOnlyInstallHasNoPrimaryJobAndIsStillTwoJobs() throws {
    let json = Data("""
    {"primary_job_id": null, "companion_jobs": [
        {"name": "clip-l", "job_id": "j1"}, {"name": "sd-vae-ft-mse", "job_id": "j2"}
    ]}
    """.utf8)
    let install = try MoldJSON.decoder.decode(CatalogInstall.self, from: json)
    #expect(install.primaryJobId == nil)
    #expect(install.jobIDs == ["j1", "j2"])
}

/// Fails today: `DownloadEvent` has no `listing` field, so the first frame
/// every subscriber gets is silently dropped.
@Test func aSnapshotFrameCarriesTheWholeListing() throws {
    let json = Data("""
    {"type": "snapshot", "listing": {
        "active_jobs": [{"id": "a", "model": "flux-dev:q4", "status": "active",
            "files_done": 1, "files_total": 3, "bytes_done": 10, "bytes_total": 100}],
        "queued": [], "history": []
    }}
    """.utf8)
    let event = try MoldJSON.decoder.decode(DownloadEvent.self, from: json)
    #expect(event.type == "snapshot")
    let listing = try #require(event.listing)
    #expect(listing.activeJobs.count == 1)
    #expect(listing.activeJobs.first?.status == .active)
    #expect(listing.queued.isEmpty)
}

/// Design fact 9: a stored token wins over the environment one, and the
/// masked form is the only version of it that ever leaves the machine.
@Test func aMaskedTokenIsTheOnlyFormThatEverArrives() throws {
    let status = try MoldJSON.decoder.decode(
        CatalogCredentialStatus.self, from: RepoFixtures.fixture("credentials-plato.json"))
    #expect(status.hf.configured)
    #expect(status.hf.isFromEnvironment)
    #expect(status.hf.masked == "hf_\u{2022}\u{2022}\u{2022}\u{2022}hhml")
    #expect(!status.civitai.configured)
    #expect(status.civitai.masked == nil)
}
