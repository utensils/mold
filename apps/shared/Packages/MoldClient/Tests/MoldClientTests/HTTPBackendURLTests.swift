import Foundation
import Testing

@testable import MoldClient

// `URL.appending(path:)` treats its whole argument as ONE path component and
// percent-encodes everything in it, `?` included. So a path written with a
// query string became `/api/gallery%3Fview=trash`, which is a route no mold
// has — and the Recently Deleted shelf listed nothing on a machine with 177
// prints in it, silently, because the app asks conditionally and a shrug
// looks the same as "nothing changed".

private let backend = HTTPBackend(
    host: MoldHost(name: "workstation", baseURL: URL(string: "http://workstation:7680")!)
)

@Test func aQueryStringSurvivesBeingTurnedIntoARequest() {
    let url = backend.request("/api/gallery?view=trash").url
    #expect(url?.absoluteString == "http://workstation:7680/api/gallery?view=trash")
    #expect(url?.query() == "view=trash")
}

@Test func severalParametersAllSurvive() {
    let url = backend.request("/api/gallery/thumbnail/a.png?size=512&fmt=jpeg").url
    #expect(url?.absoluteString
            == "http://workstation:7680/api/gallery/thumbnail/a.png?size=512&fmt=jpeg")
}

@Test func aPlainPathIsUntouched() {
    #expect(backend.request("/api/status").url?.absoluteString == "http://workstation:7680/api/status")
}

/// A base URL with a path of its own — a host behind a reverse proxy at
/// `/mold` — keeps it rather than having it replaced.
@Test func aHostBehindAPrefixKeepsIt() {
    let proxied = HTTPBackend(
        host: MoldHost(name: "p", baseURL: URL(string: "http://box/mold")!)
    )
    #expect(proxied.request("/api/status").url?.absoluteString == "http://box/mold/api/status")
}

/// A filename is ONE path component and must stay encoded: mold's own names
/// carry `~` and a title slug, and a title can hold anything.
@Test func aFilenameWithAwkwardCharactersStaysOneComponent() {
    let url = backend.request("/api/gallery/image/\(backend.escaped("a b~c#d.png"))").url
    #expect(url?.absoluteString.contains("a%20b") == true)
    #expect(url?.absoluteString.contains("#") == false)
}

/// The key rides a header, never the URL — a ticket in a query string ends up
/// in every proxy log between here and the machine.
@Test func theKeyIsAHeaderNotAQueryParameter() {
    let keyed = HTTPBackend(
        host: MoldHost(name: "p", baseURL: URL(string: "http://box:7680")!, apiKey: "secret")
    )
    let request = keyed.request("/api/gallery?view=trash")
    #expect(request.value(forHTTPHeaderField: "X-Api-Key") == "secret")
    #expect(request.url?.absoluteString.contains("secret") == false)
}

/// The app holds `any MoldBackend`, never the concrete type. Every route it
/// calls has to be reachable through the protocol -- a downcast to
/// `HTTPBackend` at the call site is how a failed cast turns into silence
/// instead of an error.
///
/// Nothing here is sent: the closure is built and never invoked, so what is
/// under test is whether it COMPILES.
@Test func aBackendHeldAsTheProtocolReachesEveryRouteTheAppUses() {
    let backend: any MoldBackend = HTTPBackend(
        host: MoldHost(name: "workstation", baseURL: URL(string: "http://workstation:7680")!)
    )
    let _: () async throws -> Void = {
        _ = backend.events()
        _ = backend.batchEvents(id: "b")
        _ = backend.downloadEvents()
        try await backend.emptyTrash()
        _ = try await backend.renameTag("a", to: "b")
        try await backend.cancelJob(id: "j")
        _ = try await backend.media("a.png", trashed: true)
        _ = try await backend.playableURL(for: "a.png")
        _ = try await backend.startDownload(DownloadRequest(model: "m"))
        _ = try await backend.trashedPrints(etag: nil)
        _ = try await backend.expand(ExpandRequest(prompt: "a cat"))
        _ = try await backend.remix(RemixRequest(sourcePrompt: "a cat"))
        _ = try await backend.history(limit: 10)
        try await backend.clearHistory(keeping: 5)
        try await backend.clearHistory()
        _ = try await backend.config()
        _ = try await backend.setConfig("models.m.default_steps", to: .number(20))
        _ = try await backend.resetConfig("models.m.default_steps")
        _ = try await backend.configProfiles()
        _ = try await backend.pairingSession()
        _ = try await backend.pairedClients()
        try await backend.revokePairedClient("client-1")
        _ = try await backend.deleteModel("m")
        _ = try await backend.modelComponents("m")
        try await backend.loadModel("m", gpu: nil)
        try await backend.unloadModel(model: nil, gpu: nil)
        _ = try await backend.downloads()
        _ = try await backend.installCatalogEntry(id: "cv:1")
        _ = try await backend.searchCatalog(CatalogQuery())
        _ = try await backend.catalogEntry(id: "cv:1")
        _ = try await backend.catalogCredentials()
        _ = try await backend.setCatalogCredential("hf", token: "t")
        _ = try await backend.clearCatalogCredential("hf")
    }
    #expect(backend.host.name == "workstation")
}

/// A device id is OPAQUE (`cuda:<32 hex>`) and must ride as ONE path
/// component -- splitting it on `:` addresses a route no mold has.
@Test func aDeviceIdIsSentAsOnePathComponent() {
    let path = backend.deviceMutationPath("cuda:9ffc81c539446490bfd9f68366f98226")
    #expect(path == "/api/devices/cuda:9ffc81c539446490bfd9f68366f98226")
    let url = backend.request(path, method: "PATCH").url
    #expect(url?.path() == "/api/devices/cuda:9ffc81c539446490bfd9f68366f98226")
}

/// `keep` trims to the most recent N instead of clearing everything; there is
/// no per-row delete, so a nil `keep` is the whole-clear route.
@Test func historyIsAskedForNewestFirstWithALimit() {
    #expect(backend.historyPath(limit: 50) == "/api/history?limit=50")
    #expect(backend.clearHistoryPath(keeping: 5) == "/api/history?keep=5")
    #expect(backend.clearHistoryPath(keeping: nil) == "/api/history")
}

/// `.` and `:` are both in `.urlPathAllowed`, so `escaped(_:)` leaves them
/// alone -- `models.flux-dev:q8.default_steps` must address ONE path
/// component, which is what axum's `Path<String>` reads back.
@Test func aConfigKeyWithADotAndAColonIsOnePathComponent() {
    let key = "models.flux-dev:q8.default_steps"
    #expect(backend.escaped(key) == key)
    let url = backend.request("/api/config/\(backend.escaped(key))", method: "PUT").url
    #expect(url?.path() == "/api/config/models.flux-dev:q8.default_steps")
}

/// `GET /api/loras?model=<name>` is the whole compatibility decision --
/// `catalog_api.rs:1098-1115` resolves the family and filters ON THE SERVER,
/// so the model name is the only thing this URL carries.
@Test func anAdapterListIsAskedForOneModel() {
    let url = backend.request(backend.loraPath(model: "z-image-turbo:q8")).url
    #expect(url?.absoluteString == "http://workstation:7680/api/loras?model=z-image-turbo:q8")
}

/// A print in the trash lives behind `?view=trash`, exactly as the listing
/// does. Fetching it from the live view answers 404 on a print that is right
/// there.
@Test func aTrashedPrintIsFetchedFromTheTrashView() {
    let live = backend.mediaRequest("a b.png", trashed: false).url
    #expect(live?.path() == "/api/gallery/image/a%20b.png")
    #expect(live?.query() == nil)

    let trashed = backend.mediaRequest("a b.png", trashed: true).url
    #expect(trashed?.path() == "/api/gallery/image/a%20b.png")
    #expect(trashed?.query() == "view=trash")
}

/// M5 S1b: delete addresses a fixed `:model` segment (escaped, since it is
/// ONE path component); load and unload carry the model in the BODY instead
/// (`routes.rs:5324-5338`, `:5651-5660`); components and downloads are plain
/// GETs; a catalog install addresses the WILDCARD `/api/catalog/*id` route,
/// where a literal `/` in an `hf:owner/repo` id must survive rather than
/// being escaped away; and a search's query string omits what was not asked.
@Test func theModelRoutesAddressTheRightPaths() throws {
    let deleteURL = backend.request(backend.modelPath("flux-dev:q4"), method: "DELETE").url
    #expect(deleteURL?.path() == "/api/models/flux-dev:q4")

    let componentsURL = backend.request(backend.modelComponentsPath("flux-schnell:q8")).url
    #expect(componentsURL?.path() == "/api/models/flux-schnell:q8/components")

    let downloadsURL = backend.request("/api/downloads").url
    #expect(downloadsURL?.path() == "/api/downloads")

    let loadBody = try MoldJSON.encoder.encode(LoadModelWireBody(model: "flux-dev:q4", gpu: 1))
    let loadObject = try #require(JSONSerialization.jsonObject(with: loadBody) as? [String: Any])
    #expect(loadObject["model"] as? String == "flux-dev:q4")
    #expect(loadObject["gpu"] as? Int == 1)

    let unloadBody = try MoldJSON.encoder.encode(UnloadModelWireBody(model: nil, gpu: nil))
    let unloadObject = try #require(JSONSerialization.jsonObject(with: unloadBody) as? [String: Any])
    #expect(unloadObject["model"] == nil)
    #expect(unloadObject["gpu"] == nil)

    // A literal `/` inside a catalog id must survive: the wildcard route is
    // built to split on it, not to have it protected as one component.
    let installURL = backend.request(backend.catalogDownloadPath("hf:owner/repo")).url
    #expect(installURL?.path() == "/api/catalog/hf:owner/repo/download")
    let cvURL = backend.request(backend.catalogDownloadPath("cv:252914")).url
    #expect(cvURL?.path() == "/api/catalog/cv:252914/download")

    let query = CatalogQuery(text: "dreamshaper", pageSize: 3)
    #expect(query.queryString == "q=dreamshaper&page_size=3")
}

/// M6 S1b: the three transfer routes (design fact 4 --
/// `routes.rs:7548-7599`, `:2973-2994`).
@Test func theTransferRoutesAddressTheRightPaths() {
    #expect(backend.transferExportPath("job 1") == "/api/queue/job%201/transfer")
    #expect(backend.transferCompletePath("job 1") == "/api/queue/job%201/transfer/complete")
    #expect(HTTPBackend.transferAdmitPath == "/api/generation-batches/transfer")

    let exportURL = backend.request(backend.transferExportPath("j")).url
    #expect(exportURL?.path() == "/api/queue/j/transfer")
    let completeURL = backend.request(backend.transferCompletePath("j")).url
    #expect(completeURL?.path() == "/api/queue/j/transfer/complete")
}

/// **Fails today**: there is no splice, and encoding a portable body through
/// `BatchAdmission` would drop any key this build's `GenerateRequest` does
/// not model -- because its hand-written `encode(to:)` enumerates a fixed
/// field list. `admissionBody` must never parse `portable`; a key it does
/// not understand has to survive byte for byte (design fact 14).
@Test func anAdmissionBodyCarriesTheExportVerbatim() throws {
    let portable = Data(
        #"{"prompt":"a cat","a_field_this_build_does_not_model":{"nested":[1,2,3]}}"#.utf8)
    let body = try HTTPBackend.transferAdmissionBody(clientBatchId: "abc-123", portable: portable)
    let text = try #require(String(data: body, encoding: .utf8))
    #expect(text == #"{"client_batch_id":"abc-123","requests":[{"prompt":"a cat","a_field_this_build_does_not_model":{"nested":[1,2,3]}}]}"#)

    // And it still parses as one JSON object with `requests` holding the
    // portable bytes UNCHANGED, including the unknown key.
    let object = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
    #expect(object["client_batch_id"] as? String == "abc-123")
    let requests = try #require(object["requests"] as? [[String: Any]])
    #expect(requests.count == 1)
    #expect(requests[0]["prompt"] as? String == "a cat")
    #expect(requests[0]["a_field_this_build_does_not_model"] != nil)
}

/// A client batch id can itself hold characters JSON has to escape -- this
/// pins that it goes through `MoldJSON.encoder`'s escaping rather than being
/// interpolated raw, which would produce broken JSON or, worse, a body an
/// attacker-controlled id could inject fields into.
@Test func anAdmissionBodyEscapesTheClientBatchId() throws {
    let body = try HTTPBackend.transferAdmissionBody(
        clientBatchId: "a\"b\\c", portable: Data("{}".utf8))
    let object = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
    #expect(object["client_batch_id"] as? String == "a\"b\\c")
}

/// `x-mold-destination-instance` is the fence against a destination that
/// restarted between the picker and the click; the destination admits only
/// when it still recognises itself (`routes.rs:2973-2994`).
@Test func theDestinationHeaderRidesTheAdmission() throws {
    let request = try backend.transferAdmissionRequest(
        clientBatchId: "abc", portable: Data("{}".utf8), destinationInstance: "inst-42")
    #expect(request.value(forHTTPHeaderField: "x-mold-destination-instance") == "inst-42")
    #expect(request.httpMethod == "POST")
    #expect(request.url?.path() == "/api/generation-batches/transfer")
    #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
}

/// M7 S1, test 12: a paired client's id rides as ONE path component, the
/// same rule the catalog id bug taught.
@Test func revokingAClientEscapesItsId() {
    let url = backend.request("/api/pairing/clients/\(backend.escaped("client one"))", method: "DELETE").url
    #expect(url?.path() == "/api/pairing/clients/client%20one")
}
