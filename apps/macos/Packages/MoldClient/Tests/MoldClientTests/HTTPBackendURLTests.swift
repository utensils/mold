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
    host: MoldHost(name: "plato", baseURL: URL(string: "http://plato:7680")!)
)

@Test func aQueryStringSurvivesBeingTurnedIntoARequest() {
    let url = backend.request("/api/gallery?view=trash").url
    #expect(url?.absoluteString == "http://plato:7680/api/gallery?view=trash")
    #expect(url?.query() == "view=trash")
}

@Test func severalParametersAllSurvive() {
    let url = backend.request("/api/gallery/thumbnail/a.png?size=512&fmt=jpeg").url
    #expect(url?.absoluteString
            == "http://plato:7680/api/gallery/thumbnail/a.png?size=512&fmt=jpeg")
}

@Test func aPlainPathIsUntouched() {
    #expect(backend.request("/api/status").url?.absoluteString == "http://plato:7680/api/status")
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
        host: MoldHost(name: "plato", baseURL: URL(string: "http://plato:7680")!)
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
    }
    #expect(backend.host.name == "plato")
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
