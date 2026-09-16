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
