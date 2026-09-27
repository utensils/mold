import Foundation
import Testing

@testable import MoldClient

// What a log line is allowed to say about WHICH request failed.
//
// `MoldLog` states the rule -- not a filename, not a prompt, not an id -- and
// the decoding and transport lines were breaking it by logging the path they
// were handed.

/// **Fails today**: `HTTPBackend+Decoding.swift` logs the concrete route as
/// `.public`, and mold's routes carry filenames, job ids, collection ids,
/// tag names and model names as path components -- and a search's query
/// carries the words a person typed.
@Test func aLoggedRouteNamesTheEndpointAndNothingInIt() {
    #expect(RouteTemplate.redacted("/api/gallery/image/a%20cat%20asleep.png")
            == "/api/gallery/…")
    #expect(RouteTemplate.redacted("/api/queue/9ffc81c5-3944-6490-bfd9-f68366f98226/pause")
            == "/api/queue/…")
    #expect(RouteTemplate.redacted("/api/models/flux-dev:q4") == "/api/models/…")
    #expect(RouteTemplate.redacted("/api/gallery/tags/holiday%20photos") == "/api/gallery/…")
}

/// A query is dropped whole. `q=` is free user input, and `page_size=3` is
/// not worth the risk of getting the rule wrong once.
@Test func aQueryStringIsDroppedWhole() {
    #expect(RouteTemplate.redacted("/api/catalog/search?q=cats%20%26%20dogs&page=2")
            == "/api/catalog/…")
    #expect(RouteTemplate.redacted("/api/gallery?view=trash") == "/api/gallery")
}

/// A route with nothing after its family is itself: there is nothing in
/// `/api/status` to redact, and redacting it would make the log useless.
@Test func aRouteWithNoDynamicPartIsWrittenWhole() {
    #expect(RouteTemplate.redacted("/api/status") == "/api/status")
    #expect(RouteTemplate.redacted("/api/generation-batches") == "/api/generation-batches")
    #expect(RouteTemplate.redacted("/api/queue") == "/api/queue")
}

/// The first component after `/api` is never interpolated in any route this
/// package builds, which is what makes keeping exactly that one safe --
/// pinned below by `theFirstComponentAfterApiIsNeverDynamic`.
@Test func aHostBehindAReverseProxyKeepsItsPrefixOutOfIt() {
    #expect(RouteTemplate.redacted("/mold/api/gallery/image/a.png") == "/api/gallery/…")
}

/// Anything that is not a mold route is redacted whole rather than guessed
/// at: failing closed is the only direction this can fail in.
@Test func anUnrecognisedPathIsRedactedWhole() {
    #expect(RouteTemplate.redacted("/") == "…")
    #expect(RouteTemplate.redacted("") == "…")
    #expect(RouteTemplate.redacted("/login?token=hunter2") == "…")
}

/// The safety of keeping one component rests on this: NO route literal in the
/// package interpolates into its first two path components (`/api/<family>`).
/// If one ever does, the rule above stops being safe and this says so.
@Test func theFirstComponentAfterApiIsNeverDynamic() throws {
    let sources = RepoFixtures.testDirectory
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appending(path: "Sources/MoldClient")
    let files = try FileManager.default
        .contentsOfDirectory(at: sources, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "swift" }

    var dynamic: [String] = []
    for file in files {
        let source = try String(contentsOf: file, encoding: .utf8)
        for match in source.matches(of: /"(\/api\/[^"]*)"/) {
            let literal = String(match.1)
            // `/api/<family>` is everything up to the second slash after it.
            let family = literal.dropFirst("/api/".count)
                .prefix { $0 != "/" && $0 != "?" }
            if family.contains("\\(") { dynamic.append("\(file.lastPathComponent): \(literal)") }
        }
    }
    #expect(dynamic == [])
}
