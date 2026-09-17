import Foundation
import Testing

@testable import MoldClient

// A per-site test cannot cover this one: the failure mode is a `fatalError`
// inside `URLComponents.percentEncodedPath`'s setter (it RAISES on an invalid
// character rather than answering nil), so a test that reached it would take
// the runner down with it. What is checkable is the RULE -- every dynamic
// component in a route string goes through an escaper -- read off the source
// the same way `ModelFamilyContractTests` reads the Rust it is pinned to.

/// Interpolations that are deliberately NOT escaped, named by FILE and
/// expression, each with its reason. A new one has to be justified here
/// rather than by omission -- `id` alone would excuse every route there is.
private let deliberatelyRaw: Set<String> = [
    // `/api/catalog/*id` is a WILDCARD route: a literal `/` inside an
    // `hf:owner/repo` id must survive rather than be protected as one
    // component (the call site says so too).
    "HTTPBackend+Catalog.swift: id",
    // A query string this same type already built, parameter by parameter.
    "HTTPBackend+Catalog.swift: qs",
    // `MediaURL` builds through `appending(path:)`, which encodes the segment
    // itself -- pre-encoding would escape the escapes and ask for `a%20b`.
    "MediaURL.swift: filename",
    // Fixed tokens and integers, not names: `view=trash`, `limit=50`.
    "HTTPBackend.swift: $0",
    "HTTPBackend+Create.swift: limit",
    "HTTPBackend+Create.swift: $0",
]

private func routeLiterals(in source: String) -> [String] {
    source.matches(of: /"(\/api\/[^"]*)"/).map { String($0.1) }
}

/// Every `\(…)` inside a route literal, as written.
private func interpolations(in literal: String) -> [String] {
    var found: [String] = []
    var rest = Substring(literal)
    while let open = rest.range(of: "\\(") {
        var depth = 1
        var index = open.upperBound
        while index < rest.endIndex, depth > 0 {
            if rest[index] == "(" { depth += 1 }
            if rest[index] == ")" { depth -= 1 }
            if depth > 0 { index = rest.index(after: index) }
        }
        found.append(String(rest[open.upperBound..<index]))
        rest = rest[rest.index(after: index)...]
    }
    return found
}

/// **Fails today**: nine routes in `HTTPBackend+Work.swift` and
/// `HTTPBackend+Generation.swift` interpolate an id straight into the path.
/// Every one carries a UUID today so none of it is reachable -- but
/// `URLComponents.percentEncodedPath` raises rather than returning nil, so an
/// id shape that ever grows a space or a `?` crashes the app instead of
/// failing one request. Every comparable route already escapes.
@Test func everyDynamicRouteComponentGoesThroughAnEscaper() throws {
    let sources = RepoFixtures.testDirectory
        .deletingLastPathComponent()  // Tests/
        .deletingLastPathComponent()  // the package root
        .appending(path: "Sources/MoldClient")
    let files = try FileManager.default.contentsOfDirectory(at: sources, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "swift" }
    #expect(files.count > 50, "the package source directory was not found")

    var unescaped: [String] = []
    for file in files {
        let source = try String(contentsOf: file, encoding: .utf8)
        for literal in routeLiterals(in: source) {
            for expression in interpolations(in: literal) {
                let escaped = expression.contains("escaped(")
                    || expression.contains("escapedQueryValue(")
                let excused = "\(file.lastPathComponent): \(expression)"
                guard !escaped, !deliberatelyRaw.contains(excused) else { continue }
                unescaped.append("\(excused)  in  \(literal)")
            }
        }
    }
    #expect(unescaped == [])
}
