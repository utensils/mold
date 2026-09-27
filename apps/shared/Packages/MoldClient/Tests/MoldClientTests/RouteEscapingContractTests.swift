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

/// Every `\(…)` inside a route literal, as written, and WHERE it sits.
///
/// The half that matters: a path component and a query VALUE take different
/// escapers, and using the path one on a query value is finding 01#6 --
/// `.urlPathAllowed` includes `&`, `=` and `+`, so the host silently parses a
/// different request. An interpolation after the literal's first `?` is a
/// query value.
private func interpolations(in literal: String) -> [(text: String, inQuery: Bool)] {
    let queryStart = literal.firstIndex(of: "?")
    var found: [(text: String, inQuery: Bool)] = []
    var rest = Substring(literal)
    while let open = rest.range(of: "\\(") {
        var depth = 1
        var index = open.upperBound
        while index < rest.endIndex, depth > 0 {
            if rest[index] == "(" { depth += 1 }
            if rest[index] == ")" { depth -= 1 }
            if depth > 0 { index = rest.index(after: index) }
        }
        let inQuery = queryStart.map { open.lowerBound > $0 } ?? false
        found.append((String(rest[open.upperBound..<index]), inQuery))
        rest = rest[rest.index(after: index)...]
    }
    return found
}

/// **Fails today**: nine routes in `HTTPBackend+Work.swift` and
/// `HTTPBackend+Generation.swift` interpolate an id straight into the path,
/// and `Catalog.queryString` / `loraPath` escape a QUERY value with the PATH
/// escaper (01#6). Accepting either escaper anywhere would let the second of
/// those back in unnoticed, so the position decides which one is required.
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
            for (expression, inQuery) in interpolations(in: literal) {
                // A query value takes `escapedQueryValue` and NOTHING else --
                // `escaped` there is the bug, not a lesser form of right.
                let escaped = inQuery
                    ? expression.contains("escapedQueryValue(")
                    : expression.contains("escaped(")
                let excused = "\(file.lastPathComponent): \(expression)"
                guard !escaped, !deliberatelyRaw.contains(excused) else { continue }
                unescaped.append(
                    "\(excused)  in  \(literal)  (\(inQuery ? "query value" : "path component"))")
            }
        }
    }
    #expect(unescaped == [])
}
