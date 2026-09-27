import Foundation
import Testing

@testable import MoldClient

// A query VALUE is escaped by a different rule than a path component, and
// mixing them up is silent: the host parses a different request than the one
// the person typed and answers it without complaint.

private let backend = HTTPBackend(
    host: MoldHost(name: "workstation", baseURL: URL(string: "http://workstation:7680")!)
)

/// Reads a query string back the way `serde_urlencoded` does -- split on `&`,
/// then on the first `=`, `+` is a space, `%XX` is a byte. This is the oracle:
/// what matters is not the spelling on the wire but what the HOST ends up
/// with.
private func decodeQuery(_ query: String) -> [(String, String)] {
    query.split(separator: "&", omittingEmptySubsequences: false).map { pair in
        let halves = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
        func decode(_ part: Substring) -> String {
            String(part).replacingOccurrences(of: "+", with: " ")
                .removingPercentEncoding ?? String(part)
        }
        return (decode(halves[0]), halves.count == 2 ? decode(halves[1]) : "")
    }
}

/// **Fails today**: `queryString` escapes with `escaped(_:)`, which is the
/// PATH rule -- `.urlPathAllowed` INCLUDES `&`, `=`, `+`, `;`, `$` and `,`.
/// Searching Discover for `cats & dogs` sends `q=cats%20&%20dogs`, which the
/// host reads as `q` = `"cats "` plus an unrelated empty parameter, and
/// answers for the wrong query with no error anywhere.
@Test func aCatalogSearchForTwoThingsAsksForTwoThings() {
    let query = CatalogQuery(text: "cats & dogs")
    let decoded = decodeQuery(query.queryString)
    #expect(decoded.count == 1)
    #expect(decoded[0].0 == "q")
    #expect(decoded[0].1 == "cats & dogs")
}

/// `+` is the other one, and it is worse because nothing looks wrong: a
/// literal `+` in a query value is a SPACE to `serde_urlencoded`, so `C++`
/// arrives as `C  `.
@Test func aSearchForAPlusKeepsItsPlus() {
    #expect(decodeQuery(CatalogQuery(text: "C++").queryString)[0].1 == "C++")
}

/// Every awkward character a person can type, through the same oracle.
@Test func aSearchValueSurvivesWhateverIsInIt() {
    for text in ["a=b", "a#b", "a;b", "100% wool", "a,b$c", "café", "a/b:c", "a b"] {
        let decoded = decodeQuery(CatalogQuery(text: text).queryString)
        #expect(decoded.count == 1, "\(text)")
        #expect(decoded[0].1 == text, "\(text)")
    }
}

/// The other parameters take the same rule -- they are server-defined tokens
/// today, but the rule is about the encoder, not about who supplies the value.
@Test func everyCatalogParameterIsEscapedTheSameWay() {
    let query = CatalogQuery(
        text: "a&b", family: "f&f", kind: "k&k", source: "s&s", sort: "o&o",
        page: 2, pageSize: 3, includeNSFW: true)
    let decoded = Dictionary(uniqueKeysWithValues: decodeQuery(query.queryString))
    #expect(decoded["q"] == "a&b")
    #expect(decoded["family"] == "f&f")
    #expect(decoded["kind"] == "k&k")
    #expect(decoded["source"] == "s&s")
    #expect(decoded["sort"] == "o&o")
    #expect(decoded["page"] == "2")
    #expect(decoded["page_size"] == "3")
    #expect(decoded["include_nsfw"] == "true")
}

/// `GET /api/loras?model=<name>` carries a model name, and a model name is
/// free enough to hold a `+`.
@Test func anAdapterListEscapesItsModelName() {
    let path = backend.loraPath(model: "some+model:q8")
    let query = try? #require(backend.request(path).url?.query(percentEncoded: true))
    #expect(decodeQuery(query ?? "")[0].1 == "some+model:q8")
}

/// The ordinary case is still readable on the wire -- this is an escaper, not
/// a blanket re-encoding, and a `:` in a model tag stays a `:`.
@Test func anOrdinaryValueIsLeftAlone() {
    #expect(CatalogQuery(text: "dreamshaper", pageSize: 3).queryString
            == "q=dreamshaper&page_size=3")
    #expect(backend.loraPath(model: "z-image-turbo:q8")
            == "/api/loras?model=z-image-turbo:q8")
}

/// A path component keeps the PATH rule: `escaped(_:)` is not the query
/// escaper and must not become it, or every config key and device id in the
/// package changes shape.
@Test func thePathEscaperIsUnchanged() {
    #expect(backend.escaped("models.flux-dev:q8.default_steps")
            == "models.flux-dev:q8.default_steps")
    #expect(backend.escaped("a b~c#d.png") == "a%20b~c%23d.png")
}
