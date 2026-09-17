import Foundation

/// A route as a log line may name it.
///
/// `MoldLog`'s rule is that nothing a person owns is ever written down, and a
/// mold route is made of exactly those things: `/api/gallery/image/<filename>`,
/// `/api/gallery/tags/<a tag they typed>`, `/api/queue/<job id>`,
/// `/api/models/<model>`, `/api/catalog/search?q=<what they searched for>`.
/// Logging the path as handed broke the rule the file above it states.
///
/// The rule here is the one that needs no table and cannot rot: keep
/// `/api/<family>` and redact everything after it. The first component after
/// `/api` is fixed in every route this package builds -- pinned by
/// `theFirstComponentAfterApiIsNeverDynamic`, which reads the source -- and
/// the SECOND is not (`/api/queue/<id>`), so exactly one is kept. Paired with
/// the decoded TYPE, which the same log line carries, that names the route in
/// every case that matters without naming anything in it.
///
/// A path that is not a mold route at all is redacted whole. Failing closed
/// is the only direction this is allowed to fail in.
enum RouteTemplate {
    static func redacted(_ path: String) -> String {
        // A query is dropped entire: `q=` is free user input, and no other
        // parameter is worth the risk of getting the rule wrong once.
        let withoutQuery = path.prefix { $0 != "?" }
        var components = withoutQuery.split(separator: "/", omittingEmptySubsequences: true)
        // A host behind a reverse proxy carries the prefix in its base URL.
        guard let api = components.firstIndex(of: "api") else { return "…" }
        components = Array(components[api...])
        guard components.count > 1 else { return "/api" }
        // Concatenated rather than interpolated into a `"/api/…"` literal:
        // the two source contract tests read exactly that shape looking for a
        // dynamic route component, and this is log text, not a request.
        let family = "/api/" + components[1]
        return components.count == 2 ? family : family + "/…"
    }
}
