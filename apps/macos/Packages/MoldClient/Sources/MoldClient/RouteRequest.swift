import Foundation

/// One host's route, as a `URLRequest`: where the path is joined on, which
/// key rides with it, and how long it may take by default.
enum RouteRequest {
    /// The idle timeout every route gets unless it asks for another -- a JSON
    /// answer's allowance, not a clip's or a model's.
    static let defaultTimeout: TimeInterval = 10

    /// A request for a path, which may carry a query string.
    ///
    /// The path is taken as ALREADY PERCENT-ENCODED — interpolate a dynamic
    /// component through `RouteEscaping.escaped(_:)`. NOT
    /// `baseURL.appending(path:)`, which treats its whole argument as one
    /// component and encodes everything in it, `?` included:
    /// `/api/gallery?view=trash` became `/api/gallery%3Fview=trash`, a route
    /// no mold has, and Recently Deleted listed nothing on a machine holding
    /// 177 prints — silently, because the app asks conditionally and a shrug
    /// looks exactly like "nothing changed".
    static func build(_ path: String, for host: MoldHost) -> URLRequest {
        let parts = path.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
        var components = URLComponents(url: host.baseURL, resolvingAgainstBaseURL: false)
        // A host behind a reverse proxy keeps the prefix in its base URL.
        var prefix = components?.percentEncodedPath ?? ""
        if prefix.hasSuffix("/") { prefix.removeLast() }
        components?.percentEncodedPath = prefix + String(parts[0])
        if parts.count == 2 { components?.percentEncodedQuery = String(parts[1]) }
        guard let url = components?.url else {
            return URLRequest(url: host.baseURL.appending(path: path))
        }
        var request = URLRequest(url: url)
        // A keyless host is open by policy. Sending no key is the correct
        // request there, not a degraded one.
        if let key = host.apiKey, !key.isEmpty {
            request.setValue(key, forHTTPHeaderField: "X-Api-Key")
        }
        request.timeoutInterval = defaultTimeout
        return request
    }
}
