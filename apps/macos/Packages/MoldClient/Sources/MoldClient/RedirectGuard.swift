import Foundation

/// Keeps the API key from following a redirect off the machine it was for.
///
/// `URLSession` forwards custom headers across a redirect, cross-origin
/// included, and there was no `URLSessionTaskDelegate` anywhere in the app to
/// say otherwise. mold's default scheme is plain `http`, so a compromised or
/// misconfigured proxy in front of a host could harvest the operator key with
/// a single 302 -- and the key is the whole of a keyed host's security.
///
/// Attached per TASK rather than per session, so a caller's own `URLSession`
/// (a stub in tests, a configured one in the app) keeps whatever delegate it
/// already has.
final class RedirectGuard: NSObject, URLSessionTaskDelegate, Sendable {
    /// The origin the credential belongs to. Anywhere else is somewhere else,
    /// however the hostname reads.
    let origin: URL

    init(origin: URL) {
        self.origin = origin
        super.init()
    }

    /// `newRequest` with the key removed if it leaves `origin`.
    ///
    /// Returning the request rather than `nil` means the redirect is still
    /// FOLLOWED -- a host that answers a 302 is not doing anything wrong, and
    /// refusing it here would break a reverse proxy adding a trailing slash.
    /// It is only the credential that stops at the boundary.
    func sanitized(_ newRequest: URLRequest) -> URLRequest {
        guard newRequest.value(forHTTPHeaderField: Self.keyHeader) != nil else {
            return newRequest
        }
        guard let destination = newRequest.url,
              HostAddress.sameOrigin(origin, destination)
        else {
            var stripped = newRequest
            stripped.setValue(nil, forHTTPHeaderField: Self.keyHeader)
            return stripped
        }
        return newRequest
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest
    ) async -> URLRequest? {
        sanitized(request)
    }

    static let keyHeader = "X-Api-Key"
}
