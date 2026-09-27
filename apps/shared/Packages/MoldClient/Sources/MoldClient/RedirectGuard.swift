import Foundation

/// Keeps the API key from following a redirect off the machine it was for.
///
/// `URLSession` forwards custom headers across a redirect, cross-origin
/// included, and there was no `URLSessionTaskDelegate` anywhere in the app to
/// say otherwise. mold's default scheme is plain `http`, so a compromised or
/// misconfigured proxy in front of a host could harvest the operator key with
/// a single 302 -- and the key is the whole of a keyed host's security.
///
/// **Every request that sets `X-Api-Key` must carry one.** `HTTPBackend` is
/// not the only place in the app that does: anything holding its own
/// `URLSession` and setting the header itself needs the same guard, and the
/// call is one argument --
///
/// ```swift
/// let (data, response) = try await session.data(
///     for: request, delegate: RedirectGuard(origin: host.baseURL))
/// ```
///
/// -- which is why this is public rather than an internal detail of the
/// transport. It is attached per TASK rather than per session, so a caller's
/// own `URLSession` keeps whatever delegate it already has, and a fresh
/// instance per call is correct: it holds one immutable `URL`.
public final class RedirectGuard: NSObject, URLSessionTaskDelegate, Sendable {
    /// The origin the credential belongs to. Anywhere else is somewhere else,
    /// however the hostname reads.
    public let origin: URL

    /// - Parameter origin: the host the key is for -- `MoldHost.baseURL`. Its
    ///   path is ignored; only scheme, host and port decide.
    public init(origin: URL) {
        self.origin = origin
        super.init()
    }

    /// `newRequest` with the key removed if it leaves `origin`.
    ///
    /// Returning the request rather than `nil` means the redirect is still
    /// FOLLOWED -- a host that answers a 302 is not doing anything wrong, and
    /// refusing it here would break a reverse proxy adding a trailing slash.
    /// It is only the credential that stops at the boundary.
    public func sanitized(_ newRequest: URLRequest) -> URLRequest {
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

    public func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest
    ) async -> URLRequest? {
        sanitized(request)
    }

    /// The header this guard is about. Public so a caller setting it by hand
    /// spells it the same way.
    public static let keyHeader = "X-Api-Key"
}
