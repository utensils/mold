import Foundation
import Testing

// NOT `@testable`: the point of this suite is that the guard is reachable by
// every caller that sets `X-Api-Key`, and `HTTPBackend` is not the only one.
// `Sources/Mold/Library/ThumbnailCache.swift` holds its own `URLSession` and
// sets the header itself -- the highest-volume request path in the app -- so
// a guard that only the transport could reach was half a fix.
import MoldClient

// `URLSession` forwards custom headers across a redirect, cross-origin
// included, and mold's default scheme is plain `http`. A compromised or
// misconfigured proxy in front of a host can therefore harvest the operator
// key with a single 302.

private let origin = URL(string: "http://workstation:7680")!

private func redirected(to target: String) -> URLRequest? {
    var request = URLRequest(url: URL(string: target)!)
    request.setValue("secret", forHTTPHeaderField: "X-Api-Key")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    return RedirectGuard(origin: origin).sanitized(request)
}

/// **Fails today**: there is no delegate anywhere in the app -- `grep` for
/// `willPerformHTTPRedirection` finds nothing -- so `X-Api-Key` rides
/// whatever the 302 names.
@Test func theKeyDoesNotFollowARedirectOffTheMachine() throws {
    let request = try #require(redirected(to: "http://elsewhere:7680/api/status"))
    #expect(request.value(forHTTPHeaderField: "X-Api-Key") == nil)
    // Only the credential is dropped. The redirect itself is still followed,
    // and everything else about the request is the session's business.
    #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
}

/// A different SCHEME or PORT is a different origin, whatever the hostname
/// says -- `https://workstation` is not `http://workstation:7680`.
@Test func aDifferentSchemeOrPortIsADifferentOrigin() throws {
    for target in ["https://workstation:7680/api/status", "http://workstation:9999/api/status",
                   "http://workstation/api/status"] {
        let request = try #require(redirected(to: target))
        #expect(request.value(forHTTPHeaderField: "X-Api-Key") == nil, "\(target)")
    }
}

/// A redirect that stays on the machine is ordinary -- a reverse proxy adding
/// a trailing slash, a host redirecting `/api/x` to `/api/x/` -- and dropping
/// the key there would break every keyed host behind one.
@Test func aRedirectWithinTheMachineKeepsTheKey() throws {
    for target in ["http://workstation:7680/api/status/", "http://WORKSTATION:7680/api/status"] {
        let request = try #require(redirected(to: target))
        #expect(request.value(forHTTPHeaderField: "X-Api-Key") == "secret", "\(target)")
    }
}

/// A request with no key needs no sanitizing, and saying so keeps the
/// delegate off the hot path of every keyless host.
@Test func aRequestWithNoKeyIsHandedBackUntouched() throws {
    var request = URLRequest(url: URL(string: "http://elsewhere/api/status")!)
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    let sanitized = try #require(RedirectGuard(origin: origin).sanitized(request))
    #expect(sanitized == request)
}

/// A destination that will not parse as an address is not this machine, so
/// the key comes off. Failing open on an unparseable URL is the one direction
/// this must never fail.
@Test func anUnreadableDestinationLosesTheKey() throws {
    var request = URLRequest(url: URL(string: "about:blank")!)
    request.setValue("secret", forHTTPHeaderField: "X-Api-Key")
    let sanitized = try #require(RedirectGuard(origin: origin).sanitized(request))
    #expect(sanitized.value(forHTTPHeaderField: "X-Api-Key") == nil)
}

/// **Fails today**: the guard is internal, so a caller outside MoldClient
/// cannot attach one -- and `ThumbnailCache` is exactly such a caller. This
/// is the whole call-site form, compiled from outside the module: one
/// argument on the request the caller already makes.
@Test func anyCallerThatSetsTheKeyCanAttachTheGuard() async throws {
    let host = MoldHost(name: "workstation", baseURL: origin, apiKey: "secret")
    var request = URLRequest(url: origin.appending(path: "/api/gallery/thumbnail/a.png"))
    request.setValue(host.apiKey, forHTTPHeaderField: RedirectGuard.keyHeader)

    // Built and not sent: what is under test is that it COMPILES against the
    // public surface, the same precedent as
    // `aBackendHeldAsTheProtocolReachesEveryRouteTheAppUses`.
    let send: (URLSession) async throws -> Void = { session in
        _ = try await session.data(
            for: request, delegate: RedirectGuard(origin: host.baseURL))
    }
    _ = send

    let guarded = RedirectGuard(origin: host.baseURL)
    #expect(guarded.origin == origin)
    var offMachine = request
    offMachine.url = URL(string: "http://elsewhere/api/gallery/thumbnail/a.png")
    #expect(guarded.sanitized(offMachine).value(forHTTPHeaderField: RedirectGuard.keyHeader) == nil)
}
