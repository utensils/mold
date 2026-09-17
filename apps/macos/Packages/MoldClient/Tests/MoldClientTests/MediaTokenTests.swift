import Foundation
import Testing

@testable import MoldClient

// The URL handed to a player. `AVPlayer` builds its own requests and cannot
// set `X-Api-Key`, so on a keyed host the only credential it can carry is the
// ticket in the query string -- which makes "no usable ticket" an auth
// failure, not a URL to try anyway.

private final class TicketTransport: StubTransport {
    nonisolated(unsafe) static var responses: [String: (status: Int, body: Data)] = [:]
    override class func response(for path: String) -> (status: Int, body: Data)? {
        responses[path]
    }
    override class var contentType: String { "application/json" }
}

private let tokenPath = "/api/gallery/media-token"

private func plant(_ json: String, status: Int = 200) {
    TicketTransport.responses[tokenPath] = (status, Data(json.utf8))
}

@Suite(.serialized)
struct MediaTokenTests {

    /// A keyless host needs no ticket, and the plain URL is the correct
    /// request there rather than a degraded one -- so nothing is minted at
    /// all. (Nothing is planted either: an unplanted route fails the stub,
    /// which is what proves no request was made.)
    @Test func aKeylessHostGetsThePlainURL() async throws {
        TicketTransport.responses = [:]
        let backend = TicketTransport.backend(apiKey: nil)
        let url = try await backend.playableURL(for: "a b.png")
        #expect(url.path() == "/api/gallery/image/a%20b.png")
        #expect(url.query() == nil)
    }

    /// The ordinary keyed case: the ticket rides in the query, signed over
    /// the same encoded path the player will actually request.
    @Test func aKeyedHostGetsATicketedURL() async throws {
        plant(#"{"token":"tok-1","expires_at":1893456000,"auth_required":true}"#)
        let backend = TicketTransport.backend(apiKey: "secret")
        let url = try await backend.playableURL(for: "a b.png")
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        #expect(items.first { $0.name == "media_token" }?.value == "tok-1")
        #expect(items.first { $0.name == "expires" }?.value == "1893456000")
        #expect(url.path() == "/api/gallery/image/a%20b.png")
        // The key itself never rides the URL -- a ticket in a query string is
        // already in every proxy log between here and the machine.
        #expect(!url.absoluteString.contains("secret"))
    }

    /// This Mac holding a stale saved key for a host that has since become
    /// keyless is the case the SERVER answers explicitly -- `auth_required:
    /// false` with no token means "use the ordinary direct URL"
    /// (`routes.rs:9800-9803`). That is not a failure.
    @Test func aStaleKeyAgainstAKeylessHostFallsBackToThePlainURL() async throws {
        plant(#"{"auth_required":false}"#)
        let backend = TicketTransport.backend(apiKey: "stale")
        let url = try await backend.playableURL(for: "a.png")
        #expect(url.query() == nil)
    }

    /// **Fails today**: the `guard` falls through to `return plain` when the
    /// mint answers `auth_required: true` with no token. `AVPlayer` then
    /// requests the media with no credential, gets a 401 it has no way to
    /// report, and shows a silent playback failure instead of an auth error
    /// -- while the doc comment three lines above promises the opposite.
    @Test func aKeyedHostWithNoUsableTicketFailsClosed() async {
        plant(#"{"auth_required":true}"#)
        let backend = TicketTransport.backend(apiKey: "secret")
        await #expect {
            _ = try await backend.playableURL(for: "a.png")
        } throws: { error in
            guard case MoldClientError.unauthorized = error else { return false }
            return true
        }
    }

    /// And the mint's own refusal still reads as itself: a keyed host reached
    /// without the right key answers 401 on the token route.
    @Test func aRefusedMintStaysAnAuthFailure() async {
        plant(#"{"error":"nope","code":"UNAUTHORIZED"}"#, status: 401)
        let backend = TicketTransport.backend(apiKey: "wrong")
        await #expect {
            _ = try await backend.playableURL(for: "a.png")
        } throws: { error in
            guard case MoldClientError.unauthorized = error else { return false }
            return true
        }
    }
}
