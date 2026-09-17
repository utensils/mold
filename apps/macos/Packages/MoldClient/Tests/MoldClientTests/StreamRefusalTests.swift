import Foundation
import Testing

@testable import MoldClient

// A refusal on an event-stream route is the same refusal as on any other
// route, and has to read the same way. `StreamTests` owns the happy paths and
// the 401; this suite is about the body that comes with a refusal.

/// Its own table: `StreamTests` plants a 401 on `/api/events` and this suite
/// plants a 503 on the same route, and swift-testing does not serialize
/// BETWEEN suites (`StubTransport`).
private final class RefusalTransport: StubTransport {
    nonisolated(unsafe) static var responses: [String: (status: Int, body: Data)] = [:]
    override class func response(for path: String) -> (status: Int, body: Data)? {
        responses[path]
    }
}

private func stubbed() -> HTTPBackend { RefusalTransport.backend() }

@Suite(.serialized)
struct StreamRefusalTests {

    /// **Fails today**: `stream` drops `bytes` unconsumed on a non-2xx and
    /// `streamFailure` builds the error from the STATUS CODE alone. A host
    /// refusing the event stream with `503 SERVER_RESTARTING` -- which names
    /// what is happening and is a wait, not a fault -- surfaced as "The
    /// machine answered with an error (503)", and the reconnect loop then
    /// retried it forever with nothing to say. The identical refusal on a
    /// plain GET goes through `check(_:_:)` and reads correctly.
    @Test func aRefusedStreamKeepsTheMachinesOwnSentence() async {
        let body = #"{"error":"This machine is restarting. Try again shortly.","code":"SERVER_RESTARTING"}"#
        RefusalTransport.responses["/api/events"] = (503, Data(body.utf8))
        let backend = stubbed()

        await #expect {
            for try await _ in backend.events() {}
        } throws: { error in
            guard case let MoldClientError.http(status, code, message) = error else { return false }
            return status == 503 && code == "SERVER_RESTARTING"
                && message == "This machine is restarting. Try again shortly."
        }
    }

    /// A licence refusal is the one refusal the app can RESOLVE, and the
    /// payload it needs rides in the body -- so a stream route that meets one
    /// has to hand back the same `.licenseRequired` a download would.
    @Test func aStreamCanRefuseForALicence() async {
        let refusal = #"{"id":"h3","name":"Hunyuan3D 2.1","url":"https://x/l","canonical":"c","sha256":"ab","summary":"s"}"#
        let body = #"{"error":"terms","code":"LICENSE_NOT_ACCEPTED","license":\#(refusal)}"#
        RefusalTransport.responses["/api/downloads/stream"] = (403, Data(body.utf8))
        let backend = stubbed()

        await #expect {
            for try await _ in backend.downloadEvents() {}
        } throws: { error in
            guard case let MoldClientError.licenseRequired(refusal, mismatch) = error
            else { return false }
            return refusal.name == "Hunyuan3D 2.1" && !mismatch
        }
    }

    /// A refusal with no body at all still reads as that status, and a 401
    /// is still `.unauthorized` -- the body is additional evidence, never the
    /// thing the answer depends on.
    @Test func aRefusalWithNoBodyStillNamesItsStatus() async {
        RefusalTransport.responses["/api/resources/stream"] = (500, Data())
        let backend = stubbed()

        await #expect {
            for try await _ in backend.resourceStream() {}
        } throws: { error in
            guard case let MoldClientError.http(status, code, message) = error else { return false }
            return status == 500 && code == nil && message == nil
        }
    }

    /// The body is read under a ceiling: nothing about a non-2xx promises the
    /// other side closes the connection, and a refusal body is a small JSON
    /// object. A megabyte of it must not become a megabyte in a message.
    @Test func aRefusalBodyIsReadUnderACeiling() async {
        let huge = String(repeating: "x", count: 4 * 1024 * 1024)
        RefusalTransport.responses["/api/events"] = (500, Data(huge.utf8))
        let backend = stubbed()

        await #expect {
            for try await _ in backend.events() {}
        } throws: { error in
            guard case let MoldClientError.http(_, _, message) = error else { return false }
            // `plainMessage` refuses anything over 400 bytes as "not a
            // sentence", so a flood reports the status and nothing else.
            return message == nil
        }
    }
}
