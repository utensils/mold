import Foundation
import Testing

@testable import MoldClient

/// `MobilePairingPayload.parse` is a port of `parseMobilePairingPayload`
/// (`studio/api/pairing.ts`): the same two forms, the same refusals. The Mac
/// PRODUCES these codes (`url`); the iPhone reads them -- so the round trip
/// through both halves is the test that matters most.
struct MobilePairingParseTests {
    private let session = PairingSession(
        token: "tok_abc", expiresAt: 1_790_000_000, authRequired: true,
        instanceId: "inst-1", hostname: "workstation")

    @Test func theMacsOwnCodeRoundTrips() throws {
        let made = try #require(MobilePairingPayload(
            session: session, baseURL: URL(string: "http://10.0.0.4:7680")!, name: "workstation"))
        let url = try #require(made.url)
        let read = try MobilePairingPayload.parse(url.absoluteString)
        #expect(read == made)
    }

    @Test func theJSONFormIsAccepted() throws {
        let json = #"{"type":"mold.mobile-pairing","version":1,"base_url":"https://box.ts.net","#
            + #""token":"t","expires_at":42,"instance_id":"i","name":"box"}"#
        let read = try MobilePairingPayload.parse(json)
        #expect(read.baseURL == "https://box.ts.net")
        #expect(read.token == "t")
        #expect(read.expiresAt == 42)
    }

    @Test func aKeylessMachinesCodeHasNoToken() throws {
        let read = try MobilePairingPayload.parse(
            "mold://pair?version=1&base_url=http%3A%2F%2Fbox%3A7680&instance_id=i&name=box")
        #expect(read.token == nil)
        #expect(read.expiresAt == nil)
    }

    @Test func somethingElseEntirelyIsNotAPairingCode() {
        for raw in ["https://example.com", "hello", "mold://other?version=1", "mold://pair#frag"] {
            #expect(throws: MobilePairingPayload.ParseError.notPairingCode, "\(raw)") {
                try MobilePairingPayload.parse(raw)
            }
        }
    }

    @Test func aPairingCodeThisAppCannotReadIsSaidSo() {
        let unsupported = [
            // A future version.
            "mold://pair?version=2&base_url=http%3A%2F%2Fbox&instance_id=i&name=b",
            // Not an http(s) machine address.
            "mold://pair?version=1&base_url=ftp%3A%2F%2Fbox&instance_id=i&name=b",
            // Missing its instance.
            "mold://pair?version=1&base_url=http%3A%2F%2Fbox&name=b",
            // An expiry that is not a number.
            "mold://pair?version=1&base_url=http%3A%2F%2Fbox&expires_at=soon&instance_id=i&name=b",
            // JSON of the wrong type.
            #"{"type":"something","version":1,"base_url":"http://b","instance_id":"i","name":"b"}"#,
        ]
        for raw in unsupported {
            #expect(throws: MobilePairingPayload.ParseError.unsupported, "\(raw)") {
                try MobilePairingPayload.parse(raw)
            }
        }
    }

    @Test func expiryIsInSecondsAndChecked() throws {
        let read = try MobilePairingPayload.parse(
            "mold://pair?version=1&base_url=http%3A%2F%2Fbox&token=t&expires_at=100&instance_id=i&name=b")
        #expect(read.isExpired(at: Date(timeIntervalSince1970: 101)))
        #expect(!read.isExpired(at: Date(timeIntervalSince1970: 99)))
        #expect(!(try MobilePairingPayload.parse(
            "mold://pair?version=1&base_url=http%3A%2F%2Fbox&instance_id=i&name=b")).isExpired(at: .distantFuture))
    }
}
