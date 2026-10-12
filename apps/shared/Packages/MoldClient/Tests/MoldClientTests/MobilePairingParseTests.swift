import Foundation
import Testing
@testable import MoldClient

struct MobilePairingParseTests {
    @Test func trustedPinsAndPrivateInvitationRoundTripInFragmentOnly() throws {
        let payload = try #require(PairingLinkContractTests.fixtures.links.first).expected
        let url = try #require(payload.url)
        #expect(url.query == nil)
        #expect(url.fragment?.contains("invite_seed=") == true)
        #expect(try MobilePairingPayload.parse(url.absoluteString) == payload)
        #expect(payload.isExpired(at: Date(timeIntervalSince1970: 4_000_000_001)))
    }
    @Test func nativeAppHandoffRetainsTheSameValidatedInvitation() throws {
        for fixture in PairingLinkContractTests.fixtures.links {
            let universal = try #require(URLComponents(string: fixture.link))
            let raw = "moldstudio://pair#" + (universal.percentEncodedFragment ?? "")
            let nativeURL = try #require(URL(string: raw))
            #expect(MobilePairingPayload.isPairingLink(nativeURL))
            #expect(try MobilePairingPayload.parse(raw) == fixture.expected)
        }
    }

    @Test func nativeAppHandoffRejectsAmbiguousOrForeignRoutes() throws {
        let universal = try #require(URLComponents(string: PairingLinkContractTests.fixtures.links.first!.link))
        let fragment = try #require(universal.percentEncodedFragment)
        for prefix in ["mold://pair", "moldstudio://other", "moldstudio://pair.evil", "moldstudio://user@pair",
                       "moldstudio://pair:443", "moldstudio://pair/other", "moldstudio://pair?token=private"] {
            let raw = prefix + "#" + fragment
            #expect(!MobilePairingPayload.isPairingLink(try #require(URL(string: raw))))
            #expect(throws: MobilePairingPayload.ParseError.notPairingCode) {
                try MobilePairingPayload.parse(raw)
            }
        }
        let tampered = "moldstudio://pair#version=9"
        #expect(throws: MobilePairingPayload.ParseError.unsupported) { try MobilePairingPayload.parse(tampered) }
    }

    @Test func unencryptedLegacySessionsCannotProduceACode() {
        for keyed in [false, true] {
            let legacy = PairingSession(token: "old", expiresAt: 4_000_000_000, authRequired: keyed, instanceId: "old", hostname: "old")
            #expect(MobilePairingPayload(session: legacy, baseURL: URL(string: "http://host")!, name: "host") == nil)
        }
    }
    @Test func tamperedHostPinAndInvitationSeedAreRefused() throws {
        let link = try #require(PairingLinkContractTests.fixtures.links.first).link
        let expected = try MobilePairingPayload.parse(link)
        let invite = try #require(expected.secureInvitation)
        for raw in [link.replacingOccurrences(of: invite.seed, with: String(repeating: "0", count: 64)),
                    link.replacingOccurrences(of: invite.binding.noiseKey, with: String(repeating: "0", count: 64))] {
            #expect(throws: MobilePairingPayload.ParseError.unsupported) { try MobilePairingPayload.parse(raw) }
        }
    }
    @Test func everyVersionOneFormIsRejected() {
        for raw in ["mold://pair?version=1&base_url=http%3A%2F%2Fbox&instance_id=i&name=b",
                    "https://utensils.io/mold/pair#version=1&base_url=http%3A%2F%2Fbox&instance_id=i&name=b",
                    #"{"type":"mold.mobile-pairing","version":1,"base_url":"http://box","instance_id":"i","name":"b"}"#] {
            #expect(throws: MobilePairingPayload.ParseError.self) { try MobilePairingPayload.parse(raw) }
        }
    }
    @Test func pairingAddressesRejectCredentialsPathsQueriesAndFragments() throws {
        let link = try #require(PairingLinkContractTests.fixtures.links.first).link
        for invalid in ["http://user:secret@host", "https://host/api", "https://host?token=secret", "https://host#private", "http://host:0"] {
            var parts = URLComponents(string: link)!
            var fields = FormURLEncoded.decode(parts.percentEncodedFragment!)
            fields.removeAll { $0.0 == "base_url" }; fields.append(("base_url", invalid))
            parts.percentEncodedFragment = fields.map { FormURLEncoded.encode($0.0) + "=" + FormURLEncoded.encode($0.1) }.joined(separator: "&")
            #expect(throws: MobilePairingPayload.ParseError.unsupported) { try MobilePairingPayload.parse(parts.url!.absoluteString) }
        }
        var parts = URLComponents(string: link)!
        var fields = FormURLEncoded.decode(parts.percentEncodedFragment!)
        fields.removeAll { $0.0 == "websocket_url" }; fields.append(("websocket_url", "wss://relay.example/prod?token=private"))
        parts.percentEncodedFragment = fields.map { FormURLEncoded.encode($0.0) + "=" + FormURLEncoded.encode($0.1) }.joined(separator: "&")
        #expect(throws: MobilePairingPayload.ParseError.unsupported) { try MobilePairingPayload.parse(parts.url!.absoluteString) }
    }

    @Test func localInvitationNeedsNoCloudAccountOrWebSocketURL() throws {
        let link = try #require(PairingLinkContractTests.fixtures.links.first).link
        var parts = URLComponents(string: link)!
        var fields = FormURLEncoded.decode(parts.percentEncodedFragment!)
        fields.removeAll { $0.0 == "websocket_url" }
        parts.percentEncodedFragment = FormURLEncoded.queryString(fields)
        let payload = try MobilePairingPayload.parse(parts.url!.absoluteString)
        #expect(try MobilePairingPayload.parse(#require(payload.url).absoluteString) == payload)
        #expect(payload.secureInvitation?.websocketURL == nil)
        var json = Dictionary(uniqueKeysWithValues: fields.map { ($0.0, $0.1 as Any) })
        json["type"] = "mold.mobile-pairing"; json["version"] = 2; json["websocket_url"] = NSNull()
        for key in ["invite", "endpoints"] {
            if let value = json[key] as? String { json[key] = try JSONSerialization.jsonObject(with: Data(value.utf8)) }
        }
        #expect(try MobilePairingPayload.parse(String(decoding: JSONSerialization.data(withJSONObject: json), as: UTF8.self)) == payload)
        json["websocket_url"] = 123
        #expect(throws: MobilePairingPayload.ParseError.unsupported) {
            try MobilePairingPayload.parse(String(decoding: JSONSerialization.data(withJSONObject: json), as: UTF8.self))
        }
    }
}
