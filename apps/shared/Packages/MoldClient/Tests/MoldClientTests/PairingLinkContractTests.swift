import Foundation
import Testing

@testable import MoldClient

/// The pairing-link contract shared with `studio/api/pairing.ts`: both read
/// `studio/api/pairing.fixtures.json`, so a code printed by the Mac, the
/// desktop app or the web reads the same on every phone.
struct PairingLinkContractTests {
    struct Fixture: Decodable {
        struct Payload: Decodable {
            let baseUrl: String
            let mode: HostConnectionMode?
            let pairingToken: String?
            let expiresAt: UInt64?
            let namespace: String?
            let hostKey: String?
            let hostNoiseKey: String?
            let hostBindingSignature: String?
            let websocketUrl: String?
            let inviteSeed: String?
            let invite: SecureInvite?
            let instanceId: String
            let name: String
            let endpoints: [ConnectionEndpoint]?
        }

        let payload: Payload
        let link: String
        let aliases: [String]

        var expected: MobilePairingPayload {
            if payload.mode == .regular {
                return MobilePairingPayload(baseURL: payload.baseUrl, token: payload.pairingToken,
                                           expiresAt: payload.expiresAt, instanceId: payload.instanceId,
                                           name: payload.name, mode: .regular)
            }
            return MobilePairingPayload(baseURL: payload.baseUrl, token: nil,
                                 expiresAt: payload.invite!.expiresAt, instanceId: payload.instanceId,
                                 name: payload.name, endpoints: payload.endpoints,
                                 secureInvitation: SecureInvitation(binding: .init(signingKey: payload.hostKey!, noiseKey: payload.hostNoiseKey!, signature: payload.hostBindingSignature!), namespace: payload.namespace!, websocketURL: URL(string: payload.websocketUrl!)!, seed: payload.inviteSeed!, invite: payload.invite!))
        }
    }

    struct Fixtures: Decodable {
        let links: [Fixture]
        let rejected: [String]
    }

    static let fixtures: Fixtures = {
        // Tests/MoldClientTests -> MoldClient -> Packages -> shared -> apps -> repository root.
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let data = try! Data(contentsOf: root.appending(path: "studio/api/pairing.fixtures.json"))
        return try! MoldJSON.decoder.decode(Fixtures.self, from: data)
    }()

    @Test(arguments: fixtures.links.map(\.link))
    func printsTheSharedLink(_ link: String) throws {
        let fixture = try #require(Self.fixtures.links.first { $0.link == link })
        #expect(fixture.expected.url?.absoluteString == link)
    }

    @Test(arguments: fixtures.links.map(\.link))
    func readsTheLinkAndItsSafeAliases(_ link: String) throws {
        let fixture = try #require(Self.fixtures.links.first { $0.link == link })
        for raw in [fixture.link] + fixture.aliases {
            #expect(try MobilePairingPayload.parse(raw) == fixture.expected, "\(raw)")
        }
    }

    @Test(arguments: fixtures.rejected)
    func refuses(_ raw: String) {
        #expect(throws: MobilePairingPayload.ParseError.self) { try MobilePairingPayload.parse(raw) }
    }

    @Test func aUniversalLinkIsRecognisedForRouting() {
        #expect(MobilePairingPayload.isPairingLink(URL(string: "https://utensils.io/mold/pair#version=1")!))
        #expect(!MobilePairingPayload.isPairingLink(URL(string: "mold://pair?version=1")!))
        #expect(!MobilePairingPayload.isPairingLink(URL(string: "https://utensils.io/mold/guide/companion")!))
        #expect(!MobilePairingPayload.isPairingLink(URL(string: "moldstudio://queue/1")!))
    }
}
