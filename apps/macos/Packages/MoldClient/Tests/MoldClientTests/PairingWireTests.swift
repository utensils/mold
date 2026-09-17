import Foundation
import Testing

@testable import MoldClient

// This fleet is keyless (design fact 7), so `pairing-session-keyed.json` and
// `pairing-clients-keyed.json` are hand-built from `routes.rs:9499-9535`
// rather than captured -- a capture of a keyed host is not possible here.

@Test func aPairingSessionDecodesTheKeyedShape() throws {
    let session = try MoldJSON.decoder.decode(
        PairingSession.self, from: RepoFixtures.fixture("pairing-session-keyed.json"))
    #expect(session.token == "6f2a9c8e4b1d47a3ae7c9f0b2d5e8a41")
    #expect(session.expiresAt == 1_700_000_120_000)
    #expect(session.authRequired)
    #expect(session.instanceId == "inst-keyed-1")
    #expect(session.hostname == "forge")
}

@Test func aPairedClientsListingDecodesTheKeyedShape() throws {
    let clients = try MoldJSON.decoder.decode(
        PairedClients.self, from: RepoFixtures.fixture("pairing-clients-keyed.json"))
    #expect(clients.authRequired)
    #expect(clients.pairingAvailable)
    #expect(clients.clients.count == 1)
    #expect(clients.clients[0].id == "client-1")
    #expect(clients.clients[0].name == "James's iPhone")
    #expect(clients.clients[0].clientKind == "mobile")
}

/// `pairing_available` is `true` even on a keyless host (`routes.rs:9678-9685`),
/// so it is NOT the gate -- the gate trap this app must not fall into.
@Test func aKeylessMachineHasNothingToPair() throws {
    let clients = try MoldJSON.decoder.decode(
        PairedClients.self, from: RepoFixtures.fixture("pairing-clients-plato.json"))
    #expect(!clients.authRequired)
    #expect(clients.pairingAvailable)
    #expect(!clients.canPair)
}

/// A code with no token redeems nothing, so there is nothing honest to show.
@Test func aCodeWithNoTokenIsNoCodeAtAll() {
    let session = PairingSession(
        token: nil, expiresAt: nil, authRequired: false, instanceId: "inst-1", hostname: nil)
    let payload = MobilePairingPayload(
        session: session, baseURL: URL(string: "http://127.0.0.1:7680")!, name: "This Mac")
    #expect(payload == nil)
}

/// One hand-derived vector against `pairing.ts:47-56`: same field order,
/// `token`/`expires_at` present only because they are non-nil, no `type` at
/// all, and a hostname-with-a-space name so the `+` rule bites. Computed
/// independently with `node -e` against `URLSearchParams` to confirm the
/// byte-for-byte spelling before pinning it here.
@Test func aPairingUrlIsByteIdenticalToTheStudios() throws {
    let session = PairingSession(
        token: "abc def+ghi:jkl/mno", expiresAt: 1_700_000_000_000, authRequired: true,
        instanceId: "inst-1", hostname: nil)
    let payload = try #require(
        MobilePairingPayload(
            session: session, baseURL: URL(string: "https://100.105.134.43:7680")!,
            name: "James's Phone"))
    let url = try #require(payload.url)
    #expect(
        url.absoluteString
            == "mold://pair?version=1&base_url=https%3A%2F%2F100.105.134.43%3A7680"
            + "&token=abc+def%2Bghi%3Ajkl%2Fmno&expires_at=1700000000000"
            + "&instance_id=inst-1&name=James%27s+Phone")
}

/// A `nil` token or `expires_at` is simply OMITTED, not sent as an empty or
/// null parameter -- `pairing.ts:50-53`'s own `if` guards.
@Test func aPairingUrlOmitsAnAbsentTokenAndExpiry() throws {
    // Only reachable with authRequired == false, since a keyed session
    // always issues both together -- but the payload type itself makes no
    // such promise, so the omission is tested independently of that.
    let session = PairingSession(
        token: "tok", expiresAt: nil, authRequired: true, instanceId: "inst-1", hostname: nil)
    let payload = try #require(
        MobilePairingPayload(
            session: session, baseURL: URL(string: "http://box:7680")!, name: "Phone"))
    let url = try #require(payload.url)
    #expect(!url.absoluteString.contains("expires_at"))
    #expect(url.absoluteString.contains("token=tok"))
}

@Test func formEncodingSpellsASpaceAsAPlusAndAColonAsPercent3A() {
    #expect(FormURLEncoded.encode("a b") == "a+b")
    #expect(FormURLEncoded.encode(":") == "%3A")
    #expect(FormURLEncoded.encode("/") == "%2F")
    #expect(FormURLEncoded.encode("a-B_9.*") == "a-B_9.*")
}
