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
    // Unix SECONDS, the unit `auth.rs:216` issues and `routes.rs:9502`
    // sends -- `unix_timestamp()` is `.as_secs()` (`auth.rs:633-638`).
    #expect(session.expiresAt == 1_700_000_120)
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

/// Keyless hosts can issue address-only codes, but hold no paired client keys.
@Test func aKeylessMachineHasNoManagedPairedClients() throws {
    let clients = try MoldJSON.decoder.decode(
        PairedClients.self, from: RepoFixtures.fixture("pairing-clients-workstation.json"))
    #expect(!clients.authRequired)
    #expect(clients.pairingAvailable)
    #expect(!clients.canManagePairedClients)
}

/// A keyed code with no token cannot be redeemed.
@Test func aKeyedCodeWithNoTokenIsNoCodeAtAll() {
    let session = PairingSession(
        token: nil, expiresAt: nil, authRequired: true, instanceId: "inst-1", hostname: nil)
    let payload = MobilePairingPayload(
        session: session, baseURL: URL(string: "http://127.0.0.1:7680")!, name: "This Mac")
    #expect(payload == nil)
}

@Test func oldKeylessSessionCannotMakeAnUnencryptedCode() {
    let session = PairingSession(token: nil, expiresAt: nil, authRequired: false, instanceId: "old", hostname: nil)
    #expect(MobilePairingPayload(session: session, baseURL: URL(string: "http://box:7680")!, name: "Box") == nil)
}

@Test func formEncodingSpellsASpaceAsAPlusAndAColonAsPercent3A() {
    #expect(FormURLEncoded.encode("a b") == "a+b")
    #expect(FormURLEncoded.encode(":") == "%3A")
    #expect(FormURLEncoded.encode("/") == "%2F")
    #expect(FormURLEncoded.encode("a-B_9.*") == "a-B_9.*")
}
