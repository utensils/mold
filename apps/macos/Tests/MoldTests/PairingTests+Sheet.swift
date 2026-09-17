import Foundation
import MoldClient
import Testing

@testable import Mold

/// `PairingSheet`'s own pure logic -- split out of `PairingTests.swift` past
/// the file-size advisory, sharing that file's `machine(_:)` helper.
@MainActor
extension PairingTests {
    /// **Fails today** -- `PairingSheet` does not exist. A code with no
    /// token redeems nothing (`MobilePairingPayload.init?`), so the sheet
    /// has nothing honest to show.
    @Test func aSessionWithNoTokenOffersNoCode() {
        let session = PairingSession(
            token: nil, expiresAt: nil, authRequired: false, instanceId: "inst-1", hostname: nil)
        let state = PairingSheet.resolve(
            session, baseURL: URL(string: "http://127.0.0.1:7680")!, name: "This Mac")
        #expect(state == .noCode)
    }

    @Test func noSessionYetIsWaitingNotNoCode() {
        let state = PairingSheet.resolve(
            nil, baseURL: URL(string: "http://127.0.0.1:7680")!, name: "This Mac")
        #expect(state == .waiting)
    }

    /// The countdown reads the MACHINE's own `expires_at`, not a client
    /// timer started when the sheet opened -- so it must answer correctly
    /// from an arbitrary `now`, not just "the moment the session arrived".
    @Test func theSheetCountsDownFromTheMachinesOwnExpiryNotFromNow() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        #expect(PairingSheet.Countdown.resolve(expiresAt: 1_700_000_030, now: now) == .remaining(30))
        #expect(PairingSheet.Countdown.resolve(expiresAt: 1_699_999_990, now: now) == .expired)
    }

    /// **Fails today**: `Countdown.resolve` divides `expires_at` by 1000, so
    /// a value the machine would actually emit is read as a moment in 1970
    /// and every code reads "Expired".
    ///
    /// `expires_at` is unix SECONDS: `auth.rs:216` adds
    /// `PAIRING_TOKEN_TTL_SECS` to `unix_timestamp()`, which is
    /// `.as_secs()` (`auth.rs:633-638`), and `routes.rs:9502` puts that
    /// number on the wire. Studio reads it the same way --
    /// `session.expires_at - Math.floor(Date.now() / 1000)`
    /// (`MobilePairingCard.vue:36-37`). The `/ 1_000` came from
    /// `PairedClient.lastUsedAtMs`, which genuinely IS milliseconds.
    @Test func theCountdownReadsTheMachinesOwnUnixSeconds() {
        // A real pairing session: issued at this instant, dead two minutes
        // later, exactly `PAIRING_TOKEN_TTL_SECS`.
        let issued = Date(timeIntervalSince1970: 1_758_067_200)
        let expiresAt: UInt64 = 1_758_067_200 + 120
        #expect(PairingSheet.Countdown.resolve(expiresAt: expiresAt, now: issued) == .remaining(120))
        #expect(PairingSheet.Countdown.resolve(
            expiresAt: expiresAt, now: issued.addingTimeInterval(121)) == .expired)
    }

    /// Reopening the sheet inside an unexpired session's window shows that
    /// code rather than minting another; a session for a DIFFERENT machine,
    /// or an expired one, is no reason to skip the request.
    @Test func theSheetReusesAnUnexpiredSessionForTheSameMachine() {
        let now = Date(timeIntervalSince1970: 1_700_000_000)
        let plato = machine(), other = machine("hal9000")
        let live = PairingSession(
            token: "t", expiresAt: 1_700_000_030, authRequired: true, instanceId: "i", hostname: "plato")
        let dead = PairingSession(
            token: "t", expiresAt: 1_699_999_990, authRequired: true, instanceId: "i", hostname: "plato")

        #expect(!PairingSheet.needsFreshCode(session: live, sessionHost: plato.id, host: plato.id, now: now))
        #expect(PairingSheet.needsFreshCode(session: live, sessionHost: other.id, host: plato.id, now: now))
        #expect(PairingSheet.needsFreshCode(session: dead, sessionHost: plato.id, host: plato.id, now: now))
        #expect(PairingSheet.needsFreshCode(session: nil, sessionHost: nil, host: plato.id, now: now))
    }
}
