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
        #expect(PairingSheet.Countdown.resolve(expiresAt: 1_700_000_030_000, now: now) == .remaining(30))
        #expect(PairingSheet.Countdown.resolve(expiresAt: 1_699_999_990_000, now: now) == .expired)
    }
}
