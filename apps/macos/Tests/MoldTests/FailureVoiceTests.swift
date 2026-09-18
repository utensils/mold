import Foundation
import MoldClient
import Testing

@testable import Mold

/// One failure, and the three things a person is owed about it.
struct FailureVoiceRow: Sendable {
    let error: MoldClientError
    /// What the machine itself said, standing as a sentence.
    let reason: String
    /// What to do about it, or nothing when there is honestly nothing.
    let advice: String?
}

/// Every case `MoldClientError` has, and the retryable and final shapes of the
/// one that carries a status. A new case added without a route -- or with one
/// worded twice -- lands here first.
///
/// At file scope because `@Test(arguments:)` reads its table from outside the
/// suite's actor.
nonisolated let failureVoiceTable: [FailureVoiceRow] = [
    .init(error: .unreachable("The request timed out."), reason: "The request timed out.",
          advice: "Check the machine under Machines."),
    .init(error: .unauthorized, reason: "It needs an API key.", advice: "Add one in Settings."),
    .init(error: .http(status: 404, code: nil, message: "Image not found."),
          reason: "Image not found.", advice: nil),
    .init(error: .http(status: 422, code: nil, message: "No such device."),
          reason: "No such device.", advice: nil),
    .init(error: .http(status: 503, code: nil, message: "Busy."), reason: "Busy.",
          advice: "Try again in a moment."),
    .init(error: .http(status: 429, code: nil, message: "Too many."), reason: "Too many.",
          advice: "Try again in a moment."),
    .init(error: .http(status: 500, code: nil, message: nil),
          reason: "It answered with an error (500).", advice: "Try again in a moment."),
    .init(error: .malformedResponse,
          reason: "It answered something this version of Mold can't read.",
          advice: "Update Mold here, or on that machine."),
]

/// The voice, pinned. Every failure a person reads is three things: what did
/// not happen, the machine's own reason for it, and the way forward when
/// there is one.
///
/// **Fails today**: the third part did not exist. A refusal reached the canvas
/// as "That didn't arrive · Image not found" -- a headline and the server's
/// raw clause, with nothing to do about either -- and the one route the app
/// did word, "Add one in Settings.", was welded onto the REASON, so the
/// banner and the canvas could not say it from the same place.
///
/// The table is per ERROR rather than per surface, because that is the whole
/// point: `Error.advice` is the only copy of each route, so a banner can
/// never drift from what the same failure says on a canvas.
@MainActor
struct FailureVoiceTests {
    @Test(arguments: failureVoiceTable)
    func aFailureSaysItsReasonAndThenItsWayForward(row: FailureVoiceRow) {
        #expect(row.error.reasonSentence == row.reason)
        #expect(row.error.advice == row.advice)
        #expect(row.error.failureSentence
            == [row.reason, row.advice].compactMap { $0 }.joined(separator: " "))
    }

    /// Nothing says the same thing twice: a route never appears inside the
    /// reason clause it follows, and it is always the tail.
    @Test(arguments: failureVoiceTable)
    func noRouteIsWeldedIntoItsReason(row: FailureVoiceRow) {
        guard let advice = row.advice else {
            #expect(row.error.failureSentence == row.reason)
            return
        }
        #expect(!row.reason.contains(advice))
        #expect(row.error.failureSentence.hasSuffix(advice))
    }

    /// Nobody reads a paragraph on a canvas.
    @Test(arguments: failureVoiceTable)
    func everySentenceStaysShort(row: FailureVoiceRow) {
        #expect(row.error.failureSentence.count <= 120)
    }

    /// "Try again" is asked of `isTransient` rather than decided a second
    /// time here. The bug the owner photographed is the first row: a 404 on
    /// the finished render ends at the machine's four words, because retrying
    /// a missing file cannot help -- what the SURFACE adds is the copy that
    /// does exist, which is why `advice` is allowed to be nil.
    @Test func onlySomethingWaitingCouldFixIsOfferedARetry() {
        for row in failureVoiceTable {
            let offersRetry = row.advice == "Try again in a moment."
            guard case .unreachable = row.error else {
                #expect(offersRetry == row.error.isTransient)
                continue
            }
            // Transient, but it has something better to say than "try again".
            #expect(row.error.isTransient)
            #expect(!offersRetry)
        }
    }

    /// An error from outside `MoldClient` -- a file read, a decode -- keeps
    /// its own words and claims no route it cannot offer.
    @Test func aForeignErrorInventsNoRoute() {
        let foreign = CocoaError(.fileNoSuchFile)
        #expect(foreign.advice == nil)
        #expect(foreign.failureSentence == foreign.reasonSentence)
    }

    /// A licence refusal is the one failure the app can RESOLVE, so it says
    /// where. Built by decoding the server's own payload, the `FakeFixtures`
    /// rule.
    @Test func aLicenceRefusalSaysWhereToAcceptIt() throws {
        let refusal = try MoldJSON.decoder.decode(LicenseRefusal.self, from: Data("""
        {"id": "hunyuan3d-2.1", "name": "Hunyuan3D 2.1 Community License",
         "url": "https://example.invalid/terms", "canonical": "https://example.invalid/terms",
         "sha256": "0", "summary": "The 2.1 community terms."}
        """.utf8))
        let required = MoldClientError.licenseRequired(refusal, mismatch: false)
        #expect(required.advice == "Accept the terms under Models.")
        #expect(required.failureSentence
            == "Hunyuan3D 2.1 Community License has to be accepted on this machine first. "
            + "Accept the terms under Models.")
    }
}
