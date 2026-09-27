import Foundation
import Testing

@testable import MoldClient

// Two refusals `classifyAdmitFailure` could not see. `TransferPlanTests`
// covers 413, a coded 503, 429 and `.unreachable` -- all of which arrive as
// `MoldClientError.http`, which is the only case the classifier matched.

/// **Fails today**: `HTTPBackend.check(_:_:)` throws `.unauthorized` for a
/// 401 rather than `.http(401, …)`, so moving a held job to a destination
/// whose API key is wrong fell through to `.ambiguous`. The plan then issues
/// ANOTHER authenticated lookup against the same machine, which fails the
/// same way, and the user is told "Acceptance by X is not confirmed. The
/// original remains held. Retry this same destination to check safely" --
/// advice that can never succeed, for a failure that was definite and
/// PRE-COMMIT.
@Test func aDestinationWithTheWrongKeyIsADefiniteRejection() {
    let result = TransferPlan.classifyAdmitFailure(MoldClientError.unauthorized)
    #expect(result == .rejected(
        "The destination needs an API key, and this Mac does not have the right one. "
            + "Add it in Settings and try again."))
}

/// A licence refusal is definite and pre-commit too, and it names the one
/// thing that resolves it -- which is not "retry".
@Test func aDestinationThatNeedsALicenceIsADefiniteRejection() {
    let refusal = LicenseRefusal(
        id: "h3", name: "Hunyuan3D 2.1", url: "https://x/l", canonical: "c",
        sha256: "ab", summary: "s")
    #expect(TransferPlan.classifyAdmitFailure(
        MoldClientError.licenseRequired(refusal, mismatch: false))
        == .rejected("Hunyuan3D 2.1 has to be accepted on the destination first."))
    #expect(TransferPlan.classifyAdmitFailure(
        MoldClientError.licenseRequired(refusal, mismatch: true))
        == .rejected("The destination pins different terms for Hunyuan3D 2.1."))
}

/// A refusal the classifier is sure about goes straight to the outcome
/// without a second lookup -- which is the whole difference the two cases
/// above make.
@Test func aDefiniteRejectionFromAnAuthFailureDoesNotLookUp() {
    let result = TransferPlan.next(
        after: .admit,
        outcome: .admitResult(TransferPlan.classifyAdmitFailure(MoldClientError.unauthorized)),
        clientBatchId: "c", destinationInstance: "i",
        destinationLabel: "hal9000", sourceLabel: "workstation")
    guard case let .outcome(.refused(sentence)) = result else {
        Issue.record("expected a refusal, got \(result)")
        return
    }
    #expect(sentence.contains("API key"))
}

/// The cases that were already right stay right -- a malformed answer says
/// nothing about whether the destination committed, so it is still ambiguous.
@Test func anUnreadableAnswerIsStillAmbiguous() {
    #expect(TransferPlan.classifyAdmitFailure(MoldClientError.malformedResponse) == .ambiguous)
}

/// A held job whose media is too large to post inline is the ONE parity gap
/// this app keeps on purpose: `HTTPBackend+Transfer.swift` posts the export
/// bytes in the request, where web and desktop route them through an upload
/// session (`studio/api/queueTransfer.ts:119-143`). The README scopes upload
/// sessions out, so what matters is that the limit is EXPLAINED rather than
/// surfacing as a bare 413 from a route nobody named -- this walks the whole
/// path, raw error to sentence.
@Test func aTransferTooLargeToPostSaysWhatTheLimitIsAndWhereTheJobStayed() {
    let raw = MoldClientError.http(status: 413, code: nil, message: nil)
    #expect(TransferPlan.classifyAdmitFailure(raw) == .tooLarge)

    let result = TransferPlan.next(
        after: .admit, outcome: .admitResult(TransferPlan.classifyAdmitFailure(raw)),
        clientBatchId: "c", destinationInstance: "i",
        destinationLabel: "hal9000", sourceLabel: "workstation")
    #expect(result == .outcome(.refused(
        "hal9000 wouldn't take it: the job's media is larger than a machine will accept in one "
            + "request (\(RequestBodyLimit.sentence)). The original is still here.")))
}
