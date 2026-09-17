import Foundation

/// The guards `TransferPlan.next` calls into -- split out purely for size.
extension TransferPlan {
    /// Whether `batch` is safe to treat as this transfer's landed job --
    /// `nil` means yes; otherwise the sentence that refuses it, always
    /// leaving the source untouched (`queueTransfer.ts:157-174`).
    public static func verify(
        _ batch: BatchStatus, clientBatchId: String, destination: String
    ) -> TransferOutcome? {
        guard batch.instanceId == destination, batch.clientBatchId == clientBatchId,
              batch.durable == true, batch.children.count == 1
        else {
            return .refused(
                "The destination returned an unexpected job identity. The original remains held.")
        }
        let child = batch.children[0]
        guard child.state != .failed, child.state != .cancelled else {
            return .refused(
                "The destination job \(child.state.rawValue). The original remains held; "
                    + "choose another machine or inspect the destination.")
        }
        return nil
    }

    /// Typed `503` refusals the destination answers BEFORE anything is
    /// committed -- `PRE_COMMIT_REFUSAL_CODES` (`generationAdmission.ts:82-90`).
    private static let preCommitRefusalCodes: Set<String> = [
        "DURABLE_ADMISSION_UNAVAILABLE", "DURABLE_MEDIA_UNAVAILABLE",
        "QUEUE_FULL", "SERVER_RESTARTING",
    ]

    /// Whether `error` proves the destination rejected the admission before
    /// commit, versus an ambiguous answer that must be reconciled with a
    /// lookup (`isDefiniteGenerationAdmissionRejection`,
    /// `generationAdmission.ts:98-113`). 413 is its own case: the body is
    /// simply too large, which is definite but names a different sentence
    /// than a generic rejection.
    ///
    /// `.unauthorized` and `.licenseRequired` are their own cases on
    /// `MoldClientError` rather than an `.http` with a status, so matching
    /// only `.http` left both falling through to `.ambiguous`: the plan then
    /// issued a SECOND authenticated lookup against the same machine, which
    /// failed the same way, and told the user to "retry this same destination
    /// to check safely" -- advice that can never succeed, for a refusal that
    /// was definite and pre-commit. Both name what would actually resolve
    /// them, in the destination's terms rather than this machine's.
    public static func classifyAdmitFailure(_ error: Error) -> AdmitResult {
        guard let clientError = error as? MoldClientError else { return .ambiguous }
        if case .unauthorized = clientError {
            return .rejected(
                "The destination needs an API key, and this Mac does not have the right one. "
                    + "Add it in Settings and try again.")
        }
        if case let .licenseRequired(refusal, mismatch) = clientError {
            return .rejected(
                mismatch
                    ? "The destination pins different terms for \(refusal.name)."
                    : "\(refusal.name) has to be accepted on the destination first.")
        }
        guard case let .http(status, code, message) = clientError else { return .ambiguous }
        if status == 413 { return .tooLarge }
        if status == 503, let code, preCommitRefusalCodes.contains(code) {
            return .rejected(message ?? "The destination refused this request.")
        }
        if (400 ..< 500).contains(status), status != 408, status != 425, status != 429 {
            return .rejected(message ?? "The destination refused this request (\(status)).")
        }
        return .ambiguous
    }

    /// Whether `error` is the destination or source answering "no such
    /// batch" / "no such job" -- the one 404 that means "proceed", never a
    /// failure of the whole transfer.
    public static func isNotFound(_ error: Error) -> Bool {
        guard let clientError = error as? MoldClientError, case let .http(status, _, _) = clientError
        else { return false }
        return status == 404
    }
}
