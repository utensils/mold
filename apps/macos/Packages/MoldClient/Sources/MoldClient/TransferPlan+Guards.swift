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
    public static func classifyAdmitFailure(_ error: Error) -> AdmitResult {
        guard let clientError = error as? MoldClientError, case let .http(status, code, message) = clientError
        else { return .ambiguous }
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
