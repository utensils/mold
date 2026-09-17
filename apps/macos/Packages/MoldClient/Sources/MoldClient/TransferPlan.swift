import Foundation

/// The decisions inside a queue transfer, pure and network-free.
///
/// Ported from the studio's only battle-tested implementation of a sequence
/// that spans two machines and can be interrupted between any two of its
/// calls (`studio/api/queueTransfer.ts:50-199`). Each guard exists for a
/// different failure:
///  - re-reading both `/api/status` identities fences a machine that
///    restarted between the picker and the click;
///  - looking the destination up FIRST recovers a prior attempt instead of
///    admitting a second job;
///  - reading the source's authority at send time tells "already
///    transferred" (destination found, source no longer held) from "no
///    longer held" (destination missing, source no longer held);
///  - an AMBIGUOUS admission failure looks the destination up again rather
///    than assuming failure, because a dropped response is not a refusal;
///  - `transfer/complete` runs only after exactly one non-failed child
///    landed, and its OWN failure is a success with a caveat, never a retry.
///
/// The driver (`TransferStore`) makes every network call and classifies what
/// came back into an `Outcome`; this type only decides what happens next,
/// which is what makes every branch testable without a backend. `verify`,
/// `classifyAdmitFailure` and `isNotFound` -- the guards `next` calls into --
/// live in `TransferPlan+Guards.swift`.
public enum TransferPlan {
    public enum Step: Hashable, Sendable {
        case verifyIdentities, checkPriorAttempt, export, admit
        /// The recovery lookup after `admit` answered something that might or
        /// might not have landed -- distinct from `checkPriorAttempt` because
        /// "nothing there" means something different the second time: not
        /// "safe to try", but "not confirmed, and the source stays held".
        case confirmAfterAmbiguousAdmit
        case complete
    }

    /// What `admit` came back with, already classified -- see
    /// `classifyAdmitFailure`.
    public enum AdmitResult: Hashable, Sendable {
        case landed(BatchStatus)
        case tooLarge
        case rejected(String)
        case ambiguous
    }

    /// What one step answered, reduced to what `next` needs to decide.
    public enum Outcome: Hashable, Sendable {
        case identities(sourceChanged: Bool, destinationChanged: Bool)
        /// The destination's answer to "has this transfer already landed",
        /// and whether the source is STILL a held batch child right now --
        /// read together, because their combination is the whole fact.
        case priorAttempt(BatchStatus?, sourceHeld: Bool)
        case admitResult(AdmitResult)
        case confirmed(BatchStatus?)
        case completed(removed: Bool)
    }

    public enum Result: Hashable, Sendable {
        case step(Step)
        case outcome(TransferOutcome)
    }

    /// What to do after `step` answered `outcome`.
    public static func next(
        after step: Step, outcome: Outcome,
        clientBatchId: String, destinationInstance: String,
        destinationLabel: String, sourceLabel: String
    ) -> Result {
        switch (step, outcome) {
        case let (.verifyIdentities, .identities(sourceChanged, destinationChanged)):
            guard !sourceChanged, !destinationChanged else {
                return .outcome(.refused(
                    "A machine's identity changed. Refresh the machines and try again."))
            }
            return .step(.checkPriorAttempt)

        case let (.checkPriorAttempt, .priorAttempt(.some(batch), sourceHeld)):
            if let refusal = verify(batch, clientBatchId: clientBatchId, destination: destinationInstance) {
                return .outcome(refusal)
            }
            guard sourceHeld else {
                return .outcome(.sent(
                    sourceRemoved: true,
                    message: "Sent to \(destinationLabel). The original was already removed from \(sourceLabel)'s queue."))
            }
            return .step(.complete)

        case let (.checkPriorAttempt, .priorAttempt(.none, sourceHeld)):
            guard sourceHeld else {
                return .outcome(.refused(
                    "This job is no longer held. Refresh the queue before sending it."))
            }
            return .step(.export)

        case let (.admit, .admitResult(result)):
            switch result {
            case let .landed(batch):
                if let refusal = verify(batch, clientBatchId: clientBatchId, destination: destinationInstance) {
                    return .outcome(refusal)
                }
                return .step(.complete)
            case .tooLarge:
                return .outcome(.refused(
                    "\(destinationLabel) wouldn't take it: the job's media is larger than a machine "
                        + "will accept in one request (about 48 MB). The original is still here."))
            case let .rejected(message):
                return .outcome(.refused(message))
            case .ambiguous:
                return .step(.confirmAfterAmbiguousAdmit)
            }

        case let (.confirmAfterAmbiguousAdmit, .confirmed(.some(batch))):
            if let refusal = verify(batch, clientBatchId: clientBatchId, destination: destinationInstance) {
                return .outcome(refusal)
            }
            return .step(.complete)

        case (.confirmAfterAmbiguousAdmit, .confirmed(.none)):
            return .outcome(.refused(
                "Acceptance by \(destinationLabel) is not confirmed. The original remains held. "
                    + "Retry this same destination to check safely."))

        case let (.complete, .completed(removed)):
            return .outcome(.sent(
                sourceRemoved: removed,
                message: removed
                    ? "Sent to \(destinationLabel). The original was removed from \(sourceLabel)'s queue."
                    : "Sent to \(destinationLabel). The original could not be removed; "
                        + "check \(sourceLabel) before retrying it."))

        default:
            // Every reachable (step, outcome) pair is matched above; a call
            // out of order is a driver bug, not a network outcome, so it
            // refuses rather than crashing.
            return .outcome(.refused("Internal error: transfer step out of order."))
        }
    }

}

/// The final word on a transfer.
public enum TransferOutcome: Hashable, Sendable {
    case sent(sourceRemoved: Bool, message: String)
    case refused(String)
}
