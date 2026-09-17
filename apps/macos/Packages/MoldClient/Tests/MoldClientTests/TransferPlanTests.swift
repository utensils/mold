import Foundation
import Testing

@testable import MoldClient

/// `TransferPlan.next` ported from `studio/api/queueTransfer.ts:50-199`
/// (design M6 S4). Every case here is a guard the studio learned the hard
/// way -- no network, no `FakeBackend`, just the decision.
private func batch(
    instanceId: String = "dest-1", clientBatchId: String = "transfer-1",
    durable: Bool? = true, children: [BatchChildState]
) -> BatchStatus {
    BatchStatus(
        id: "b1", clientBatchId: clientBatchId, instanceId: instanceId, durable: durable,
        children: children.enumerated().map {
            BatchChild(index: $0.offset, jobId: "j\($0.offset)", state: $0.element, error: nil,
                       errorCode: nil, retryable: nil, revision: 1, updatedAtMs: nil, result: nil)
        })
}

private func next(
    _ step: TransferPlan.Step, _ outcome: TransferPlan.Outcome,
    clientBatchId: String = "transfer-1", destinationInstance: String = "dest-1",
    destinationLabel: String = "hal9000", sourceLabel: String = "plato"
) -> TransferPlan.Result {
    TransferPlan.next(
        after: step, outcome: outcome, clientBatchId: clientBatchId,
        destinationInstance: destinationInstance, destinationLabel: destinationLabel,
        sourceLabel: sourceLabel)
}

@Test func anIdentityThatChangedBetweenThePickerAndTheClickStopsTheSend() {
    let result = next(.verifyIdentities, .identities(sourceChanged: true, destinationChanged: false))
    #expect(result == .outcome(.refused("A machine's identity changed. Refresh the machines and try again.")))
}

@Test func matchingIdentitiesProceedToThePriorAttemptCheck() {
    let result = next(.verifyIdentities, .identities(sourceChanged: false, destinationChanged: false))
    #expect(result == .step(.checkPriorAttempt))
}

@Test func aDestinationThatAlreadyHoldsThisBatchIsNotSentToTwice() {
    let landed = batch(children: [.running])
    let result = next(.checkPriorAttempt, .priorAttempt(landed, sourceHeld: true))
    #expect(result == .step(.complete))
}

@Test func aSourceThatIsNoLongerHeldNeverReachesTheDestination() {
    let result = next(.checkPriorAttempt, .priorAttempt(nil, sourceHeld: false))
    #expect(result == .outcome(.refused("This job is no longer held. Refresh the queue before sending it.")))
}

@Test func noPriorAttemptAndAHeldSourceMovesOnToExport() {
    let result = next(.checkPriorAttempt, .priorAttempt(nil, sourceHeld: true))
    #expect(result == .step(.export))
}

@Test func aPriorAttemptFoundAfterTheSourceWasAlreadyRemovedIsStillASend() {
    let landed = batch(children: [.complete])
    let result = next(.checkPriorAttempt, .priorAttempt(landed, sourceHeld: false))
    #expect(result == .outcome(.sent(
        sourceRemoved: true,
        message: "Sent to hal9000. The original was already removed from plato's queue.")))
}

@Test func aBatchWithTwoChildrenIsRefusedAsAnUnexpectedIdentity() {
    let landed = batch(children: [.running, .running])
    let result = next(.checkPriorAttempt, .priorAttempt(landed, sourceHeld: true))
    #expect(result == .outcome(.refused(
        "The destination returned an unexpected job identity. The original remains held.")))
}

@Test func aWrongDestinationInstanceIsRefusedAsAnUnexpectedIdentity() {
    let landed = batch(instanceId: "some-other-instance", children: [.running])
    let result = next(.checkPriorAttempt, .priorAttempt(landed, sourceHeld: true))
    #expect(result == .outcome(.refused(
        "The destination returned an unexpected job identity. The original remains held.")))
}

@Test func aDestinationChildThatFailedLeavesTheSourceHeld() {
    let landed = batch(children: [.failed])
    let result = next(.checkPriorAttempt, .priorAttempt(landed, sourceHeld: true))
    #expect(result == .outcome(.refused(
        "The destination job failed. The original remains held; choose another machine or inspect the destination.")))
}

@Test func aDestinationChildThatWasCancelledLeavesTheSourceHeld() {
    let landed = batch(children: [.cancelled])
    let result = next(.checkPriorAttempt, .priorAttempt(landed, sourceHeld: true))
    #expect(result == .outcome(.refused(
        "The destination job cancelled. The original remains held; choose another machine or inspect the destination.")))
}

@Test func anAdmittedBatchMovesOnToComplete() {
    let landed = batch(children: [.accepted])
    let result = next(.admit, .admitResult(.landed(landed)))
    #expect(result == .step(.complete))
}

@Test func aBodyTooLargeSaysWhatTheLimitIs() {
    let result = next(.admit, .admitResult(.tooLarge))
    #expect(result == .outcome(.refused(
        "hal9000 wouldn't take it: the job's media is larger than a machine will accept in one request "
            + "(about 48 MB). The original is still here.")))
}

@Test func aDefiniteRejectionDoesNotLookUp() {
    let result = next(.admit, .admitResult(.rejected(
        "This job uses a machine-local LoRA. Install and select that adapter on the destination before resubmitting.")))
    #expect(result == .outcome(.refused(
        "This job uses a machine-local LoRA. Install and select that adapter on the destination before resubmitting.")))
}

@Test func aMachineLocalLoraRefusalIsQuotedVerbatim() {
    let sentence = "This job uses a machine-local LoRA. Install and select that adapter on the destination before resubmitting."
    #expect(TransferPlan.classifyAdmitFailure(MoldClientError.http(status: 422, code: nil, message: sentence)) == .rejected(sentence))
}

@Test func anAmbiguousAdmissionFailureLooksUpInsteadOfAssuming() {
    let result = next(.admit, .admitResult(.ambiguous))
    #expect(result == .step(.confirmAfterAmbiguousAdmit))
}

@Test func aConfirmedLookupAfterAnAmbiguousAdmitMovesOnToComplete() {
    let landed = batch(children: [.running])
    let result = next(.confirmAfterAmbiguousAdmit, .confirmed(landed))
    #expect(result == .step(.complete))
}

@Test func noConfirmationAfterAnAmbiguousAdmitLeavesTheSourceHeld() {
    let result = next(.confirmAfterAmbiguousAdmit, .confirmed(nil))
    #expect(result == .outcome(.refused(
        "Acceptance by hal9000 is not confirmed. The original remains held. Retry this same destination to check safely.")))
}

@Test func theHappyPathCompleteReportsTheSourceRemoved() {
    let result = next(.complete, .completed(removed: true))
    #expect(result == .outcome(.sent(
        sourceRemoved: true, message: "Sent to hal9000. The original was removed from plato's queue.")))
}

@Test func aCompleteThatFailsIsStillASend() {
    let result = next(.complete, .completed(removed: false))
    #expect(result == .outcome(.sent(
        sourceRemoved: false,
        message: "Sent to hal9000. The original could not be removed; check plato before retrying it.")))
}

// MARK: - classifyAdmitFailure / isNotFound

@Test func a413IsClassifiedAsTooLarge() {
    #expect(TransferPlan.classifyAdmitFailure(MoldClientError.http(status: 413, code: nil, message: nil)) == .tooLarge)
}

@Test func aKnownPreCommitRefusalCodeIsADefiniteRejection() {
    let result = TransferPlan.classifyAdmitFailure(
        MoldClientError.http(status: 503, code: "QUEUE_FULL", message: "The destination's queue is full."))
    #expect(result == .rejected("The destination's queue is full."))
}

@Test func aGenericServerErrorIsAmbiguous() {
    #expect(TransferPlan.classifyAdmitFailure(MoldClientError.http(status: 500, code: nil, message: "oops")) == .ambiguous)
}

@Test func aTooManyRequestsErrorIsAmbiguousNotADefiniteRejection() {
    #expect(TransferPlan.classifyAdmitFailure(MoldClientError.http(status: 429, code: nil, message: "slow down")) == .ambiguous)
}

@Test func anUnreachableErrorIsAmbiguous() {
    #expect(TransferPlan.classifyAdmitFailure(MoldClientError.unreachable("timed out")) == .ambiguous)
}

@Test func a404IsRecognizedAsNotFound() {
    #expect(TransferPlan.isNotFound(MoldClientError.http(status: 404, code: nil, message: nil)))
    #expect(!TransferPlan.isNotFound(MoldClientError.http(status: 409, code: nil, message: nil)))
    #expect(!TransferPlan.isNotFound(MoldClientError.unreachable("down")))
}

// MARK: - verify

@Test func verifyAcceptsAMatchingSingleChildBatch() {
    let landed = batch(children: [.running])
    #expect(TransferPlan.verify(landed, clientBatchId: "transfer-1", destination: "dest-1") == nil)
}

@Test func verifyRefusesAMismatchedClientBatchId() {
    let landed = batch(clientBatchId: "some-other-id", children: [.running])
    #expect(TransferPlan.verify(landed, clientBatchId: "transfer-1", destination: "dest-1") != nil)
}

@Test func verifyRefusesANonDurableBatch() {
    let landed = batch(durable: nil, children: [.running])
    #expect(TransferPlan.verify(landed, clientBatchId: "transfer-1", destination: "dest-1") != nil)
}
