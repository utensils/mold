import Foundation
import Testing

@testable import MoldClient

private func heldEntry(
    model: String? = "flux2-dev:q8", heldReason: String? = "The host said this.",
    error: String? = nil, retryable: Bool? = true
) -> QueueEntry {
    QueueEntry(id: "j", model: model, state: .held, position: nil, startedAtUnixMs: nil,
               heldReason: heldReason, error: error, retryable: retryable, durable: nil,
               batchId: "batch", clientBatchId: "client", dispatchAttempts: nil, gpu: nil,
               targetGpu: nil, batchIndex: nil, explicitlyPaused: nil, replayed: nil)
}

private func child(errorCode: String?, retryable: Bool? = nil) -> BatchChild {
    BatchChild(index: 0, jobId: "j", state: .held, error: nil, errorCode: errorCode,
               retryable: retryable, revision: 1, updatedAtMs: nil, result: nil)
}

/// The typed code names two things, and the answer takes the model from the
/// ROW's own field -- the sentence is never parsed for it.
@Test func aMissingModelHoldNamesTheModelFromTheRow() {
    let entry = heldEntry(
        model: "flux2-dev:q8", heldReason: "That checkpoint isn't installed on this machine.")
    let hold = QueueHold.resolve(entry: entry, child: child(errorCode: "MODEL_NOT_FOUND"))
    #expect(hold == .missingModel(
        "flux2-dev:q8", sentence: "That checkpoint isn't installed on this machine."))
}

@Test func anUntypedHoldIsTheMachinesOwnSentenceAndARetry() throws {
    let listing = try MoldJSON.decoder.decode(
        QueueListing.self, from: RepoFixtures.fixture("queue.json"))
    let row = try #require(listing.entries.first)
    let hold = QueueHold.resolve(entry: row, child: nil)
    #expect(hold == .prose(row.heldReason ?? "", retryable: true))
}

@Test func anUnretryableHoldOffersNoRetry() {
    let entry = heldEntry(retryable: false)
    let hold = QueueHold.resolve(entry: entry, child: nil)
    #expect(hold == .prose("The host said this.", retryable: false))
}

/// A row with no batch status answer at all still has the machine's own
/// sentence -- the state this pane has shipped in since M1.
@Test func aHostThatAnsweredNoBatchStatusStillShowsTheRowsSentence() {
    let entry = heldEntry(heldReason: "GPU ran out of memory.", retryable: true)
    let hold = QueueHold.resolve(entry: entry, child: nil)
    #expect(hold == .prose("GPU ran out of memory.", retryable: true))
}

@Test func aLiveRowHasNoHoldAtAll() {
    let entry = heldEntry()
    let queued = QueueEntry(
        id: entry.id, model: entry.model, state: .queued, position: nil, startedAtUnixMs: nil,
        heldReason: entry.heldReason, error: entry.error, retryable: entry.retryable, durable: nil,
        batchId: entry.batchId, clientBatchId: entry.clientBatchId, dispatchAttempts: nil,
        gpu: nil, targetGpu: nil, batchIndex: nil, explicitlyPaused: nil, replayed: nil)
    #expect(QueueHold.resolve(entry: queued, child: nil) == nil)
}

@Test func legacyTechnicalHeldReasonGetsPlainEnglishWithoutLosingDiagnostic() {
    let raw = "tensor shape mismatch in attention kernel"
    let entry = heldEntry(heldReason: raw)
    let hold = QueueHold.resolve(entry: entry, child: nil)
    #expect(hold?.summary(modelName: "Model", hostName: "Machine") == UserFacingError.message(raw))
    #expect(QueueFailureDetails.diagnostic(entry) == raw)
}

@Test func manifestDownloadHintRepairsAComponentWithoutBatchCode() throws {
    let row = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"h3","state":"held","model":"h3-turbo","retryable":true,"held_reason":"Turbo adapter is missing.","model_download":{"model":"h3-turbo"}}"#.utf8))
    let hold = QueueHold.resolve(entry: row, child: nil)
    #expect(hold == .missingModel("h3-turbo", sentence: "Turbo adapter is missing."))
    #expect(hold?.summary(modelName: "H3", hostName: "Machine") == "Turbo adapter is missing.")
}

@Test func mismatchedOrBlankDownloadHintsCannotOverrideTheJobModel() throws {
    for name in ["other-model", "", "   "] {
        let data = try JSONSerialization.data(withJSONObject: ["id": "job", "state": "held", "model": "h3", "retryable": true,
                                                               "model_download": ["model": name]])
        let row = try MoldJSON.decoder.decode(QueueEntry.self, from: data)
        #expect(QueueHold.resolve(entry: row, child: child(errorCode: "MODEL_NOT_FOUND")) == .prose("", retryable: true))
    }
}

@Test func nonretryableManifestHintCannotOfferDownload() throws {
    let row = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"h3","state":"held","model":"h3","retryable":false,"model_download":{"model":"h3"}}"#.utf8))
    #expect(QueueHold.resolve(entry: row, child: nil) == .prose("", retryable: false))
}

@Test func malformedPresentHintKeepsTheRowButDisablesLegacyRecovery() throws {
    for hint: Any in [NSNull(), "wrong", 4, ["model": NSNull()], ["model": 4], [:]] {
        let data = try JSONSerialization.data(withJSONObject: ["id": "job", "state": "held", "model": "h3", "retryable": true,
                                                               "held_reason": "Keep this diagnostic", "model_download": hint])
        let row = try MoldJSON.decoder.decode(QueueEntry.self, from: data)
        #expect(row.heldReason == "Keep this diagnostic")
        #expect(QueueHold.resolve(entry: row, child: child(errorCode: "MODEL_NOT_FOUND")) == .prose("Keep this diagnostic", retryable: true))
    }
}

@Test func newHintRequiresExplicitRetryPermission() throws {
    let row = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"h3","state":"held","model":"h3","model_download":{"model":"h3"}}"#.utf8))
    #expect(QueueHold.resolve(entry: row, child: child(errorCode: "MODEL_NOT_FOUND")) == .prose("", retryable: true))
}

@Test func hintedQueueEntryRoundTripPreservesExistingFields() throws {
    let data = Data(#"{"id":"job","model":"h3","model_download":{"model":"h3"},"state":"held","retryable":true,"error_detail":"adapter diagnostic","held_reason":"Adapter missing","batch_id":"batch","client_batch_id":"client","dispatch_attempts":2,"gpu":1,"target_gpu":2,"batch_index":3,"explicitly_paused":false,"replayed":true,"seed_pinned":true,"model_display_name":"H3","position":4,"started_at_unix_ms":123,"durable":true}"#.utf8)
    let row = try MoldJSON.decoder.decode(QueueEntry.self, from: data)
    #expect(try MoldJSON.decoder.decode(QueueEntry.self, from: MoldJSON.encoder.encode(row)) == row)
    #expect(row.errorDetail == "adapter diagnostic")
    #expect(row.modelDownload?.model == "h3")
}
