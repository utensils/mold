import Foundation
import Testing

@testable import MoldClient

private func mixed() throws -> QueueListing {
    try MoldJSON.decoder.decode(QueueListing.self, from: RepoFixtures.fixture("queue-mixed.json"))
}

private func row(_ id: String, in listing: QueueListing) throws -> QueueEntry {
    try #require(listing.entries.first { $0.id == id })
}

/// **Fails today**: `QueueState` has no `.queued` case, so `"queued"` --
/// the string every ordinary waiting row actually carries -- falls through
/// `OpenWireEnum` to `.unknown`, `isLive` answers `false`, and no live row
/// can be cancelled. This is the milestone's first test and the whole reason
/// S1 exists (design M6 fact 1).
@Test func aWaitingRowIsQueuedAndCanBeActedOn() throws {
    let entry = try row("q-standalone", in: try mixed())
    #expect(entry.state == .queued)
    #expect(entry.state.isLive)
    #expect(entry.state.isReorderable)
}

/// `accepted` is `BatchChildState`'s word for the same idea on a DIFFERENT
/// endpoint. Pins decision 1 in both directions: `/api/queue` never sends
/// it, and `/api/generation-batches/status` never sends `queued`.
@Test func theBatchEndpointsWordForWaitingIsNotThisEndpoints() throws {
    let queueRow = try MoldJSON.decoder.decode(
        QueueEntry.self, from: Data(#"{"id":"x","state":"accepted"}"#.utf8))
    #expect(queueRow.state == .unknown)

    let child = try MoldJSON.decoder.decode(
        BatchChild.self, from: Data(#"{"index":0,"job_id":"j","state":"accepted"}"#.utf8))
    #expect(child.state == .accepted)
}

@Test func aRestartPauseAndADeliberateOneReadDifferently() throws {
    let entry = try row("p-explicit", in: try mixed())
    #expect(entry.explicitlyPaused == true)
    #expect(entry.waitDescription == "Paused")
}

@Test func aHostThatCannotTellThemApartSaysThePlainerThing() throws {
    let entry = try row("p-restart", in: try mixed())
    #expect(entry.explicitlyPaused == nil)
    #expect(entry.waitDescription == "Paused after restart")
}

@Test func aRunningRowNamesItsGpuAndAQueuedOneNamesNoLane() throws {
    let listing = try mixed()
    let running = try row("r-running", in: listing)
    #expect(running.gpu == 1)

    let queued = try row("q-standalone", in: listing)
    #expect(queued.gpu == nil)
    // Absent means Auto -- not a missing answer.
    #expect(queued.targetGpu == nil)
}

@Test func aRowThatIsNotABatchChildHasNoRetryAuthority() throws {
    let listing = try mixed()
    let standalone = try row("q-standalone", in: listing)
    #expect(standalone.authority(instanceId: "i1") == nil)

    let child = try row("b-child-1", in: listing)
    let authority = try #require(child.authority(instanceId: "i1"))
    #expect(authority == QueueAuthority(
        instanceId: "i1", batchId: "batch-mixed", clientBatchId: "client-mixed", jobId: "b-child-1"))
}

/// The PATCH body has exactly one key, so a `position`-only move can never
/// clobber a lane pin -- `QueuePatchRequest`'s other two fields are a double
/// `Option` (`routes.rs:7235-7243`), and any key beyond `position` risks
/// resetting one of them to Auto.
@Test func aReorderSendsAPositionAndNothingElse() throws {
    let data = try MoldJSON.encoder.encode(QueueReorderPatch(position: 3))
    let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(Array(object.keys) == ["position"])
    #expect(object["position"] as? Int == 3)
}

/// The bulk status body names batches only -- `GenerationBatchStatusRequest`
/// also has a `client_batch_ids` field, and this backend never populates it
/// (`types.rs:11056-11062`).
@Test func aBulkStatusCallNamesOnlyBatchIds() throws {
    let data = try MoldJSON.encoder.encode(GenerationBatchStatusQuery(batchIds: ["b1", "b2"]))
    let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(Array(object.keys) == ["batch_ids"])
    #expect(object["batch_ids"] as? [String] == ["b1", "b2"])
}

@Test func aBatchTheMachineDoesNotKnowComesBackAsMissing() throws {
    let listing = try MoldJSON.decoder.decode(
        BatchStatusListing.self, from: RepoFixtures.fixture("batch-status.json"))
    #expect(listing.batches.map(\.id) == ["batch-mixed"])
    #expect(listing.missing.batchIds == ["batch-unknown"])
    #expect(listing.missing.clientBatchIds.isEmpty)
}

/// **Fails today**: `MoldEvent`'s decode switch has no arms for any of these
/// tags, so every one returns `nil` (design M6 fact 10).
@Test func everyQueueFrameThisMilestoneNeedsDecodes() {
    #expect(MoldEvent(name: "event", data: #"{"type":"job_queued","id":"j1","model":"flux-dev:q8"}"#)
        == .job(.queued(id: "j1", model: "flux-dev:q8")))
    #expect(MoldEvent(
        name: "event", data: #"{"type":"job_started","id":"j1","model":"flux-dev:q8","gpu":1}"#)
        == .job(.started(id: "j1", model: "flux-dev:q8", gpu: 1)))
    // A single-GPU host sends no `gpu` key at all.
    #expect(MoldEvent(name: "event", data: #"{"type":"job_started","id":"j1","model":"flux-dev:q8"}"#)
        == .job(.started(id: "j1", model: "flux-dev:q8", gpu: nil)))
    #expect(MoldEvent(name: "event", data: #"{"type":"job_ended","id":"j1"}"#) == .job(.ended(id: "j1")))
    #expect(MoldEvent(name: "event", data: #"{"type":"job_state_committed","id":"j1"}"#)
        == .job(.stateCommitted(id: "j1")))
    #expect(MoldEvent(name: "event", data: #"{"type":"generation_states_committed"}"#)
        == .job(.statesCommitted))
    #expect(MoldEvent(name: "event", data: #"{"type":"queue_paused"}"#) == .queue(.paused))
    #expect(MoldEvent(name: "event", data: #"{"type":"queue_resumed"}"#) == .queue(.resumed))
    #expect(MoldEvent(name: "event", data: #"{"type":"queue_plan_changed","plan":{}}"#)
        == .queue(.planChanged))
}

/// `chain_job_queued` decodes to nothing, deliberately -- old clients ignore
/// unknown `type` tags, which is exactly why a chain job never inherits
/// print-queue affordances it does not support (`types.rs:13209-13214`).
@Test func aChainFrameIsStillIgnored() {
    #expect(MoldEvent(
        name: "event", data: #"{"type":"chain_job_queued","id":"c1","model":"m","stage_count":3}"#)
        == nil)
}

@Test func anUnknownTagIsStillIgnored() {
    #expect(MoldEvent(name: "event", data: #"{"type":"something_new_in_0_30"}"#) == nil)
}
