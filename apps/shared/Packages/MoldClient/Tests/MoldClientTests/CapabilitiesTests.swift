import Foundation
import Testing

@testable import MoldClient

// `Fixtures/capabilities.json` is a live `GET /api/capabilities` from workstation
// (mold 0.29.0), with the private-H3 and mesh blocks removed because this app
// reads neither. It is here so the decoder is pinned against a real payload
// rather than against what we believed the shape was.

private func live() throws -> Capabilities {
    try MoldJSON.decoder.decode(Capabilities.self, from: RepoFixtures.fixture("capabilities.json"))
}

/// An empty object is every field absent -- the shape an ancient host sends.
private let ancient = try! MoldJSON.decoder.decode(Capabilities.self, from: Data("{}".utf8))

@Test func readsAHostThatCanDoEverything() throws {
    let caps = try live()
    #expect(caps.generates)
    #expect(caps.maxBatchOutputs == 64)
    #expect(caps.hasDurableQueue)
    #expect(caps.canReorderQueue)
    #expect(caps.canCancelAllQueued)
    #expect(caps.canOrganize)
    #expect(caps.canBulkMutate)
    #expect(caps.supportsConditionalGallery)
    #expect(caps.supportsGalleryRowEvents)
    #expect(caps.trashRetentionDays == 30)
    #expect(caps.hasEvents)
    #expect(caps.canBrowsePeers)
    #expect(caps.catalogFamilies.contains("flux"))
    #expect(caps.devices?.lifecycle == true)
    #expect(caps.identity?.maxPhotos == 4)
    #expect(caps.expand?.remix == true)
    #expect(caps.expand?.model == "qwen3-expand:q8")
}

// MARK: - What absence means, field by field

/// Definitively NO. mold only advertises this block when the identity runtime
/// is actually available, so a host that says nothing cannot do it.
@Test func identityAbsenceIsARefusal() {
    #expect(ancient.identity == nil)
    #expect(ancient.supportsIdentity == false)
}

/// UNKNOWN, not no. An older host expanded prompts without advertising it, so
/// the app offers the control and lets the request answer.
@Test func expandAbsenceIsUnknownSoTheAppMayStillTry() {
    #expect(ancient.expand == nil)
    #expect(ancient.mayExpandPrompts)
}

/// A host whose expander model is not installed still advertises the block,
/// and names the model to offer to pull rather than making the client
/// hard-code `qwen3-expand`.
@Test func anUninstalledExpanderNamesWhatToPull() throws {
    let json = Data("""
    {"expand":{"configured":true,"model_present":false,"backend":"local",\
    "remix":false,"model":"qwen3-expand:q8"}}
    """.utf8)
    let caps = try MoldJSON.decoder.decode(Capabilities.self, from: json)
    #expect(caps.mayExpandPrompts)
    #expect(caps.expanderModelToPull == "qwen3-expand:q8")
    #expect(caps.expand?.remix == false)
}

/// An API-backed expander has no local model, so there is nothing to pull and
/// `model_present` is not an answer about installation at all.
@Test func anApiExpanderHasNothingToPull() throws {
    let json = Data(#"{"expand":{"configured":true,"backend":"api","remix":true}}"#.utf8)
    let caps = try MoldJSON.decoder.decode(Capabilities.self, from: json)
    #expect(caps.mayExpandPrompts)
    #expect(caps.expanderModelToPull == nil)
}

/// An OLDER HOST, not a refusal: conditional GETs and row events were added
/// after the gallery was, so absence means fall back to listing.
@Test func galleryEfficiencyAbsencesMeanAnOlderHost() {
    #expect(ancient.supportsConditionalGallery == false)
    #expect(ancient.supportsGalleryRowEvents == false)
    #expect(ancient.hasEvents == false)
}

/// The presence of the batch number is how a client knows the host generates
/// at all -- there is no separate boolean.
@Test func aHostThatDoesNotGenerateSaysSoByOmittingTheBatchNumber() {
    #expect(ancient.generates == false)
    #expect(ancient.maxBatchOutputs == 1)
}

/// Trash is off unless advertised on, and zero days means keep forever --
/// which must never be shown as "purged in 0 days".
@Test func trashRetentionDistinguishesOffFromForever() throws {
    #expect(ancient.trashEnabled == false)
    #expect(ancient.trashRetentionDays == nil)

    let forever = try MoldJSON.decoder.decode(
        Capabilities.self,
        from: Data(#"{"gallery":{"trash":{"enabled":true,"retention_days":0}}}"#.utf8)
    )
    #expect(forever.trashEnabled)
    #expect(forever.trashRetentionDays == nil)
}

/// Device lifecycle and restart-enable are different powers. A runtime that
/// cannot enforce a live change must never persist one, but may still be able
/// to arrange it for the next restart.
@Test func deviceLifecycleAndRestartEnableAreSeparatePowers() throws {
    let json = Data(#"{"devices":{"available":true,"lifecycle":false,"restart_enable":true}}"#.utf8)
    let caps = try MoldJSON.decoder.decode(Capabilities.self, from: json)
    #expect(caps.devices?.available == true)
    #expect(caps.devices?.lifecycle == false)
    #expect(caps.devices?.restartEnable == true)

    #expect(ancient.devices == nil)
    #expect(ancient.canSeeDevices == false)
}

/// A live device change needs BOTH `devices.lifecycle` (the route exists) and
/// `dispatch.v2Authoritative` (the runtime will honour it) -- one without the
/// other is a persisted change nothing will ever enforce.
@Test func dispatchIsAuthoritativeOnlyUnderSchedulerV2() throws {
    let caps = try live()
    #expect(caps.dispatch?.activeMode == "v2")
    #expect(caps.dispatchIsAuthoritative)
    #expect(ancient.dispatch == nil)
    #expect(ancient.dispatchIsAuthoritative == false)
}

/// Reference uploads exist on this host but are unavailable, because the
/// protocol needs API-key auth and workstation is keyless. Advertised-but-off is not
/// the same as absent, and neither is a reason to fail a small reference.
@Test func referenceUploadsCanBeAdvertisedAndStillUnavailable() throws {
    let caps = try live()
    #expect(caps.referenceUploads != nil)
    #expect(caps.canUploadLargeReferences == false)
    #expect(ancient.canUploadLargeReferences == false)
}

/// An unknown enum value from a newer host degrades rather than failing the
/// whole decode -- a capability block is the last thing that should be lost
/// over one unrecognized string.
@Test func aBackendThisBuildHasNeverHeardOfDoesNotFailTheDecode() throws {
    let json = Data(#"{"expand":{"configured":true,"backend":"something-new"}}"#.utf8)
    let caps = try MoldJSON.decoder.decode(Capabilities.self, from: json)
    #expect(caps.expand?.backend == .unknown)
}
