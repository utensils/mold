import Foundation
import Testing

@testable import MoldClient

/// Reading what a machine says as it happens.
///
/// Every frame below is one this app actually received from a live mold, not
/// one written from the documentation. The two disagreed in a way that matters
/// and is pinned here: the SSE `event:` name is the literal word "event" for
/// everything except the opening `authority` frame, and the real type is the
/// `type` field INSIDE the payload.
@Suite struct MoldEventSuite {

    @Test func theStreamOpensBySayingWhichMachineThisIs() {
        let event = MoldEvent(name: "authority",
                              data: #"{"instance_id":"ff00bea2-a8fc-4ffa-80c6-f5f80cfa5580"}"#)
        #expect(event == .authority(instanceID: "ff00bea2-a8fc-4ffa-80c6-f5f80cfa5580"))
    }

    /// The trap: routing on the SSE name alone finds "event" for every gallery
    /// change and decodes none of them.
    @Test func theTypeIsInsideThePayloadNotTheFrameName() {
        let event = MoldEvent(
            name: "event",
            data: #"{"type":"gallery_updated","filename":"a.png"}"#)
        #expect(event == .gallery(.updated(filename: "a.png", row: nil)))
    }

    @Test func aRowRidesAlongWhenTheMachineSendsOne() {
        let data = """
            {"type":"gallery_added","filename":"a.png","image":{"filename":"a.png",\
            "metadata":{"prompt":"owls"},"timestamp":1000,"favorite":true}}
            """
        guard case let .gallery(.added(filename, row)) = MoldEvent(name: "event", data: data)
        else { Issue.record("not a gallery add"); return }
        #expect(filename == "a.png")
        // Present means insert without asking again; absent means go and read.
        #expect(row?.isFavorite == true)
    }

    @Test func theOtherGalleryVerbsDecode() {
        #expect(MoldEvent(name: "event", data: #"{"type":"gallery_removed","filename":"a.png"}"#)
            == .gallery(.removed(filename: "a.png")))
        #expect(MoldEvent(name: "event", data: #"{"type":"gallery_trashed","filename":"a.png"}"#)
            == .gallery(.trashed(filename: "a.png")))
        #expect(MoldEvent(name: "event", data: #"{"type":"gallery_restored","filename":"a.png"}"#)
            == .gallery(.restored(filename: "a.png", row: nil)))
        #expect(MoldEvent(name: "event", data: #"{"type":"gallery_collections_changed"}"#)
            == .gallery(.collectionsChanged))
    }

    /// The buffer overran and this client missed deltas. The only honest answer
    /// is to read the listings again -- carrying on would leave the screen
    /// quietly wrong with no way to notice.
    @Test func aLaggedStreamAsksToBeRepaired() {
        #expect(MoldEvent(name: "resync_required", data: #"{"missed_events":12}"#)
            == .resyncRequired)
    }

    /// Unknown tags are IGNORED, never errors. mold adds events, and a client
    /// that treats a new one as a failure breaks on a server upgrade.
    /// `chain_job_started` is a REAL tag mold sends and a deliberate case of
    /// this, not a made-up one -- see `MoldEvent.init(name:data:)`'s
    /// `default:` arm (M6 decision, `types.rs:13209-13214`).
    @Test func anUnknownEventIsIgnoredRatherThanFailing() {
        #expect(MoldEvent(name: "event", data: #"{"type":"chain_job_started","id":"c1","model":"m"}"#) == nil)
        #expect(MoldEvent(name: "event", data: #"{"type":"something_new_in_0_30"}"#) == nil)
        #expect(MoldEvent(name: "event", data: "not json at all") == nil)
        #expect(MoldEvent(name: nil, data: "") == nil)
    }

    /// Carries nothing: the frame names three of `DeviceInfo`'s fourteen
    /// fields and the pane draws eight, so this is an invalidation, not a
    /// patch -- go and read `/api/devices`.
    @Test func aDeviceStateChangedFrameIsAnInvalidation() {
        let data = #"{"type":"device_state_changed","device_id":"cuda:9ffc81c539446490bfd9f68366f98226","desired_enabled":false,"admin_state":"draining"}"#
        #expect(MoldEvent(name: "event", data: data) == .deviceStateChanged)
    }

    /// The keep-alive comments mold sends every 15 seconds are not events.
    @Test func aKeepAliveIsNotAnEvent() {
        var parser = SSEParser()
        #expect(parser.consume(line: ": keep-alive") == nil)
    }
}

/// The live payload, so a gate that is meant to open actually opens.
@Suite struct EventsCapabilitySuite {
    @Test func theCapturedWorkstationPayloadAdvertisesEvents() throws {
        let capabilities = try MoldJSON.decoder.decode(
            Capabilities.self, from: RepoFixtures.fixture("capabilities.json"))
        #expect(capabilities.events?.available == true)
        #expect(capabilities.hasEvents)
    }
}
