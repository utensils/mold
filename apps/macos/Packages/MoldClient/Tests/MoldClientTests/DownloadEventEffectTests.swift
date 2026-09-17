import Foundation
import Testing

@testable import MoldClient

/// One frame of `GET /api/downloads/stream`, decoded the way the wire sends
/// it -- internally tagged, `{"type": …}` (`types.rs:13068-13070`).
private func frame(_ json: String) throws -> DownloadEvent {
    try MoldJSON.decoder.decode(DownloadEvent.self, from: Data(json.utf8))
}

/// **Fails today**: there is no `effect`, and the store's own branch is "has
/// an id and is not terminal" -- which is true of `catalog_ready`, whose `id`
/// is the CATALOG entry's, not a job's.
@Test func aCatalogReadyFrameSaysNothingAboutAnyRow() throws {
    // `downloads.rs:521-523`, after the group's last terminal frame. The id
    // is `hf:owner/repo` -- there is no job by that name, and no further
    // frame for it will ever arrive.
    let event = try frame(#"{"type":"catalog_ready","id":"hf:black-forest-labs/FLUX.1-dev","ok":true}"#)
    #expect(event.effect == .ignore)
}

/// `enqueued` and `started` are the only deltas that may CREATE a row --
/// `downloads.ts:63-95`.
@Test func onlyEnqueuedAndStartedIntroduceARow() throws {
    #expect(try frame(#"{"type":"enqueued","id":"j1","model":"flux-dev:q4","position":0}"#).effect == .introduce)
    #expect(try frame(#"{"type":"started","id":"j1","files_total":3,"bytes_total":99}"#).effect == .introduce)
}

/// A `progress` naming a job the client has not seen is not a new job:
/// `downloads.ts:96-97` returns the state untouched.
@Test func progressAndFileDoneOnlyUpdateWhatIsAlreadyKnown() throws {
    #expect(try frame(#"{"type":"progress","id":"j1","files_done":1,"bytes_done":10}"#).effect == .update)
    #expect(try frame(#"{"type":"file_done","id":"j1","filename":"model.safetensors"}"#).effect == .update)
}

@Test func theThreeTerminalFramesSettleTheirRow() throws {
    #expect(try frame(#"{"type":"job_done","id":"j1","model":"flux-dev:q4"}"#).effect == .settle)
    #expect(try frame(#"{"type":"job_failed","id":"j1","error":"disk full"}"#).effect == .settle)
    #expect(try frame(#"{"type":"job_cancelled","id":"j1"}"#).effect == .settle)
}

/// `dequeued` rides just behind the `job_cancelled` that already settled the
/// row (`downloads.rs:586-591`), so it removes rather than records.
@Test func dequeuedForgetsTheRowWithoutRecordingAnOutcome() throws {
    #expect(try frame(#"{"type":"dequeued","id":"j1"}"#).effect == .forget)
}

/// A frame a newer server adds is not a row this build may invent.
@Test func anUnknownFrameIsIgnoredRatherThanGuessedAt() throws {
    #expect(try frame(#"{"type":"something_new","id":"j1"}"#).effect == .ignore)
}
