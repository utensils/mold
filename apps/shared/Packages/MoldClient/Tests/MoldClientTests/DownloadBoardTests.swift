import Foundation
import Testing

@testable import MoldClient

/// The download reducer both apps draw from: a row appears on
/// `enqueued`/`started`, moves on `progress`, and settles into a finished job
/// that remembers the last figures it had.
private func frame(_ json: String) throws -> DownloadEvent {
    try MoldJSON.decoder.decode(DownloadEvent.self, from: Data(json.utf8))
}

@Test func aStartedFrameIntroducesARowAndProgressMovesIt() throws {
    var board: [String: DownloadProgress] = [:]
    _ = DownloadBoard.apply(try frame(#"{"type":"started","id":"j1","model":"flux-dev:q4","bytes_total":100}"#), to: &board)
    _ = DownloadBoard.apply(try frame(#"{"type":"progress","id":"j1","bytes_done":40,"bytes_total":100}"#), to: &board)
    #expect(board["j1"]?.model == "flux-dev:q4")
    #expect(board["j1"]?.bytesDone == 40)
    #expect(board["j1"]?.fraction == 0.4)
}

@Test func progressForAnUnknownJobCreatesNothing() throws {
    var board: [String: DownloadProgress] = [:]
    let settled = DownloadBoard.apply(try frame(#"{"type":"progress","id":"j9","bytes_done":1}"#), to: &board)
    #expect(board.isEmpty)
    #expect(settled == nil)
}

@Test func aTerminalFrameSettlesWithTheLastKnownFigures() throws {
    var board: [String: DownloadProgress] = ["j1": DownloadProgress(model: "sdxl", bytesDone: 70, bytesTotal: 100)]
    let settled = DownloadBoard.apply(try frame(#"{"type":"job_failed","id":"j1","error":"disk full"}"#), to: &board)
    #expect(board.isEmpty)
    #expect(settled?.model == "sdxl")
    #expect(settled?.status == .failed)
    #expect(settled?.bytesDone == 70)
    #expect(settled?.error == "disk full")
}

@Test func dequeuedDropsTheRowWithoutAnOutcome() throws {
    var board: [String: DownloadProgress] = ["j1": DownloadProgress(model: "sdxl")]
    #expect(DownloadBoard.apply(try frame(#"{"type":"dequeued","id":"j1"}"#), to: &board) == nil)
    #expect(board.isEmpty)
}

@Test func aListingBecomesTheBoard() throws {
    let listing = DownloadsListing(activeJobs: [DownloadJob(id: "a", model: "m", status: .active, bytesDone: 5, bytesTotal: 10)],
                                   active: nil, queued: [DownloadJob(id: "q", model: "n", status: .queued)], history: [])
    let board = DownloadBoard.adopt(listing)
    #expect(board["a"]?.fraction == 0.5)
    #expect(board["q"]?.fraction == nil)
}

@Test func byteSentenceReadsLikeTheDesign() {
    let row = DownloadProgress(model: "m", bytesDone: 2_100_000_000, bytesTotal: 11_800_000_000)
    #expect(row.sentence(bytesPerSecond: 42_000_000) == "2.1 / 11.8 GB · 42 MB/s")
    #expect(DownloadProgress(model: "m").sentence(bytesPerSecond: nil) == "Starting…")
}
