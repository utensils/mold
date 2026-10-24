import Foundation
import Testing
@testable import MoldClient

struct VideoJobRowTests {
    private struct Case: Decodable {
        struct Input: Decodable {
            let state: VideoJobState
            let progressPhase: VideoJobProgressPhase?
            let completedClips: String
            let clipCount: String
            let totalDurationMs: String
            let canCancel: Bool
            let current: Bool
        }
        struct Row: Decodable {
            let caption: String
            let detail: String
            let symbol: VideoJobRow.Symbol
            let progress: Double?
        }
        let input: Input
        let row: Row
    }

    @Test func rowsMatchTheCrossSurfaceContract() throws {
        let root = try #require(RepoFixtures.repoRoot)
        let data = try Data(contentsOf: root.appendingPathComponent("docs/contracts/video-job-rows.json"))
        let cases = try MoldJSON.decoder.decode([Case].self, from: data)
        #expect(cases.count >= 10)
        for fixture in cases {
            let input = fixture.input
            let completed = try #require(UInt64(input.completedClips))
            let clips = try #require(UInt64(input.clipCount))
            let total = try #require(UInt64(input.totalDurationMs))
            let row = VideoJobRow(VideoJobRow.Input(
                state: input.state, progressPhase: input.progressPhase, completedClips: completed,
                clipCount: clips, totalDurationMs: total, canCancel: input.canCancel, current: input.current))
            #expect(row.caption == fixture.row.caption, "\(input)")
            #expect(row.detail == fixture.row.detail, "\(input)")
            #expect(row.symbol == fixture.row.symbol, "\(input)")
            #expect(row.progress == fixture.row.progress, "\(input)")
        }
    }

    @Test func aSingleClipNeverSaysClips() {
        for state in [VideoJobState.running, .paused, .queued, .interrupted] {
            let row = VideoJobRow(.init(state: state, progressPhase: nil, completedClips: 0, clipCount: 1,
                                        totalDurationMs: 5_000, canCancel: true, current: true))
            #expect(!row.caption.contains("clips") && !row.detail.contains("clips"), "\(row)")
        }
    }

    @Test func progressPhaseIsAdditiveAndOpen() throws {
        let base = #"{"id":"p","state":"running","plan":\#(VideoJobContractTests.plan()),"completed_clips":"0","active_child_id":null,"output_filename":null,"error":null,"error_detail":null,"created_at_ms":1,"updated_at_ms":2,"can_cancel":true,"can_resume":false,"can_delete":false"#
        let older = try MoldJSON.decoder.decode(VideoJob.self, from: Data((base + "}").utf8))
        #expect(older.progressPhase == nil)
        let joining = try MoldJSON.decoder.decode(VideoJob.self, from: Data((base + #","progress_phase":"joining"}"#).utf8))
        #expect(joining.progressPhase == .joining)
        let future = try MoldJSON.decoder.decode(VideoJob.self, from: Data((base + #","progress_phase":"uploading"}"#).utf8))
        #expect(future.progressPhase == .unknown)
        // An unknown phase reads like an older server's bare `running`.
        #expect(VideoJobRow(future, current: true).caption == "Rendering clip 1 of 4")
        #expect(VideoJobRow(joining, current: true).caption == "Joining clips")
    }

    @Test func liveParentsAreTheOnesStillGoing() {
        #expect([VideoJobState.queued, .running, .paused].allSatisfy { $0.isLive })
        #expect(![VideoJobState.completed, .failed, .interrupted, .cancelled, .unknown].contains { $0.isLive })
    }

    @Test func sequenceJobsUseTheSameWording() {
        func row(_ state: ChainJobState, phase: String? = nil, stage: Int = 0, of count: Int = 3,
                 cancelling: Bool = false, current: Bool = true) -> VideoJobRow {
            VideoJobRow(chain: state, executionPhase: phase, currentStage: stage, stageCount: count,
                        cancelling: cancelling, current: current)
        }
        #expect(row(.queued).caption == "Waiting")
        #expect(row(.running, phase: "queued").caption == "Waiting")
        #expect(row(.running, phase: "rendering", stage: 1).caption == "Rendering clip 2 of 3")
        #expect(row(.running, phase: "rendering", stage: 1).progress == 1.0 / 3)
        #expect(row(.running, phase: "finalizing", stage: 2).caption == "Joining clips")
        #expect(row(.running, stage: 1, cancelling: true).caption == "Stopping")
        #expect(row(.paused, stage: 1).caption == "Paused · 1 of 3 clips done")
        #expect(row(.completed, stage: 2).caption == "Done")
        #expect(row(.cancelled).symbol == .stopped)
        #expect(row(.running, stage: 0, of: 1).caption == "Rendering")
        #expect(row(.running, stage: 0, of: 1).detail == "1 clip")
        #expect(row(.running, stage: 0).detail == "3 clips")
        #expect(row(.running, phase: "rendering", stage: 1, current: false).caption == "Last known: Rendering clip 2 of 3")
    }

    @Test func countsUseTheRightNoun() {
        #expect(Plural.count(1, of: "clip") == "1 clip")
        #expect(Plural.count(0, of: "clip") == "0 clips")
        #expect(Plural.count(3, of: "scene") == "3 scenes")
        #expect(Plural.count(2, of: "match", plural: "matches") == "2 matches")
    }
}
