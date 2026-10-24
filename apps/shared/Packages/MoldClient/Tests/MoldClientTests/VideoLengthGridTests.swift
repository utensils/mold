import Foundation
import Testing
@testable import MoldClient

struct VideoLengthGridTests {
    static func h3Limits() throws -> ChainLimits {
        try MoldJSON.decoder.decode(ChainLimits.self, from: Data("""
        {"model":"minimax-h3","frames_per_clip_cap":345,"fps":24,"frames_per_clip_recommended":124,
         "max_stages":16,"max_total_frames":5000,"supports_sequence":true,
         "authoring":{"motion_tail_frames":1,"carry_mode":"rgb","min_frames":107,"frame_step":17,"frame_offset":5}}
        """.utf8))
    }

    static func h3Temporal() throws -> TemporalProfile {
        try MoldJSON.decoder.decode(TemporalProfile.self, from: Data("""
        {"frames":{"default":124,"min":107,"max":345,"step":17,"mode":"adjustable"},
         "frame_offset":5,"fps":{"mode":"fixed","value":24}}
        """.utf8))
    }

    @Test func h3SnapsToItsServerGridFromTheMinimumToOneGeneration() throws {
        let grid = VideoLengthGrid.resolve(temporal: try Self.h3Temporal(), fps: nil, limits: try Self.h3Limits())
        #expect(grid.fps == 24)
        #expect(grid.minFrames == 107 && grid.maxFrames == 345)
        #expect(grid.frames.first == 107 && grid.frames.last == 345)
        #expect(grid.frames.allSatisfy { ($0 - 5) % 17 == 0 })
        #expect(grid.label(frames: 345) == "14.4 s")
        #expect(grid.label(frames: 107) == "4.5 s")
        #expect(grid.snap(200) == 192 && grid.snap(201) == 209)
        #expect(grid.snap(1) == 107)
        #expect(grid.snap(10_000) == 345)
        #expect(grid.seconds(frames: 345) == "14.375")
        #expect(grid.nearestFrames(milliseconds: 14_000) == 328)
    }

    @Test func theLongestAlwaysOneClipSceneSpendsTheOverlap() throws {
        let grid = VideoLengthGrid.resolve(temporal: try Self.h3Temporal(), fps: nil, limits: try Self.h3Limits())
        // (345 - 1) / 24 = 14.33 s, floored to a tenth.
        #expect(grid.sceneMaxTenths == 143)
        // Duration trims to the exact length, so a scene may be as short as 0.1 s.
        #expect(grid.sceneMinTenths == 1)
        #expect(grid.clampSceneTenths(500) == 143)
        #expect(grid.clampSceneTenths(0) == 1)
        // The single-prompt slider runs 1.0 s to one generation, floored to a tenth.
        #expect(grid.singleMinTenths == 10 && grid.singleMaxTenths == 143)
        #expect(VideoLengthGrid.label(milliseconds: 14_000) == "14.0 s")
        #expect(VideoLengthGrid.label(milliseconds: 14_375) == "14.4 s")
    }

    /// Duration admission splits on the PLAN's clip window, which is the
    /// recommended one for every family but H3 (LTX-Video renders 25-frame
    /// clips), never the chain limits' single-request cap.
    @Test func durationAuthoringReadsThePlanBeforeChainLimits() throws {
        let plan = try MoldJSON.decoder.decode(VideoJobPlan.self, from: Data("""
        {"validation_revision":"","model":"ltx-video","width":768,"height":512,"fps":24,"single_clip_max_frames":25,
         "overlap_frames":8,"join_mode":"blended_shots","clip_count":"4","part_count":"1","added_frames":"96",
         "source_frames":"0","total_frames":"96","added_duration_ms":"4000","total_duration_ms":"4000","warnings":[]}
        """.utf8))
        let duration = VideoLengthGrid.duration(temporal: try Self.h3Temporal(), fps: nil, limits: try Self.h3Limits(), plan: plan)
        #expect(duration.maxFrames == 25 && duration.overlapFrames == 8)
        #expect(duration.singleMaxTenths == 10 && duration.sceneMaxTenths == 7)
        let early = VideoLengthGrid.duration(temporal: try Self.h3Temporal(), fps: nil, limits: try Self.h3Limits(), plan: nil)
        #expect(early.maxFrames == 345 && early.overlapFrames == 1, "limits stand in until a plan exists")
    }

    @Test func limitsOutrankAPlanWhichOutranksTheRecipeAndAnOrdinaryCeilingOutranksAll() throws {
        let plan = try MoldJSON.decoder.decode(VideoJobPlan.self, from: Data("""
        {"validation_revision":"","model":"h3","width":768,"height":768,"fps":24,"single_clip_max_frames":277,
         "overlap_frames":9,"join_mode":"continuous","clip_count":"1","part_count":"1","added_frames":"200",
         "source_frames":"0","total_frames":"200","added_duration_ms":"8000","total_duration_ms":"8000","warnings":[]}
        """.utf8))
        let limited = VideoLengthGrid.resolve(temporal: try Self.h3Temporal(), fps: nil, limits: try Self.h3Limits(), plan: plan)
        #expect(limited.maxFrames == 345 && limited.overlapFrames == 1)
        let planned = VideoLengthGrid.resolve(temporal: try Self.h3Temporal(), fps: nil, limits: nil, plan: plan)
        #expect(planned.maxFrames == 277 && planned.overlapFrames == 9)
        let ordinary = VideoLengthGrid.resolve(temporal: try Self.h3Temporal(), fps: nil, limits: nil, ceiling: 260)
        #expect(ordinary.maxFrames == 260)
        #expect(ordinary.frames.last == 260)
    }

    @Test func withoutLimitsTheRecipeGridStands() throws {
        let temporal = try MoldJSON.decoder.decode(TemporalProfile.self, from: Data("""
        {"frames":{"default":97,"min":9,"max":257,"step":8,"mode":"adjustable"},"frame_offset":1,
         "fps":{"mode":"adjustable","default":24,"min":12,"max":30,"step":1}}
        """.utf8))
        let grid = VideoLengthGrid.resolve(temporal: temporal, fps: 30, limits: nil)
        #expect(grid.fps == 30 && grid.step == 8 && grid.offset == 1)
        #expect(grid.minFrames == 9 && grid.maxFrames == 257)
        // Blended work spends eight frames when the host names no tail.
        #expect(grid.overlapFrames == 8)
    }

    @Test func totalsNameTheScenesWithTheRightNoun() {
        #expect(VideoLengthGrid.total(milliseconds: [12_000, 14_300, 12_300]) == "Total 38.6 s · 3 scenes")
        #expect(VideoLengthGrid.total(milliseconds: [14_000]) == "Total 14 s · 1 scene")
        #expect(VideoLengthGrid.tenths(seconds: "14.3") == 143)
        #expect(VideoLengthGrid.tenths(seconds: "14.375") == 144)
        #expect(VideoLengthGrid.tenths(seconds: "bad") == nil)
        #expect(VideoLengthGrid.seconds(tenths: 143) == "14.3")
        #expect(VideoLengthGrid.seconds(tenths: 140) == "14")
    }
}
