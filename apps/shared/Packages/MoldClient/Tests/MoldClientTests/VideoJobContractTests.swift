import Foundation
import Testing
@testable import MoldClient

struct VideoJobContractTests {
    @Test func requestUsesLiteralPromptsAndDecimalDurationWithoutNarrowingSeed() throws {
        var generation = GenerateRequest(prompt: #"one\new"# + "\r\ntwo", model: "model", width: 768, height: 768,
                                         steps: 2, guidance: 1, seed: UInt64.max)
        generation.negativePrompt = #"blur\n"#
        let request = VideoJobRequest(generation: generation, durationMs: UInt64.max,
            source: .init(filename: "original.mp4", expectedArchiveIdentitySha256: String(repeating: "a", count: 64)),
            scenes: [.init(prompt: #"scene\n"#, durationMs: UInt64.max)])
        let object = try #require(JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(request)) as? [String: Any])
        #expect(object["duration_ms"] as? String == String(UInt64.max))
        let rawGeneration = try #require(object["generation"] as? [String: Any])
        #expect(rawGeneration["prompt"] as? String == generation.prompt)
        #expect(rawGeneration["negative_prompt"] as? String == generation.negativePrompt)
        #expect((rawGeneration["seed"] as? NSNumber)?.uint64Value == UInt64.max)
        #expect(object["validation_revision"] == nil)
        let scenes = try #require(object["scenes"] as? [[String: Any]])
        #expect(scenes[0]["duration_ms"] as? String == String(UInt64.max))
        #expect(scenes[0]["prompt"] as? String == #"scene\n"#)
    }

    @Test func responseCountsAcceptExactDecimalAndIntegerButRefuseInvalidValues() throws {
        for count in [#""18446744073709551615""#, "18446744073709551615"] {
            let plan = try MoldJSON.decoder.decode(VideoJobPlan.self, from: Data(Self.plan(clipCount: count).utf8))
            #expect(plan.clipCount == UInt64.max)
            #expect(plan.singleClipMaxFrames == 121)
            #expect(plan.joinMode == .continuous)
        }
        for count in [#""+1""#, #"" 1""#, #""18446744073709551616""#, "-1", "1.5"] {
            #expect(throws: (any Error).self) {
                try MoldJSON.decoder.decode(VideoJobPlan.self, from: Data(Self.plan(clipCount: count).utf8))
            }
        }
    }

    @Test func parentStatusUsesServerLifecycleFlagsAndPrivateMediaNeverRidesStatus() throws {
        let data = Data("""
        {"id":"parent","state":"interrupted","plan":\(Self.plan()),"completed_clips":"2",
         "active_child_id":null,"output_filename":null,"error":"Interrupted","error_detail":null,
         "created_at_ms":1,"updated_at_ms":2,"can_cancel":false,"can_resume":true,"can_delete":true}
        """.utf8)
        let job = try MoldJSON.decoder.decode(VideoJob.self, from: data)
        #expect(job.state == .interrupted)
        #expect(job.completedClips == 2)
        #expect(!job.canCancel && job.canResume && job.canDelete)
        #expect(job.outputFilename == nil)
    }

    @Test func olderServersDoNotAdvertiseTheNewParentOrLibraryExtend() throws {
        for json in [#"{}"#, #"{"video_jobs":{"supported":false,"literal_prompts":true,"library_extend":true}}"#] {
            let caps = try MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
            #expect(!caps.canCreateVideoJob)
            #expect(!caps.canExtendLibraryVideo)
        }
        let caps = try MoldJSON.decoder.decode(Capabilities.self,
            from: Data(#"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true}}"#.utf8))
        #expect(caps.canCreateVideoJob && caps.canExtendLibraryVideo)
    }

    static func plan(clipCount: String = #""4""#) -> String {
        """
        {"validation_revision":"revision","model":"model","width":768,"height":768,"fps":24,
         "single_clip_max_frames":121,"overlap_frames":9,"join_mode":"continuous",
         "clip_count":\(clipCount),"part_count":"1","added_frames":"720","source_frames":"0",
         "total_frames":"720","added_duration_ms":"30000","total_duration_ms":"30000","warnings":[]}
        """
    }
}
