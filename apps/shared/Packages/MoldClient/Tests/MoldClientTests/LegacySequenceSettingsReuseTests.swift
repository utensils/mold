import Foundation
import Testing
@testable import MoldClient

struct LegacySequenceSettingsReuseTests {
    func metadata(extra: String = "") throws -> OutputMetadata {
        try MoldJSON.decoder.decode(OutputMetadata.self, from: Data((#"{"prompt":"joined should not become one prompt","negative_prompt":"avoid flicker","model":"video","seed":18446744073709551615,"width":768,"height":768,"steps":3,"guidance":0,"fps":24,"frames":197,"chain":{"stage_count":2,"motion_tail_frames":17,"stages":[{"prompt":"first\\nreal\r\nline","frames":107,"transition":"smooth","seed":"18446744073709551615","loras":[{"path":"local-adapter","scale":0.75}]},{"prompt":"second","frames":107,"transition":"fade","fade_frames":9,"seed":"0"}]"# + extra + "}}").utf8))
    }
    @Test func restoresEveryAuthoredStageAndEffectiveSeedWithoutPromptFlattening() throws {
        let restored = try LegacySequenceSettingsReuse(metadata: metadata())
        #expect(restored.sequence.stages.map(\.prompt) == ["first\\nreal\r\nline", "second"])
        #expect(restored.sequence.stages.map(\.frames) == [107,107])
        #expect(restored.sequence.stages.map(\.transition) == ["smooth","fade"])
        #expect(restored.sequence.stages[1].input.fadeFrames == 9)
        #expect(restored.sequence.stages[0].input.seedOffset == 0)
        #expect(restored.sequence.stages[1].input.seedOffset == UInt64.max)
        #expect(restored.sequence.stages[0].input.loras?.first?.path == "local-adapter")
        #expect(restored.motionTailFrames == 17)
        #expect(restored.draft.frames == 107)
        #expect(restored.draft.seed == UInt64.max)
    }
    @Test func linkedIncomingCarryCannotBeSilentlyReusedAsAnIndependentClip() throws {
        #expect(throws: (any Error).self) {
            try LegacySequenceSettingsReuse(metadata: metadata(extra: #", "continue_from":{"job_id":"original","finalize_take":1}"#))
        }
    }
    @Test func historicalUnrecordedPerClipControlsAreDisclosed() throws {
        let restored = try LegacySequenceSettingsReuse(metadata: metadata())
        #expect(restored.notice.contains("negative prompts"))
    }
    @Test func perClipSourcesRestoreInExactRolesAndRejectUnexpectedOrDuplicateInputs() throws {
        let original = try LegacySequenceSettingsReuse(metadata: metadata())
        let first = RetainedSourceMedia.Member(memberId: "first", role: "stage_source:0", displayName: "first.png", sizeBytes: 1)
        let second = RetainedSourceMedia.Member(memberId: "second", role: "stage_source:1", displayName: "second.png", sizeBytes: 1)
        let restored = try original.materialized([(second, Data([2])), (first, Data([1]))])
        #expect(restored.sequence.stages[0].input.sourceImage == "AQ==")
        #expect(restored.sequence.stages[1].input.sourceImage == "Ag==")
        #expect(throws: (any Error).self) { try original.materialized([(first, Data([1])), (first, Data([1]))]) }
        let other = RetainedSourceMedia.Member(memberId: "bad", role: "control_image", displayName: "unsupported", sizeBytes: 1)
        #expect(throws: (any Error).self) { try original.materialized([(other, Data([1]))]) }
    }

    @Test func missingTopSeedLocksTheRecordedFirstClipSeedBeforeComputingOffsets() throws {
        let full = try metadata()
        let encoded = try MoldJSON.encoder.encode(full)
        var object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "seed")
        let sparse = try MoldJSON.decoder.decode(OutputMetadata.self, from: JSONSerialization.data(withJSONObject: object))
        let restored = try LegacySequenceSettingsReuse(metadata: sparse)
        #expect(restored.draft.seed == UInt64.max)
        #expect(restored.draft.locksSeed)
        #expect(restored.sequence.stages[0].input.seedOffset == 0)
        #expect(restored.sequence.stages[1].input.seedOffset == UInt64.max)
    }

    @Test func ambiguousNumericRoleAliasesCannotOverwriteAnotherClipInput() throws {
        let original = try LegacySequenceSettingsReuse(metadata: metadata())
        for role in ["stage_source:00", "stage_source:+0", "stage_source:-0"] {
            let member = RetainedSourceMedia.Member(memberId: role, role: role, displayName: "alias", sizeBytes: 1)
            #expect(throws: (any Error).self) { try original.materialized([(member, Data([1]))]) }
        }
    }

}
