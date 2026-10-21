import Foundation
import Testing
@testable import MoldClient

struct RetainedVideoJobReuseRoleTests {
    @Test func parentLogicalStageInputsKeepTheirOriginalRolesForSessionHydration() throws {
        let member = RetainedSourceMedia.Member(memberId: "reference", role: "stage_reference:0:0", displayName: "original", sizeBytes: 1)
        let reference = GenerationReference(kind: "image", media: .init(authority: "descriptor"), mimeType: "image/png", width: 1, height: 1)
        let request = GenerateRequest(prompt: "literal", model: "video", width: 768, height: 768, steps: 3, guidance: 0)
        var scoped = request; scoped.references = [reference]
        #expect(RetainedSourceMedia.members([member], forHydrating: scoped) == [member])
        #expect(RetainedReferenceGuard.canHydrate(references: [reference], original: [reference], members: [member]))
        #expect(RetainedSourceMedia.field(forRole: "stage_reference:1:0") == nil)
    }
    @Test func archivedParentClosingFrameIsRawImageRatherThanAKeyframeDocument() throws {
        let member = RetainedSourceMedia.Member(memberId: "closing", role: "stage_keyframe:0:0", displayName: "closing.png", sizeBytes: 1, frameIndex: 106)
        var draft = RenderDraft(); draft.frames = 107
        let restored = try RetainedSourceMedia.materializedDraft([(member, Data([1]))], into: draft)
        #expect(restored.media.keyframes.first?.frame == 106)
        #expect(restored.media.keyframes.first?.image == "AQ==")
    }
}
