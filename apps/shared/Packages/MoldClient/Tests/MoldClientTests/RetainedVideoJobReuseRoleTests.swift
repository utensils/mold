import CryptoKit
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
    @Test func crossHostParentReferencesFollowCanonicalClipPositionsRatherThanDownloadOrder() throws {
        func ref(_ bytes: Data) -> GenerationReference {
            let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            return GenerationReference(kind: "image", media: .init(authority: "descriptor"), mimeType: "image/png",
                provenance: .init(sha256: digest), width: 1, height: 1)
        }
        let first = RetainedSourceMedia.Member(memberId: "a", role: "stage_reference:0:0", displayName: "a", sizeBytes: 1)
        let second = RetainedSourceMedia.Member(memberId: "b", role: "stage_reference:0:1", displayName: "b", sizeBytes: 1)
        var request = GenerateRequest(prompt: "new", model: "video", width: 768, height: 768, steps: 3, guidance: 0)
        request.references = [ref(Data([1])), ref(Data([2]))]
        let restored = try RetainedSourceMedia.relayed([(second, Data([2])), (first, Data([1]))], into: request)
        #expect(restored.references?.map(\.media.data) == ["AQ==", "Ag=="])
        let duplicate = RetainedSourceMedia.Member(memberId: "c", role: "stage_reference:0:0", displayName: "duplicate", sizeBytes: 1)
        #expect(throws: (any Error).self) { try RetainedSourceMedia.relayed([(first, Data([1])), (duplicate, Data([2]))], into: request) }
        let ordinary = RetainedSourceMedia.Member(memberId: "ordinary", role: "references", displayName: "ordinary", sizeBytes: 1)
        #expect(throws: (any Error).self) { try RetainedSourceMedia.relayed([(ordinary, Data([1])), (second, Data([2]))], into: request) }
        request.references = [ref(Data([2]))]
        #expect(throws: (any Error).self) { try RetainedSourceMedia.relayed([(second, Data([2]))], into: request) }
    }

}
