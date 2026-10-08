import Foundation
import Testing
@testable import MoldClient

struct RetainedDraftRestorationTests {
    @Test func restoresBothEndpointsAndKeepsAuthoredCanvas() throws {
        var draft = RenderDraft()
        draft.frames = 141
        draft.width = 768
        draft.height = 512
        draft.canvasIntent = .manual
        let keyframes = [KeyframeCondition(frame: 140, image: "closing", name: "last.png")]
        let restored = try RetainedSourceMedia.materializedDraft([
            (member("source_image", name: "first.png"), Data([1, 2, 3])),
            (member("keyframes"), try MoldJSON.encoder.encode(keyframes[0]))
        ], into: draft)
        #expect(restored.media.sourceImage == "AQID")
        #expect(restored.media.sourceImageOriginal == "AQID")
        #expect(restored.media.sourceImageName == "first.png")
        #expect(restored.media.keyframes == keyframes)
        #expect(restored.width == 768 && restored.height == 512 && restored.canvasIntent == .manual)
    }

    @Test func restoresOrderedKeyframesAndReferencesWithoutPromotingAMissingTarget() throws {
        let frames = [KeyframeCondition(frame: 0, image: "first"), .init(frame: 48, image: "middle"), .init(frame: 96, image: "last")]
        let downloaded = try frames.map { (member("keyframes"), try MoldJSON.encoder.encode($0)) }
        let restored = try RetainedSourceMedia.materializedDraft(downloaded + [
            (member("edit_images", name: "target.png"), Data([1])),
            (member("edit_images", name: "reference.png"), Data([2]))
        ], into: RenderDraft())
        #expect(restored.media.keyframes == frames)
        #expect(restored.media.editImages == ["AQ==", "Ag=="])
    }

    @Test func allLegacyMediaRolesBecomeVisibleAndPreserveSettings() throws {
        var draft = RenderDraft()
        draft.media.identity = .init(photos: [], weight: 0.7, startStep: 2)
        draft.media.control = .init(model: "canny", scale: 0.6)
        draft.media.extendOverlapFrames = 17
        let restored = try RetainedSourceMedia.materializedDraft([
            (member("identity_images", name: "face-a.png"), Data([1])),
            (member("identity_images", name: "face-b.png"), Data([2])),
            (member("control_image"), Data([3])),
            (member("audio_file_path", name: "audio.wav"), Data([4])),
            (member("source_video", name: "source.mp4"), Data([5])),
            (member("extend_video_path", name: "extend.mp4"), Data([6]))
        ], into: draft)
        #expect(restored.media.identity?.photos.map(\.encoded) == ["AQ==", "Ag=="])
        #expect(restored.media.identity?.weight == 0.7 && restored.media.identity?.startStep == 2)
        #expect(restored.media.control?.image == "Aw==" && restored.media.control?.scale == 0.6)
        #expect(restored.media.audioFile == "BA==" && restored.media.audioFileName == "audio.wav")
        #expect(restored.media.sourceVideo == "BQ==" && restored.media.sourceVideoName == "source.mp4")
        #expect(restored.media.extendVideo == "Bg==" && restored.media.extendOverlapFrames == 17)
    }

    @Test func manualAttachmentsWinAndSourceMaskPairCannotBeSplit() throws {
        var draft = RenderDraft()
        draft.media.sourceImage = "new-source"
        draft.media.editImages = ["new-target"]
        draft.media.keyframes = [.init(frame: 0, image: "new-frame")]
        let restored = try RetainedSourceMedia.materializedDraft([
            (member("source_image"), Data([1])), (member("mask_image"), Data([2])),
            (member("edit_images"), Data([3])),
            (member("keyframes"), try MoldJSON.encoder.encode(KeyframeCondition(frame: 140, image: "old")))
        ], into: draft)
        #expect(restored == draft)
    }

    @Test func malformedFrameDocumentFailsAtomically() {
        #expect(throws: (any Error).self) {
            try RetainedSourceMedia.materializedDraft([
                (member("source_image"), Data([1])), (member("keyframes"), Data("not a frame".utf8))
            ], into: RenderDraft())
        }
    }

    @Test func extensionOverlapRestoresFromMetadata() throws {
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self,
            from: Data(#"{"extend_video_path":"clip.mp4","extend_overlap_frames":17}"#.utf8))
        #expect(RenderDraft(reusing: metadata).media.extendOverlapFrames == 17)
    }

    @Test func restoresExplicitZeroReferenceWeight() throws {
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self,
            from: Data(#"{"reference_weight":0,"edit_image_sha256s":["target"]}"#.utf8))
        #expect(RenderDraft(reusing: metadata).media.referenceWeight == 0)
    }

    private func member(_ role: String, name: String = "input") -> RetainedSourceMedia.Member {
        .init(memberId: role + name, role: role, displayName: name, sizeBytes: 3)
    }
    @Test func mixedIdentityRolesFailBeforeAnyAttachmentIsPublished() throws {
        let members = ["source_image", "identity_image", "identity_images"].map {
            (member: RetainedSourceMedia.Member(memberId: $0, role: $0, displayName: $0, sizeBytes: 1), bytes: Data([1]))
        }
        #expect(throws: RetainedSourceMedia.RelayFailure.self) {
            try RetainedSourceMedia.materializedDraft(members, into: RenderDraft())
        }
    }

}
