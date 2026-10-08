import Foundation
import Testing
@testable import MoldClient

@Suite struct RetainedBoundaryDurationTests {
    @Test(arguments: ["h3-endpoints", "wan-pair"])
    func shortenedClipRetargetsVisibleRetainedClosingFrame(wire: String) throws {
        let capabilities = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data("""
        {"boundary_frames":{"mode":"adjustable","wire":"\(wire)","min_frames":9,"first_required":false,"last_required":false},"accepts_keyframes":true}
        """.utf8))
        var draft = RenderDraft()
        draft.frames = 209
        draft.media.adoptedReferenceCapabilities = capabilities
        let frames = wire == "wan-pair" ? [KeyframeCondition(frame: 0, image: "first"), .init(frame: 208, image: "last")] : [.init(frame: 208, image: "last")]
        let downloaded = try frames.enumerated().map { index, frame in
            (member: RetainedSourceMedia.Member(memberId: "\(index)", role: "keyframes", displayName: "frame", sizeBytes: 1), bytes: try MoldJSON.encoder.encode(frame))
        }
        draft = try RetainedSourceMedia.materializedDraft(downloaded, into: draft)
        draft.frames = 107
        let request = RenderRequest.one(draft, model: "fixture")
        #expect(request.keyframes?.last?.frame == 106)
        #expect(request.keyframes?.last?.image == "last")
        draft.media.keyframes = []
        #expect(RenderRequest.one(draft, model: "fixture").keyframes == nil)
    }
}
