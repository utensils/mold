import Testing
@testable import MoldClient

struct SequenceStageBoundaryTests {
    @Test func editingAReusedClipLengthKeepsItsClosingFrameAtTheEnd() {
        var stage = SequenceDraft.Stage(prompt: "same", frames: 107)
        stage.keyframes = [.init(frame: 0, image: "opening"), .init(frame: 50, image: "interior"), .init(frame: 106, image: "closing")]
        stage.frames = 131
        #expect(stage.keyframes?.map(\.frame) == [0, 50, 130])
        #expect(stage.keyframes?.map(\.image) == ["opening", "interior", "closing"])
    }
    @Test func aSingleFrameOpeningDoesNotBecomeAClosingFrame() {
        var stage = SequenceDraft.Stage(prompt: "same", frames: 1)
        stage.keyframes = [.init(frame: 0, image: "opening")]
        stage.frames = 107
        #expect(stage.keyframes?.first?.frame == 0)
    }
}
