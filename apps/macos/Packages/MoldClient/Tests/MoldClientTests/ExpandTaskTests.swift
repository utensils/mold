import Foundation
import Testing

@testable import MoldClient

// What kind of render a prompt is being written for (findings 01#13, 02#12).

private func request(_ build: (inout RenderDraft) -> Void = { _ in }) -> GenerateRequest {
    var draft = RenderDraft()
    draft.prompt = "a tin robot"
    build(&draft)
    return draft.request(model: "m")
}

/// **Fails today**: every accepted expand offer hard-coded `.textToImage`,
/// so a clip, an img2img render and a keyframe interpolation all recorded
/// that their prompt was written for a still.
@Test func aStillFamilyIsAlwaysTextToImage() {
    #expect(ExpandTask.forRequest(family: "flux", request: request()) == .textToImage)
    #expect(ExpandTask.forRequest(family: "sdxl", request: request()) == .textToImage)
    #expect(ExpandTask.forRequest(family: nil, request: request()) == .textToImage)
    // Even with a source picture: img2img on a still family is still a still.
    #expect(ExpandTask.forRequest(family: "flux", request: request {
        $0.media.sourceImage = "SRC"
    }) == .textToImage)
}

@Test func aClipFamilyReadsItsOwnConditioning() {
    #expect(ExpandTask.forRequest(family: "ltx2", request: request()) == .textToVideo)
    #expect(ExpandTask.forRequest(family: "ltx-2", request: request()) == .textToVideo)
    #expect(ExpandTask.forRequest(family: "wan", request: request {
        $0.media.sourceImage = "SRC"
    }) == .imageToVideo)
    #expect(ExpandTask.forRequest(family: "ltx2", request: request {
        $0.media.sourceVideo = "VID"
    }) == .videoToVideo)
    #expect(ExpandTask.forRequest(family: "ltx2", request: request {
        $0.media.extendVideo = "VID"
    }) == .videoToVideo)
    #expect(ExpandTask.forRequest(family: "ltx2", request: request {
        $0.media.audioFile = "AUD"
    }) == .audioDrivenVideo)
    #expect(ExpandTask.forRequest(family: "ltx2", request: request {
        $0.media.keyframes = [
            KeyframeCondition(frame: 1, image: "A"), KeyframeCondition(frame: 40, image: "B"),
        ]
    }) == .keyframeInterpolation)
}

@Test func anExplicitPipelineOutranksTheImplicitPriority() {
    #expect(ExpandTask.forRequest(family: "ltx2", request: request {
        $0.pipeline = "t2a"
    }) == .textToAudio)
    #expect(ExpandTask.forRequest(family: "ltx2", request: request {
        $0.pipeline = "lip-dub"
        $0.media.sourceImage = "SRC"
    }) == .audioDrivenVideo)
    #expect(ExpandTask.forRequest(family: "ltx2", request: request {
        $0.pipeline = "keyframe"
    }) == .keyframeInterpolation)
    // An unknown pipeline falls through to the conditioning, rather than
    // being answered from a name this build predates.
    #expect(ExpandTask.forRequest(family: "ltx2", request: request {
        $0.pipeline = "something-new"
        $0.media.sourceImage = "SRC"
    }) == .imageToVideo)
}

/// A single-frame wan render with no conditioning is a still (#798) -- and
/// deliberately only when nothing conditions it.
@Test func aSingleFrameWanRenderIsAStillUnlessSomethingConditionsIt() {
    #expect(ExpandTask.forRequest(family: "wan", request: request {
        $0.frames = 1
    }) == .textToImage)
    #expect(ExpandTask.forRequest(family: "wan", request: request {
        $0.frames = 1
        $0.media.sourceImage = "SRC"
    }) == .imageToVideo)
    #expect(ExpandTask.forRequest(family: "ltx2", request: request {
        $0.frames = 1
    }) == .textToVideo)
}

@Test func h3ReadsItsTwoBoundaryFrames() {
    #expect(ExpandTask.forRequest(family: "minimax-h3", request: request()) == .textToVideo)
    #expect(ExpandTask.forRequest(family: "minimax_h3", request: request {
        $0.media.sourceImage = "FIRST"
    }) == .imageToVideo)
    #expect(ExpandTask.forRequest(family: "minimaxh3", request: request {
        $0.media.sourceImage = "FIRST"
        $0.media.keyframes = [KeyframeCondition(frame: 40, image: "LAST")]
    }) == .keyframeInterpolation)
}

/// The EXCLUSIVE relation parks a well, and the task follows the request that
/// ships -- not whatever the draft happens to be holding.
@Test func theTaskFollowsWhatTheRequestReallyCarries() {
    var draft = RenderDraft()
    draft.media.sourceMode = .singleOrReferences
    draft.media.sourceImage = "SRC"
    draft.media.editImages = ["REF"]
    draft.media.lastExclusiveWrite = .references
    #expect(ExpandTask.forRequest(
        family: "ltx2", request: draft.request(model: "m")) == .textToVideo)

    draft.media.lastExclusiveWrite = .source
    #expect(ExpandTask.forRequest(
        family: "ltx2", request: draft.request(model: "m")) == .imageToVideo)
}
