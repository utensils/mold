import Foundation
import Testing

@testable import MoldClient

/// A placement preview prices a render; it does not make one (finding 02#5).

private func fullDraft() -> RenderDraft {
    var draft = RenderDraft()
    draft.prompt = "a tin robot in a field"
    draft.negativePrompt = "blurry"
    draft.originalPrompt = "a robot"
    draft.title = "Client X, unannounced"
    draft.tags = ["blue"]
    draft.collectionName = "Client X"
    draft.media.sourceMode = .singleAndReferences
    draft.media.sourceImage = "SOURCEBYTES"
    draft.media.sourceImageName = "secret-brief.png"
    draft.media.maskImage = "MASKBYTES"
    draft.media.editImages = ["REF1", "REF2"]
    draft.media.keyframes = [KeyframeCondition(frame: 1, image: "KEYBYTES", name: "k.png")]
    draft.media.audioFile = "AUDIOBYTES"
    draft.media.sourceVideo = "VIDEOBYTES"
    draft.media.control = ControlConditioning(image: "CTRLBYTES", model: "controlnet-canny-sd15:fp16")
    draft.media.identity = IdentityConditioning(
        photos: [IdentityPhoto(encoded: "FACEBYTES", name: "me.heic")])
    return draft
}

/// **Fails today**: `placementRequest` is literally `request(...)`, so every
/// keystroke re-uploaded the whole conditioning set -- and the filing text --
/// to price a render nobody had asked for yet.
@Test func aPlacementPreviewCarriesNoBytesAndNoUserText() {
    let request = RenderRequest.placement(fullDraft(), model: "sd15:fp16", maxIdentityPhotos: 4)

    #expect(request.prompt.isEmpty)
    #expect(request.negativePrompt == "")
    #expect(request.originalPrompt == "")
    #expect(request.sourceImage == "")
    #expect(request.sourceImageName == "")
    #expect(request.maskImage == "")
    #expect(request.controlImage == "")
    #expect(request.audioFile == "")
    #expect(request.sourceVideo == "")
    #expect(request.idImage == "")
    #expect(request.idImageName == "")
    #expect(request.editImages == ["", ""])
    #expect(request.keyframes?.map(\.image) == [""])

    // Filing is DELETED, not blanked: all three are additive, absent is their
    // normal shape, and a preview files nothing. The TITLE is filing too --
    // it is exactly the "Client X, unannounced" the rule is written about.
    #expect(request.title == nil)
    #expect(request.tags == nil)
    #expect(request.collection == nil)
}

@Test func aPlacementPreviewKeepsEverythingThePlannerReads() {
    var draft = fullDraft()
    draft.width = 1344
    draft.height = 768
    draft.steps = 28
    draft.guidance = 4.5
    draft.frames = 97
    draft.fps = 24
    let request = RenderRequest.placement(draft, model: "sd15:fp16", maxIdentityPhotos: 4)

    #expect(request.model == "sd15:fp16")
    #expect(request.width == 1344)
    #expect(request.height == 768)
    #expect(request.steps == 28)
    #expect(request.guidance == 4.5)
    #expect(request.frames == 97)
    #expect(request.fps == 24)
    // Presence is structural -- it decides the conditioning path and the
    // activation budget -- so an attached picture stays PRESENT and empty
    // rather than disappearing.
    #expect(request.sourceImage != nil)
    #expect(request.strength != nil)
    #expect(request.keyframes?.first?.frame == 1)
    #expect(request.controlModel == "controlnet-canny-sd15:fp16")
}

@Test func aRequestWithNoMediaGrowsNoEmptyFields() {
    var draft = RenderDraft()
    draft.prompt = "a cat"
    let request = RenderRequest.placement(draft, model: "m")
    #expect(request.sourceImage == nil)
    #expect(request.editImages == nil)
    #expect(request.idImage == nil)
    #expect(request.keyframes == nil)
}

/// The submitted request is untouched -- redaction belongs to the preview.
@Test func theSubmittedRequestStillCarriesEverything() {
    let request = RenderRequest.one(fullDraft(), model: "sd15:fp16", maxIdentityPhotos: 4)
    #expect(request.prompt == "a tin robot in a field")
    #expect(request.sourceImage == "SOURCEBYTES")
    #expect(request.editImages == ["REF1", "REF2"])
    #expect(request.idImage == "FACEBYTES")
    #expect(request.title == "Client X, unannounced")
    #expect(request.tags?.isEmpty == false)
    #expect(request.collection != nil)
}
