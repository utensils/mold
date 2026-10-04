import Foundation
import Testing
@testable import MoldClient

@Test func staleReferenceImportsNeverMutateAChangedDraftOrDestination() {
    var media = DraftMedia(); media.editImages = ["A", "B"]
    let host = UUID()
    let fence = ReferenceImportFence(model: "m", host: host, recipe: nil, media: media)
    #expect(fence.isCurrent(model: "m", host: host, recipe: nil, media: media))
    media.editImages.swapAt(0, 1)
    #expect(!fence.isCurrent(model: "m", host: host, recipe: nil, media: media))
    media.editImages = ["A", "B"]
    #expect(!fence.isCurrent(model: "other", host: host, recipe: nil, media: media))
    #expect(!fence.isCurrent(model: "m", host: UUID(), recipe: nil, media: media))
}

@Test func placementDropsNewBoundaryNamesAndPixels() {
    var request = GenerateRequest(prompt: "private", model: "m", width: 256, height: 256, steps: 4, guidance: 0, batchSize: 1)
    request.keyframes = [.init(frame: 0, image: "private bytes", name: "private photo.png")]
    let preview = request.redactedForPlacement()
    #expect(preview.keyframes?.first?.name == nil)
    #expect(preview.keyframes?.first?.image == "")
    #expect(preview.keyframes?.first?.frame == 0)
}

@Test func replacementOfDuplicateReferenceKeepsSelectedOccurrence() {
    let duplicate = GenerationReference(kind: "audio", media: .init(authority: "inline", data: "AA=="), mimeType: "audio/wav")
    let replacement = GenerationReference(kind: "audio", media: .init(authority: "inline", data: "AQ=="), mimeType: "audio/wav")
    var media = DraftMedia()
    media.generationReferences = [duplicate, duplicate]
    let fence = ReferenceImportFence(model: "h3", host: nil, recipe: nil, media: media)
    #expect(fence.isCurrent(model: "h3", host: nil, recipe: nil, media: media))
    media.replaceGenerationReference(at: 1, with: replacement)
    #expect(media.generationReferences == [duplicate, replacement])
}
