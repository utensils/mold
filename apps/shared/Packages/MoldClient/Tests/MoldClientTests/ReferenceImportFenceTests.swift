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
    media.generationReferences = [.canonical(duplicate), .canonical(duplicate)]
    let fence = ReferenceImportFence(model: "h3", host: nil, recipe: nil, media: media)
    #expect(fence.isCurrent(model: "h3", host: nil, recipe: nil, media: media))
    media.replaceGenerationReference(at: 1, with: replacement)
    #expect(media.generationReferences == [.canonical(duplicate), .canonical(replacement)])
}

@Test func normalizedReferenceImportRejectsCredentialAndInstanceChangesWithSameHostID() {
    let host = MoldHost(name: "machine", baseURL: URL(string: "http://host")!, apiKey: "old")
    let fence = ReferenceImportFence(model: "h3", host: host.id, recipe: nil, media: DraftMedia(),
        authority: host, instance: "one")
    #expect(fence.isCurrent(model: "h3", host: host.id, recipe: nil, media: DraftMedia(), authority: host, instance: "one"))
    var changed = host; changed.apiKey = "new"
    #expect(!fence.isCurrent(model: "h3", host: host.id, recipe: nil, media: DraftMedia(), authority: changed, instance: "one"))
    #expect(!fence.isCurrent(model: "h3", host: host.id, recipe: nil, media: DraftMedia(), authority: host, instance: "two"))
}

@Test func choosingAnotherLibraryPrefixInvalidatesAnUnfinishedInputImport() {
    let host = UUID(), media = DraftMedia()
    let first = VideoJobSource(filename: "first.mp4", expectedArchiveIdentitySha256: "first")
    let second = VideoJobSource(filename: "second.mp4", expectedArchiveIdentitySha256: "second")
    let captured = ReferenceImportFence(model: "video", host: host, recipe: nil, media: media, videoSource: first)
    #expect(captured.isCurrent(model: "video", host: host, recipe: nil, media: media, videoSource: first))
    #expect(!captured.isCurrent(model: "video", host: host, recipe: nil, media: media, videoSource: second))
    #expect(!captured.isCurrent(model: "video", host: host, recipe: nil, media: media, videoSource: nil))
}
