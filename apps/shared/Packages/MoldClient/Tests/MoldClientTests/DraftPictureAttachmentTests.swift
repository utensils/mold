import Foundation
import Testing
@testable import MoldClient

struct DraftPictureAttachmentTests {
    @Test func typedReferenceAppendIsAtomicAndKeepsScalarEdits() throws {
        var json = try #require(JSONSerialization.jsonObject(with: RepoFixtures.fixture("recipe-ltx2.json")) as? [String: Any])
        var recipes = try #require(json["recipes"] as? [[String: Any]])
        recipes[0]["capabilities"] = ["generation_references": ["mode": "adjustable", "required": true,
            "kinds": ["image"], "max_count": 1, "max_images": 1, "max_videos": 0, "max_audios": 0,
            "min_duration_ms": 2000, "max_duration_ms": 15000, "max_video_duration_ms": 15000,
            "max_audio_duration_ms": 15000, "max_inline_bytes": 33554432, "requires_visual": true]]
        json["recipes"] = recipes
        let recipe = try MoldJSON.decoder.decode(GenerationProfileSet.self,
            from: JSONSerialization.data(withJSONObject: json)).recipes[0]
        let reference = GenerationReference(kind: "image", media: .init(authority: "inline", data: "YQ=="),
                                             mimeType: "image/png", width: 64, height: 64)
        var draft = RenderDraft()
        draft.prompt = "preserve me"; draft.steps = 17
        try DraftPictureAttachment.addTypedReference(reference, to: &draft, recipe: recipe)
        #expect(draft.prompt == "preserve me" && draft.steps == 17)
        #expect(draft.media.generationReferences == [.canonical(reference)])
        let before = draft
        #expect(throws: (any Error).self) {
            try DraftPictureAttachment.addTypedReference(reference, to: &draft, recipe: recipe)
        }
        #expect(draft == before)
    }

    @Test func replacingSourceInvalidatesOnlyItsOldMask() {
        var draft = RenderDraft()
        draft.media.sourceImageOriginal = "old"
        draft.media.maskImage = "old-source-coordinates"
        let picked = ImportedPicture(encoded: "new", name: "new.png", data: Data())
        DraftPictureAttachment.useAsSource(picked, in: &draft, recipe: nil)
        #expect(draft.media.maskImage == nil)
        draft.media.maskImage = "new-source-coordinates"
        DraftPictureAttachment.useAsSource(picked, in: &draft, recipe: nil)
        #expect(draft.media.maskImage == "new-source-coordinates")
    }
}
