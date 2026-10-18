import Foundation
import Testing
@testable import MoldClient

struct SourceMediaAttachmentTests {
    private func recipe(capabilities: [String: Any]) throws -> GenerationRecipe {
        var profile = try #require(JSONSerialization.jsonObject(with: RepoFixtures.fixture("recipe-ltx2.json")) as? [String: Any])
        var recipes = try #require(profile["recipes"] as? [[String: Any]])
        recipes[0]["capabilities"] = capabilities
        profile["recipes"] = recipes
        return try MoldJSON.decoder.decode(GenerationProfileSet.self, from: JSONSerialization.data(withJSONObject: profile)).recipes[0]
    }
    @Test func typedAudioVideoOfferReferenceReuseWithoutLegacySourceSlots() throws {
        let recipe = try recipe(capabilities: ["generation_references": ["mode": "adjustable", "required": false, "kinds": ["image", "audio", "video"], "max_count": 12, "max_images": 9, "max_videos": 3, "max_audios": 3, "min_duration_ms": 2000, "max_duration_ms": 15000, "max_video_duration_ms": 15000, "max_audio_duration_ms": 15000, "max_inline_bytes": 33554432, "requires_visual": true]])
        for kind in ["audio", "video"] {
            let member = RetainedSourceMedia.Member(memberId: kind, role: "reference", displayName: kind, sizeBytes: 1, mediaKind: kind)
            #expect(SourceMediaAttachment.choices(member: member, recipe: recipe, media: .init()) == [.reference])
        }
    }
    @Test func unknownMembersNeverOfferAnAttachment() throws {
        let recipe = try recipe(capabilities: ["reads_source_image": true])
        let member = RetainedSourceMedia.Member(memberId: "one", role: "future", displayName: "image.png", sizeBytes: 1)
        #expect(SourceMediaAttachment.choices(member: member, recipe: recipe, media: .init()).isEmpty)
    }
    @Test func inspectedUnknownOnlyChangesKindAndOffersCompatibleActions() throws {
        let member = RetainedSourceMedia.Member(memberId: "opaque-member", role: "references",
            displayName: "references-1", sizeBytes: 123, label: "Original reference", mediaKind: "unknown",
            position: 2, frameIndex: 19, imageRole: "front", contentSha256: String(repeating: "a", count: 64))
        let inspected = member.inspected(as: .image)
        var originalJSON = try #require(JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(member)) as? [String: Any])
        let inspectedJSON = try #require(JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(inspected)) as? [String: Any])
        originalJSON["media_kind"] = "image"
        #expect(NSDictionary(dictionary: originalJSON).isEqual(to: inspectedJSON))
        #expect(member.contentKind == "unknown")
        let recipe = try recipe(capabilities: ["reads_source_image": true])
        #expect(SourceMediaAttachment.choices(member: member, recipe: recipe, media: .init()).isEmpty)
        #expect(SourceMediaAttachment.choices(member: inspected, recipe: recipe, media: .init()) == [.source])
        #expect(SourceMediaAttachment.choices(member: member.inspected(as: .audio), recipe: recipe, media: .init()).isEmpty)
        #expect(member.inspected(as: .video).contentKind == "video")
    }
    @Test func boundaryInputsOfferBothExplicitEndpoints() throws {
        let recipe = try recipe(capabilities: ["boundary_frames": ["mode": "adjustable", "wire": "wan-pair", "min_frames": 9, "first_required": false, "last_required": false]])
        let member = RetainedSourceMedia.Member(memberId: "one", role: "source_image", displayName: "input.png", sizeBytes: 1)
        #expect(SourceMediaAttachment.choices(member: member, recipe: recipe, media: .init()) == [.firstFrame, .lastFrame])
    }
    @Test func keyframeRecipesOfferBothEndpoints() throws {
        let recipe = try recipe(capabilities: ["keyframes": ["mode": "adjustable", "required": false]])
        let member = RetainedSourceMedia.Member(memberId: "one", role: "source_image", displayName: "input.png", sizeBytes: 1)
        #expect(SourceMediaAttachment.choices(member: member, recipe: recipe, media: .init()) == [.firstFrame, .lastFrame])
    }
    @Test func explicitBoundaryRefusalDoesNotFallBackToLegacyWan() throws {
        let recipe = try recipe(capabilities: ["boundary_frames": ["mode": "unsupported", "wire": "wan-pair", "min_frames": 9, "first_required": false, "last_required": false], "wan_recipe": ["mode": "adjustable", "supports_distill_strength": false, "supports_first_last_frame": true]])
        #expect(BoundaryFramePolicy.resolve(capabilities: recipe.capabilities) == nil)
    }
    @Test func interpolationEndpointReplacementPreservesInteriorFramesAndCanvas() throws {
        let recipe = try recipe(capabilities: ["keyframes": ["mode": "adjustable", "required": false]])
        var draft = RenderDraft(); draft.frames = 49; draft.width = 768; draft.height = 512
        draft.media.keyframes = [.init(frame: 0, image: "first"), .init(frame: 16, image: "middle"), .init(frame: 48, image: "last")]
        let picture = ImportedPicture(encoded: "replacement", name: "replacement.png", data: Data([1]))
        SourceMediaAttachment.setEndpoint(first: false, picture: picture, draft: &draft, recipe: recipe)
        #expect(draft.media.keyframes.map(\.frame) == [0,16,48])
        #expect(draft.media.keyframes[1].image == "middle")
        #expect(draft.media.keyframes[2].name == "replacement.png")
        #expect(draft.width == 768 && draft.height == 512)
    }
    @Test func continuationPreservesPromptAndParksConflictingImage() async throws {
        let recipe = try recipe(capabilities: ["supports_extend": true])
        let member = RetainedSourceMedia.Member(memberId: "one", role: "source_video", displayName: "input.mp4", sizeBytes: 3)
        var draft = RenderDraft(); draft.prompt = "Keep this prompt"; draft.media.sourceImage = "old-image"
        let next = try await SourceMediaAttachment.continuation.applying(data: Data([1,2,3]), member: member, to: draft, recipe: recipe)
        #expect(next.prompt == draft.prompt)
        #expect(next.media.extendVideo == "AQID")
        #expect(next.media.sourceImage == nil && next.media.parked.sourceImage == "old-image")
    }
}
