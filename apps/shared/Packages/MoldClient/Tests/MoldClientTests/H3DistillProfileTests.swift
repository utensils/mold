import Foundation
import Testing
@testable import MoldClient

/// Consume the generated server contract rather than a second native copy of
/// these tiers. Both native apps use this adoption and request-building path.
struct H3DistillProfileTests {
    static let tiers = [
        "minimax-h3-fl2va:comfy-pruned-int8-dmad-4step",
        "minimax-h3-fl2va:comfy-pruned-int8-pdmd-2step",
        "minimax-h3-fl2va:comfy-pruned-int8-pdmd-4step",
        "minimax-h3-fl2va:fasth3-v2-int8",
    ]

    private func profile(_ model: String) throws -> GenerationProfileSet {
        struct Document: Decodable {
            struct Row: Decodable {
                struct Model: Decodable { let model: String }
                let models: [Model]
                let profile: GenerationProfileSet
            }
            let profiles: [Row]
        }
        let root = try #require(RepoFixtures.repoRoot)
        let data = try Data(contentsOf: root.appending(path: "docs/generated/generation-profiles-v1.json"))
        let document = try MoldJSON.decoder.decode(Document.self, from: data)
        return try #require(document.profiles.first { $0.models.contains { $0.model == model } }).profile
    }

    @Test(arguments: tiers)
    func textOnlyUsesFixedTierScheduleAndOmitsConditioning(model: String) throws {
        let profile = try profile(model)
        let recipe = try #require(profile.recipes.first)
        var previous = RenderDraft()
        previous.steps = 50
        previous.guidance = 7
        previous.advanced.scheduler = "euler"
        previous.outputFormat = "webm"
        previous.preferredAudio = false
        let draft = previous.adopting(recipe, isNewModel: true, family: "minimax-h3",
                                      model: model, profile: profile)
        let request = RenderRequest.one(draft, model: model)
        let gridPoints = model.contains("pdmd-2step") ? 3 : model.contains("fasth3") ? 9 : 5
        // Cold manifest profiles omit source_image; live hosts advertise optional.
        // Both contracts allow a source without requiring one.
        #expect(recipe.capabilities.readsSourceImage)
        #expect(!recipe.capabilities.requiresSourceImage)
        #expect(recipe.steps.mode == .fixed)
        #expect(request.steps == gridPoints)
        #expect(request.guidance == 0)
        #expect(request.scheduler == nil)
        #expect(request.sourceImage == nil)
        #expect(request.keyframes == nil || request.keyframes?.isEmpty == true)
        #expect(request.references == nil)
        #expect(request.outputFormat == "mp4")
        #expect(draft.requiresAudio)
        #expect(request.enableAudio == nil)
        #expect(request.videoOnly == nil)
        #expect(BoundaryFramePolicy.refusal(draft: draft, capabilities: recipe.capabilities) == nil)
        #expect(ExpandTask.forRequest(family: "minimax-h3", request: request) == .textToVideo)
        #expect(AdvancedControlsOffered.resolve(recipe: recipe, in: profile, family: "minimax-h3").schedulers.isEmpty)
    }

    @Test(arguments: tiers)
    func endpointsRemainOptionalAndFollowTheRequestedLastFrame(model: String) throws {
        let profile = try profile(model)
        let recipe = try #require(profile.recipes.first)
        var draft = RenderDraft().adopting(recipe, isNewModel: true, family: "minimax-h3", model: model, profile: profile)
        let first = ImportedPicture(encoded: "FIRST", name: "first.png", data: Data())
        let last = ImportedPicture(encoded: "LAST", name: "last.png", data: Data())
        BoundaryFramePolicy.set(first: true, picture: first, draft: &draft, capabilities: recipe.capabilities)
        let firstOnly = RenderRequest.one(draft, model: model)
        #expect(firstOnly.sourceImage == first.encoded)
        #expect(firstOnly.keyframes == nil || firstOnly.keyframes?.isEmpty == true)
        #expect(ExpandTask.forRequest(family: "minimax-h3", request: firstOnly) == .imageToVideo)
        BoundaryFramePolicy.set(first: false, picture: last, draft: &draft, capabilities: recipe.capabilities)
        draft.frames = 148
        let both = RenderRequest.one(draft, model: model)
        #expect(both.sourceImage == first.encoded)
        #expect(both.keyframes?.map(\.frame) == [147])
        #expect(both.keyframes?.map(\.image) == [last.encoded])
        #expect(both.references == nil)
        #expect(ExpandTask.forRequest(family: "minimax-h3", request: both) == .keyframeInterpolation)
        BoundaryFramePolicy.set(first: true, picture: nil, draft: &draft, capabilities: recipe.capabilities)
        #expect(RenderRequest.one(draft, model: model).sourceImage == nil)
        #expect(BoundaryFramePolicy.refusal(draft: draft, capabilities: recipe.capabilities) == nil)
        BoundaryFramePolicy.set(first: false, picture: nil, draft: &draft, capabilities: recipe.capabilities)
        let cleared = RenderRequest.one(draft, model: model)
        #expect(cleared.sourceImage == nil)
        #expect(cleared.keyframes == nil || cleared.keyframes?.isEmpty == true)
        #expect(ExpandTask.forRequest(family: "minimax-h3", request: cleared) == .textToVideo)
    }

    @Test(arguments: tiers)
    func liveAndLegacyCapabilitiesKeepTheHostAuthoritative(model: String) throws {
        let absent = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data("{}".utf8))
        let legacy = absent.resolvingLegacyReferences(model: model)
        #expect(BoundaryFramePolicy.resolve(capabilities: legacy) == "h3-endpoints")
        #expect(legacy.boundaryFrames?.firstRequired == false)
        let live = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data("""
        {"source_image":"optional","boundary_frames":{"mode":"adjustable","wire":"h3-endpoints",
        "min_frames":107,"first_required":false,"last_required":false}}
        """.utf8))
        #expect(live.resolvingLegacyReferences(model: model) == live)
        #expect(live.readsSourceImage && !live.requiresSourceImage)
        // An older/limited host that requires a first frame must not be
        // overruled by a newer client merely recognizing the model name.
        let required = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data("""
        {"source_image":"required","boundary_frames":{"mode":"adjustable","wire":"h3-endpoints",
        "min_frames":107,"first_required":true,"last_required":false}}
        """.utf8))
        let resolved = required.resolvingLegacyReferences(model: model)
        #expect(resolved == required)
        #expect(BoundaryFramePolicy.refusal(draft: RenderDraft(), capabilities: resolved) != nil)
    }

}
