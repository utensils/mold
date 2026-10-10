import CoreGraphics
import Foundation
import Testing
@testable import MoldClient

struct BoundarySourceFitTests {
    private func recipe(_ wire: String, domain: String = "buckets") throws -> GenerationRecipe {
        struct Document: Decodable {
            struct Row: Decodable { let profile: GenerationProfileSet }
            let profiles: [Row]
        }
        let root = try #require(RepoFixtures.repoRoot)
        let data = try Data(contentsOf: root.appending(path: "docs/generated/generation-profiles-v1.json"))
        let document = try MoldJSON.decoder.decode(Document.self, from: data)
        let recipe = try #require(document.profiles.flatMap(\.profile.recipes).first {
            BoundaryFramePolicy.resolve(capabilities: $0.capabilities) == wire
                && $0.resolution.hasCanvas
        })
        if domain == "buckets" { return recipe }
        var json = try #require(JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(recipe)) as? [String: Any])
        var resolution = try #require(json["resolution"] as? [String: Any])
        resolution["domain"] = domain; json["resolution"] = resolution
        return try MoldJSON.decoder.decode(GenerationRecipe.self, from: JSONSerialization.data(withJSONObject: json))
    }

    private func picture(_ width: Int, _ height: Int) -> ImportedPicture {
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        let data = SourceFitRender.encodePNG(context.makeImage()!)!
        return ImportedPicture(encoded: data.base64EncodedString(), name: "frame.png", data: data)
    }

    @Test(arguments: ["h3-endpoints", "wan-pair"])
    func lastOnlyHasDefaultFitControlsAndFitsSubmissionBytes(wire: String) async throws {
        let recipe = try recipe(wire)
        var draft = RenderDraft().adopting(recipe, isNewModel: true)
        let last = picture(90, 160)
        BoundaryFramePolicy.set(first: false, picture: last, draft: &draft,
                                capabilities: recipe.capabilities, recipe: recipe)
        let modes = SourceFitOptions.resolve(recipe: recipe, media: draft.media)
        #expect(modes == [.cropFill, .padFit, .lanczosResize])
        #expect(draft.media.sourceFit == .default)
        draft.width = 128; draft.height = 64
        let index = draft.media.keyframes.first?.frame
        let fitted = try await draft.fittingSource(recipe: recipe)
        let pixels = try #require(fitted.media.keyframes.first.flatMap { ReferenceCanvas.uprightPixels(ofBase64: $0.image) })
        #expect(pixels.width == 128 && pixels.height == 64)
        #expect(fitted.media.keyframes.first?.frame == index)
        #expect(fitted.media.sourceImage == nil)
        #expect(fitted.media.maskImage == nil)
        #expect(draft.media.keyframes.first?.image == last.encoded)
        let request = RenderRequest.one(fitted, model: "boundary")
        #expect(request.keyframes?.first?.frame == index)
        #expect(request.sourceImage == nil)
        #expect(request.sourceFit == .default)
    }

    @Test(arguments: ["h3-endpoints", "wan-pair"])
    func bothEndpointsFitWithoutChangingAuthoringBytesOrMakingAMask(wire: String) async throws {
        let recipe = try recipe(wire)
        var draft = RenderDraft().adopting(recipe, isNewModel: true)
        BoundaryFramePolicy.set(first: true, picture: picture(160, 90), draft: &draft,
                                capabilities: recipe.capabilities, recipe: recipe)
        BoundaryFramePolicy.set(first: false, picture: picture(90, 160), draft: &draft,
                                capabilities: recipe.capabilities, recipe: recipe)
        draft.width = 128; draft.height = 64
        draft.media.sourceFit = .padRepaint
        let originals = draft.media.keyframes
        let originalSource = draft.media.sourceImageOriginal
        let fitted = try await draft.fittingSource(recipe: recipe)
        for frame in fitted.media.keyframes {
            let pixels = try #require(ReferenceCanvas.uprightPixels(ofBase64: frame.image))
            #expect(pixels.width == 128 && pixels.height == 64)
        }
        if wire == "h3-endpoints" {
            let pixels = try #require(fitted.media.sourceImage.flatMap { ReferenceCanvas.uprightPixels(ofBase64: $0) })
            #expect(pixels.width == 128 && pixels.height == 64)
        }
        #expect(fitted.media.keyframes.map(\.frame) == originals.map(\.frame))
        #expect(fitted.media.sourceImageOriginal == originalSource)
        #expect(draft.media.keyframes == originals)
        #expect(fitted.media.maskImage == nil)
        #expect(fitted.media.sourceFit == .default)
    }

    @Test func cancelledEndpointPreparationCannotProduceASubmission() async throws {
        let recipe = try recipe("h3-endpoints")
        var draft = RenderDraft().adopting(recipe, isNewModel: true)
        BoundaryFramePolicy.set(first: false, picture: picture(90, 160), draft: &draft,
                                capabilities: recipe.capabilities, recipe: recipe)
        let snapshot = draft
        let work = Task {
            // Arrange a known cancelled task independently of scheduler timing.
            while !Task.isCancelled { await Task.yield() }
            return try await snapshot.fittingBoundaryFrames(recipe: recipe)
        }
        work.cancel()
        do {
            _ = try await work.value
            Issue.record("Cancelled endpoint preparation must not return submission bytes")
        } catch is CancellationError {
            // The admission caller receives cancellation, never a partial fit.
        }
    }

    @Test func sourceDrivenAndParkedEndpointsOfferNoFit() async throws {
        let recipe = try recipe("h3-endpoints", domain: "source-driven")
        var draft = RenderDraft()
        draft.media.keyframes = [KeyframeCondition(frame: 8, image: picture(90, 160).encoded, name: "last.png")]
        #expect(SourceFitOptions.resolve(recipe: recipe, media: draft.media).isEmpty)
        let unchanged = try await draft.fittingSource(recipe: recipe)
        #expect(unchanged.media.keyframes == draft.media.keyframes)
        draft.media.boundaryKeyframes["h3-endpoints"] = draft.media.keyframes
        draft.media.keyframes = []
        #expect(SourceFitOptions.resolve(recipe: try self.recipe("h3-endpoints"), media: draft.media).isEmpty)
    }
}
