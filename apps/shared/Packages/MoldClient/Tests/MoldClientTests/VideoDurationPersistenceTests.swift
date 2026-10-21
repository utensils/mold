import Foundation
import Testing
@testable import MoldClient

struct VideoDurationPersistenceTests {
    @Test func ordinaryDraftAndDurationScenesRestoreAtomicallyWithSourceOwner() throws {
        let host = MoldHost(name: "Workstation", baseURL: URL(string: "http://workstation:7680")!, apiKey: "PRIVATE_KEY")
        let owner = VideoJobOwner(host: host, instanceId: "installation")
        var ordinary = RenderDraft(); ordinary.prompt = #"literal\new"#
        ordinary.seed = UInt64.max
        let source = VideoJobSource(filename: "ordinary-video.mp4", expectedArchiveIdentitySha256: String(repeating: "a", count: 64))
        let authoring = VideoDurationAuthoring(draft: .init(seconds: "30", scenes: [.init(prompt: "scene", seconds: "30")], source: source),
            sourceOwner: .init(owner: owner, title: "Original"))
        var descriptor = DraftDescriptor(ordinary, model: "model", family: nil, recipeID: nil)
        descriptor.videoDuration = authoring
        let data = try MoldJSON.localEncoder.encode(descriptor)
        let restored = try MoldJSON.localDecoder.decode(DraftDescriptor.self, from: data)
        #expect(restored.videoDuration == authoring)
        #expect(restored.seed == UInt64.max)
        #expect(restored.prompt == ordinary.prompt)
        #expect(String(data: data, encoding: .utf8)?.contains("PRIVATE_KEY") == false)
    }
    @Test func restoredLibrarySourceCannotMoveToAnotherOriginOrInstallation() throws {
        let host = MoldHost(name: "Workstation", baseURL: URL(string: "http://workstation:7680")!)
        let owner = VideoJobOwner(host: host, instanceId: "installation")
        let authoring = VideoDurationAuthoring(draft: .init(seconds: "30", source: .init(filename: "ordinary.mp4")),
            sourceOwner: .init(owner: owner, title: "Original"))
        let generation = GenerateRequest(prompt: "new footage", model: "model", width: 768, height: 768, steps: 2, guidance: 1)
        #expect(try authoring.request(generation: generation, owner: owner).source?.filename == "ordinary.mp4")
        #expect(throws: (any Error).self) {
            try authoring.request(generation: generation, owner: .init(host: host, instanceId: "replacement"))
        }
        var moved = host; moved.baseURL = URL(string: "http://other:7680")!
        #expect(throws: (any Error).self) {
            try authoring.request(generation: generation, owner: .init(host: moved, instanceId: "installation"))
        }
    }
    @Test func continuousExtensionOwnsOnlyAnEmptyOpeningBoundary() throws {
        var authoring = VideoDurationAuthoring(draft: .init(seconds: "5", source: .init(filename: "original.mp4")))
        let plan = try MoldJSON.decoder.decode(VideoJobPlan.self, from: Data(VideoJobContractTests.plan().utf8))
        var generation = GenerateRequest(prompt: "tail", model: "model", width: 768, height: 768, steps: 2, guidance: 1)
        #expect(authoring.usesSourceEnding(plan: plan))
        #expect(!authoring.usesSourceEnding(plan: nil))
        #expect(authoring.openingConflict(plan: plan, generation: generation) == nil)
        generation.sourceImage = "new-opening"
        #expect(authoring.openingConflict(plan: plan, generation: generation) != nil)
        #expect(generation.sourceImage == "new-opening")
        let blended = try MoldJSON.decoder.decode(VideoJobPlan.self, from: Data(VideoJobContractTests.plan().replacingOccurrences(of: "continuous", with: "blended_shots").utf8))
        #expect(authoring.openingConflict(plan: blended, generation: generation) == nil)
        authoring.startFresh()
        #expect(authoring.openingConflict(plan: plan, generation: generation) == nil)
    }

    @Test func durationProjectionRetainsClosingFrameLocatorWithoutChangingAutomaticPreference() {
        var draft = RenderDraft(); draft.frames = 121; draft.fps = 24
        draft.supportsDurationPrediction = true; draft.prefersAutomaticDuration = true
        #expect(RenderRequest.one(draft, model: "video").frames == nil)
        #expect(RenderRequest.one(draft.durationGenerationDraft, model: "video").frames == 121)
        #expect(draft.prefersAutomaticDuration)
    }

}
