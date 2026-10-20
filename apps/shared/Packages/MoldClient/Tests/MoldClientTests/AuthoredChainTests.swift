import Foundation
import Testing
@testable import MoldClient

struct AuthoredChainTests {
    func limits(linked: Bool = true, sourceFit: Bool? = nil) throws -> ChainLimits {
        try MoldJSON.decoder.decode(ChainLimits.self, from: Data("""
        {"model":"h3-test","fps":24,"frames_per_clip_cap":345,"frames_per_clip_recommended":209,
        "max_stages":16,"max_total_frames":5520,"supports_sequence":true,"supports_audio":true,
        "transition_modes":["smooth","cut","fade"],"fade_frames_max":32,
        "linked_parts":{"supported":\(linked),"max_stages_per_part":16,"max_parts":null},
        "authoring":\(sourceFit.map { "{\"motion_tail_frames\":1,\"carry_mode\":\"rgb\",\"min_frames\":107,\"frame_step\":17,\"frame_offset\":5,\"source_fit_supported\":\($0)}" } ?? "null")}
        """.utf8))
    }
    func base() -> GenerateRequest {
        var value = GenerateRequest(prompt: "First", model: "h3-test", width: 768, height: 512,
                                    steps: 9, guidance: 0, seed: UInt64.max)
        value.fps = 24; value.enableAudio = true
        return value
    }
    @Test func authoredRequestKeepsTypedMediaAndFullSeed() throws {
        var stage = ChainStageInput(prompt: "First", frames: 209)
        stage.keyframes = [.init(frame: 208, image: "aW1hZ2U=", name: "end")]
        let request = try AuthoredChainRequest(base(), stages: [stage], limits: limits())
        let data = try MoldJSON.encoder.encode(request)
        let wire = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(wire["ephemeral"] == nil)
        #expect(wire["output_mode"] as? String == "sequence")
        let restored = try MoldJSON.decoder.decode(AuthoredChainRequest.self, from: data)
        #expect(restored.seed == UInt64.max)
        #expect(restored.stages.first?.keyframes?.first?.frame == 208)
    }
    @Test func draftEditingPreservesMediaAndNeverRemovesLastClip() throws {
        var draft = SequenceDraft(prompt: "First", frames: 209)
        draft.stages[0].keyframes = [.init(frame: 208, image: "aW1hZ2U=")]
        draft.append(); draft.stages[1].prompt = "Second"
        draft.move(from: 0, to: 1)
        #expect(draft.stages[1].keyframes?.count == 1)
        draft.remove(id: draft.stages[0].id); draft.remove(id: draft.stages[0].id)
        #expect(draft.stages.count == 1)
        #expect(try draft.request(base: base(), limits: limits()).stages[0].prompt == "First")
    }
    @Test func unsupportedControlsAreRefusedRatherThanDiscarded() throws {
        var request = base(); request.audioFile = "YXVkaW8="
        #expect(throws: SequenceDraftError.self) {
            try AuthoredChainRequest(request, stages: [.init(prompt: "First", frames: 209)], limits: limits())
        }
    }
    @Test func interruptedJobsRemainRecoverableAndTakeIsExact() throws {
        let data = Data("""
        {"id":"job-one","state":"interrupted","model":"h3-test","stage_count":1,"current_stage":0,
         "finalizes":[{"output":"internal.mp4","gallery_filename":"first.mp4","take":1}],"stages":[]}
        """.utf8)
        let detail = try MoldJSON.decoder.decode(ChainJobDetail.self, from: data)
        #expect(detail.state == .interrupted)
        #expect(detail.latestFinalize?.take == 1)
        #expect(detail.galleryFilename == "first.mp4")
    }
    @Test func olderHostCannotAccidentallyStartAnIndependentContinuation() throws {
        var request = try AuthoredChainRequest(base(), stages: [.init(prompt: "Next", frames: 209)], limits: limits())
        request.continueFrom = .init(jobId: "parent", finalizeTake: 1)
        #expect(throws: SequenceDraftError.self) { try request.validateSubmission(limits: limits(linked: false)) }
    }
    @Test func sourceAuthorityLocksRecipeAndRejectsOtherOwner() throws {
        let detail = try MoldJSON.decoder.decode(ChainJobDetail.self, from: Data("""
        {"id":"parent","state":"completed","model":"h3-test","stage_count":1,"current_stage":1,
        "finalizes":[{"gallery_filename":"older-1.mp4"},{"gallery_filename":"older-2.mp4"},{"gallery_filename":"first.mp4"}],
        "script":{"schema":"mold.chain.v1","chain":{"model":"h3-test","width":768,"height":512,
        "fps":24,"seed":18446744073709551615,"steps":9,"guidance":0,"strength":1,
        "motion_tail_frames":1,"output_format":"mp4","enable_audio":true},
        "stage":[{"prompt":"Original","frames":209,"transition":"smooth"}]}}
        """.utf8))
        #expect(throws: SequenceDraftError.self) { try ChainContinuationSource(hostId: UUID(), instanceId: nil, detail: detail, limits: limits()) }
        #expect(throws: SequenceDraftError.self) { try ChainContinuationSource(hostId: UUID(), instanceId: "", detail: detail, limits: limits()) }
        #expect(throws: SequenceDraftError.self) { try ChainContinuationSource(hostId: UUID(), instanceId: "  ", detail: detail, limits: limits()) }
        let host = UUID()
        let source = try ChainContinuationSource(hostId: host, instanceId: "one", detail: detail, limits: limits())
        let request = try source.request(stages: [.init(prompt: "Next", frames: 209)])
        #expect(request.continueFrom == .init(jobId: "parent", finalizeTake: 3))
        #expect(request.motionTailFrames == 1)
        #expect(request.enableAudio == true)
        #expect(request.width == 768 && request.height == 512)
        #expect(source.script.chain.seed == UInt64.max)
        #expect(throws: SequenceDraftError.self) { try source.requireOwner(hostId: UUID(), instanceId: "one") }
        #expect(throws: SequenceDraftError.self) { try source.requireOwner(hostId: host, instanceId: "two") }
    }
    @Test func authoredInitializerPreservesComposerInputsWithoutRepeatingSource() throws {
        var base = base(); base.sourceImage = "aW1hZ2U="
        base.negativePrompt = "blur"; base.loras = [.init(path: "/adapter", name: "style")]
        let request = try AuthoredChainRequest(base, stages: [.init(prompt: "One", frames: 209),
            .init(prompt: "Two", frames: 209)], limits: limits())
        #expect(request.stages[0].sourceImage == base.sourceImage)
        #expect(request.stages[1].sourceImage == nil)
        #expect(request.stages.allSatisfy { $0.negativePrompt == "blur" && $0.loras?.first?.path == "/adapter" })
    }
    @Test func latestFinalizeDoesNotFallBackToOlderGalleryResult() throws {
        let detail = try MoldJSON.decoder.decode(ChainJobDetail.self, from: Data("""
        {"id":"parent","state":"completed","model":"h3-test","stage_count":1,"current_stage":1,
         "finalizes":[{"gallery_filename":"old.mp4","take":1},{"take":2}]}
        """.utf8))
        #expect(detail.galleryFilename == nil)
        #expect(throws: SequenceDraftError.self) {
            try ChainContinuationSource(hostId: UUID(), instanceId: nil, detail: detail, limits: limits())
        }
    }
    @Test func smoothClipMustContainNewFrames() throws {
        var request = try AuthoredChainRequest(base(), stages: [.init(prompt: "One", frames: 209)], limits: limits())
        request.motionTailFrames = 209
        request.stages.append(.init(prompt: "Two", frames: 209))
        #expect(throws: SequenceDraftError.self) { try request.validateSubmission(limits: limits()) }
    }

    @Test func realServerFinalizeRecordsDeriveTheLatestOneBasedTake() throws {
        let detail = try MoldJSON.decoder.decode(ChainJobDetail.self, from: Data("""
        {"id":"parent","state":"completed","model":"h3-test","stage_count":1,"current_stage":1,
        "finalizes":[{"output":"final/output-1.mp4","gallery_filename":"one.mp4","at_unix_ms":1,"stage_seeds":["18446744073709551615"]},
        {"output":"final/output-2.mp4","gallery_filename":"two.mp4","at_unix_ms":2,"stage_seeds":["18446744073709551615"]}]}
        """.utf8))
        #expect(detail.latestFinalize?.take == 2)
        #expect(detail.galleryFilename == "two.mp4")
    }

    @Test func globalReferencesAreOnlyRepeatedWhenExplicitlySelected() throws {
        var base = base()
        base.references = [.init(kind: "image", media: .init(authority: "inline", data: "aW1hZ2U="), mimeType: "image/png")]
        var draft = SequenceDraft(prompt: "One", frames: 209); draft.append(); draft.stages[1].prompt = "Two"
        #expect(try draft.request(base: base, limits: limits(), repeatReferences: false).stages[1].generationReferences == nil)
        #expect(try draft.request(base: base, limits: limits(), repeatReferences: true).stages[1].generationReferences == base.references)
    }
    @Test func replicatedAuthoredMediaBudgetIsCheckedBeforeEncoding() throws {
        var request = try AuthoredChainRequest(base(), stages: [.init(prompt: "One", frames: 209)], limits: limits())
        // Base64 representation of just over 32 MiB, counted without decoding.
        request.stages[0].sourceImage = String(repeating: "A", count: 44_739_248)
        #expect(throws: SequenceDraftError.self) { try request.validateSubmission(limits: limits()) }
    }

    @Test func globalClosingFrameEndsTheFinalClipAtItsOwnLength() throws {
        var base = base(); base.frames = 209
        base.keyframes = [.init(frame: 0, image: "Zmlyc3Q="), .init(frame: 208, image: "bGFzdA==")]
        var draft = SequenceDraft(prompt: "One", frames: 209); draft.append()
        draft.stages[1].prompt = "Two"; draft.stages[1].frames = 192
        let request = try draft.request(base: base, limits: limits())
        #expect(request.stages[0].keyframes?.map(\.frame) == [0])
        #expect(request.stages[1].keyframes?.map(\.frame) == [191])
        #expect(request.stages[1].keyframes?.first?.image == "bGFzdA==")
    }

    @Test func chainInputsUseOriginalInlineReferenceInsteadOfTransientUpload() throws {
        var base = base()
        let reference = GenerationReference(kind: "image", media: .init(authority: "inline", data: "aW1hZ2U="), mimeType: "image/png")
        base.pendingReferenceInputs = [.canonical(reference)]
        let request = try AuthoredChainRequest(base, stages: [.init(prompt: "One", frames: 209)], limits: limits())
        #expect(request.stages[0].generationReferences == [reference])
        base.pendingReferenceInputs = nil
        base.references = [.init(kind: "image", media: .init(authority: "upload", handle: "secret"), mimeType: "image/png")]
        #expect(throws: SequenceDraftError.self) {
            try AuthoredChainRequest(base, stages: [.init(prompt: "One", frames: 209)], limits: limits())
        }
    }

    @Test func sourceFitSurvivesAuthoredRequestAndScriptProjection() throws {
        var base = base(); base.sourceFit = .default
        #expect(throws: SequenceDraftError.self) {
            try AuthoredChainRequest(base, stages: [.init(prompt: "One", frames: 209)], limits: limits())
        }
        #expect(throws: SequenceDraftError.self) {
            try AuthoredChainRequest(base, stages: [.init(prompt: "One", frames: 209)], limits: limits(sourceFit: false))
        }
        let request = try AuthoredChainRequest(base, stages: [.init(prompt: "One", frames: 209)], limits: limits(sourceFit: true))
        let json = try #require(JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(request)) as? [String: Any])
        #expect(json["source_fit"] != nil)
        let data = Data("""
        {"schema":"mold.chain.v1","chain":{"model":"h3-test","width":768,"height":512,"fps":24,
        "steps":9,"guidance":0,"strength":1,"motion_tail_frames":1,"output_format":"mp4",
        "source_fit":{"mode":"crop-fill","alignX":"center","alignY":"center"}},
        "stage":[{"prompt":"One","frames":209,"transition":"smooth"}]}
        """.utf8)
        let script = try MoldJSON.decoder.decode(ChainScript.self, from: data)
        #expect(script.chain.sourceFit == .default)
    }

}
