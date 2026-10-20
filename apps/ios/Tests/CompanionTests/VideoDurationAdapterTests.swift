import Foundation
import MoldClient
import MoldClientTesting
import Testing

@testable import MoldCompanion

extension GenerateControllerTests {
    @Test func explicitReuseCannotKeepAnEarlierVideoPrefix() async throws {
        let (generate, _) = try await setUp()
        let host = try #require(generate.hosts.hosts.first)
        let owner = VideoJobOwner(host: host, instanceId: "earlier-installation")
        generate.videoDuration = .init(draft: .init(seconds: "30", source: .init(filename: "earlier.mp4")),
            sourceOwner: .init(owner: owner, title: "Earlier"))
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
            #"{"filename":"selected.mp4","metadata":{"prompt":"Selected new settings","model":"ltx-2.5-22b-distilled:bf16"},"timestamp":1790000000,"format":"mp4"}"#.utf8))
        generate.reuse(LibraryEntry(host: host, print: print))
        #expect(generate.videoDuration == nil, "Use These Settings starts a fresh movie instead of silently extending an earlier prefix")
    }

    @Test func durationSubmissionKeepsLiteralPreparedFieldsAndLostReplyNeverFallsBack() async throws {
        var bound: FakeBackend?
        let (original, _) = try await setUp { host, _ in
            if bound == nil { bound = FakeBackend(host: host) }
            return bound!
        }
        let host = try #require(original.hosts.hosts.first)
        let backend = try #require(bound)
        let capabilities = try MoldJSON.decoder.decode(Capabilities.self, from: Data(
            #"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true}}"#.utf8))
        let status = try MoldJSON.decoder.decode(ServerStatus.self, from: Data(
            #"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"installation"}"#.utf8))
        original.hosts.setReachability(.up(status), for: host.id)
        original.hosts.setCapabilities(capabilities, for: host.id)
        let clip = try model("ltx-2.5-22b-distilled:bf16")
        original.hosts.setModels([clip], for: host.id)
        backend.stub("status()", returning: status)
        backend.stub("capabilities()", returning: capabilities)
        backend.stub("prepareVideoJob(_:)") { args in PreparedVideoJob(request: args[0] as! VideoJobRequest) }
        let plan = #"{"validation_revision":"proof","model":"model","width":768,"height":768,"fps":24,"single_clip_max_frames":121,"overlap_frames":17,"join_mode":"continuous","clip_count":"7","part_count":"1","added_frames":"720","source_frames":"0","total_frames":"720","added_duration_ms":"30000","total_duration_ms":"30000","warnings":[]}"#
        backend.stub("validateVideoJob(_:)", returning: try MoldJSON.decoder.decode(VideoJobPlan.self, from: Data(plan.utf8)))
        backend.stub("createVideoJob(_:operationId:destinationInstanceId:)", throwing: URLError(.networkConnectionLost))
        let jobs = VideoJobCoordinator(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let generate = GenerateController(hosts: original.hosts, ledger: original.ledger,
            drafts: original.drafts, initialMachine: .auto,
            sequence: NativeSequenceSession(defaults: UserDefaults(suiteName: UUID().uuidString)!), videoJobs: jobs)
        generate.setKind(.clip); generate.choose(clip)
        generate.draft.prompt = #"A literal \new scene"#
        generate.draft.title = "Filed movie"; generate.draft.tags = ["uat"]
        generate.draft.collectionName = "Video UAT"
        generate.videoDuration = .init(draft: .init(seconds: "30"))
        generate.generate()
        await generate.durationTask?.value
        let call = try #require(backend.calls.first { $0.route == "createVideoJob(_:operationId:destinationInstanceId:)" })
        let request = try #require(call.arguments.first as? VideoJobRequest)
        #expect(request.durationMs == 30_000 && request.validationRevision == "proof")
        #expect(request.generation.prompt == generate.draft.prompt)
        #expect(request.generation.title == "Filed movie" && request.generation.tags?.contains("uat") == true)
        #expect(request.generation.collection == .named("Video UAT"))
        #expect(jobs.pending.count == 1)
        #expect(backend.count("submit(_:)") == 0, "uncertain parent creation must never fall back to one-shot admission")
        #expect(generate.durationTask == nil)
        generate.generate()
        await generate.durationTask?.value
        #expect(backend.count("createVideoJob(_:operationId:destinationInstanceId:)") == 1,
            "Check the known operation before another press can create a duplicate")
        jobs.suspend()
    }

    @Test func durationStartsAtRecipeLengthAndSurvivesRecipeChanges() async throws {
        let (generate, _) = try await setUp()
        let clip = try model("ltx-2.5-22b-distilled:bf16")
        let host = try #require(generate.hosts.hosts.first)
        generate.hosts.setModels([clip], for: host.id)
        generate.setKind(.clip)
        generate.draft.frames = 121
        generate.draft.fps = 24
        generate.ensureVideoDuration()
        #expect(generate.videoDuration?.draft.seconds == "5.041")
        generate.videoDuration?.draft.seconds = "30"
        generate.chooseRecipe(generate.recipe!.id)
        generate.ensureVideoDuration()
        #expect(generate.videoDuration?.draft.seconds == "30")
    }

    @Test func durationScenesAndOriginalOwnerRestoreTogetherWithoutAReadyMachine() async throws {
        let (generate, _) = try await setUp()
        let host = try #require(generate.hosts.hosts.first)
        let owner = VideoJobOwner(host: host, instanceId: "original-installation")
        var duration = VideoDurationDraft(seconds: "30", scenes: [.init(prompt: "New scene", seconds: "30")])
        duration.source = .init(filename: "original.mp4")
        generate.videoDuration = .init(draft: duration, sourceOwner: .init(owner: owner, title: "Original"))
        generate.saveDraft()
        generate.hosts.setReachability(nil, for: host.id)
        let restored = GenerateController(hosts: generate.hosts, ledger: generate.ledger,
            drafts: generate.drafts, initialMachine: .auto)
        #expect(restored.videoDuration == generate.videoDuration)
        #expect(restored.durationOwner == nil)
        #expect(restored.durationBlocker != nil, "a disconnected restored duration must never fall through to ordinary generation")
    }

    @Test func durationPreservesPreparedSiblingInputsAndSeedProvenance() async throws {
        let (generate, _) = try await setUp()
        let host = try #require(generate.hosts.hosts.first)
        let owner = VideoJobOwner(host: host, instanceId: "original-installation")
        var prepared = generate.draft
        prepared.batchSize = 3
        prepared.locksSeed = true; prepared.seed = UInt64.max
        prepared.title = "A new movie"; prepared.negativePrompt = #"Keep \ literal"#
        let authoring = VideoDurationAuthoring(draft: .init(seconds: "30"))
        let requests = try generate.durationRequests(from: prepared, authoring: authoring,
            owner: owner, model: "selected-model", copies: 3, randomBase: 1)
        #expect(requests.count == 3)
        #expect(requests.map(\.generation.seed) == [UInt64.max, 0, 1])
        #expect(requests.allSatisfy { $0.durationMs == 30_000 && $0.generation.batchSize == 1 })
        #expect(requests.allSatisfy { $0.generation.negativePrompt == prepared.negativePrompt && $0.generation.title == prepared.title })
        #expect(Set(requests.compactMap(\.generation.batchId)).count == 1)
        #expect(requests.map(\.generation.batchIndex) == [1, 2, 3])
        #expect(requests.map(\.generation.batchCount) == [3, 3, 3])
    }

    @Test func durationCapabilityRequiresLiteralPromptsAndKeepsLegacySequenceSeparate() async throws {
        let (generate, _) = try await setUp()
        let clip = try model("ltx-2.5-22b-distilled:bf16")
        let host = try #require(generate.hosts.hosts.first)
        generate.hosts.setModels([clip], for: host.id)
        generate.setKind(.clip)
        #expect(!generate.canAuthorVideoDuration)
        let cap = try MoldJSON.decoder.decode(Capabilities.self, from: Data(
            #"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true}}"#.utf8))
        generate.hosts.setCapabilities(cap, for: host.id)
        #expect(generate.canAuthorVideoDuration)
        generate.sequence.begin(prompt: "Existing authored sequence", frames: 121)
        #expect(!generate.canAuthorVideoDuration)
    }
}
