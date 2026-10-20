import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

@MainActor struct VideoJobCoordinatorTests {
    func setup() throws -> (FakeBackend, VideoJobOwner, UserDefaults) {
        let backend = FakeBackend()
        backend.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self,
            from: Data(#"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"installation"}"#.utf8)))
        backend.stub("capabilities()", returning: try MoldJSON.decoder.decode(Capabilities.self,
            from: Data(#"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true}}"#.utf8)))
        backend.stub("prepareVideoJob(_:)") { args in
            PreparedVideoJob(request: args[0] as! VideoJobRequest)
        }
        backend.stub("validateVideoJob(_:)", returning: try MoldJSON.decoder.decode(VideoJobPlan.self, from: Data(VideoJobContractTests.plan().utf8)))
        return (backend, .init(host: backend.host, instanceId: "installation"), UserDefaults(suiteName: "video-parent-\(UUID())")!)
    }
    func request() -> VideoJobRequest {
        .init(generation: .init(prompt: #"camera\new"#, model: "model", width: 768, height: 768, steps: 2, guidance: 1), durationMs: 30_000)
    }
    func detail(_ id: String, state: String = "running", cancel: Bool = true, resume: Bool = false, delete: Bool = false) throws -> VideoJob {
        try MoldJSON.decoder.decode(VideoJob.self, from: Data("""
        {"id":"\(id)","state":"\(state)","plan":\(VideoJobContractTests.plan()),"completed_clips":"0",
         "active_child_id":null,"output_filename":null,"error":null,"error_detail":null,
         "created_at_ms":1,"updated_at_ms":2,"can_cancel":\(cancel),"can_resume":\(resume),"can_delete":\(delete)}
        """.utf8))
    }

    @Test func preparedRequestIsFreshlyValidatedAndOperationIsSavedBeforeCreate() async throws {
        let (backend, owner, saved) = try setup()
        let read: @MainActor @Sendable () -> Data? = { saved.data(forKey: VideoJobCoordinator.journalKey) }
        let ready = try detail("unused")
        backend.stub("createVideoJob(_:operationId:destinationInstanceId:)") { args in
            let request = try #require(args[0] as? VideoJobRequest)
            #expect(request.generation.sourceImage == "prepared-pixels")
            #expect(request.validationRevision == "revision")
            let data = try #require(await read())
            let receipt = try MoldJSON.decoder.decode([VideoJobAdmission].self, from: data)
            #expect(receipt.first?.operationId == args[1] as? String)
            #expect(String(data: data, encoding: .utf8)?.contains("prepared-pixels") == false)
            let json = try MoldJSON.encoder.encode(ready)
            let text = String(data: json, encoding: .utf8)!.replacingOccurrences(of: "unused", with: args[1] as! String)
            return try MoldJSON.decoder.decode(VideoJob.self, from: Data(text.utf8))
        }
        let coordinator = VideoJobCoordinator(defaults: saved)
        await coordinator.submit(request(), owner: owner, backend: backend, current: { true }, unchanged: { true }, prepare: {
            var prepared = request(); prepared.generation.sourceImage = "prepared-pixels"; return prepared
        })
        #expect(coordinator.pending.count == 1)
        #expect(coordinator.jobs.first?.detail?.state == .running)
        #expect(coordinator.error == nil)
        coordinator.suspend()
    }

    @Test func editedDraftDuringPreparationCannotCreateOrLeaveBusy() async throws {
        let (backend, owner, saved) = try setup()
        let coordinator = VideoJobCoordinator(defaults: saved)
        var unchanged = true
        await coordinator.submit(request(), owner: owner, backend: backend, current: { true }, unchanged: { unchanged }, prepare: {
            unchanged = false; return request()
        })
        #expect(backend.count("createVideoJob(_:operationId:destinationInstanceId:)") == 0)
        #expect(coordinator.pending.isEmpty)
        #expect(!coordinator.isSubmitting)
    }

    @Test func lostAdmissionIsRecoveredByExactUuidWithoutAnotherCreate() async throws {
        let (backend, owner, saved) = try setup()
        backend.stub("createVideoJob(_:operationId:destinationInstanceId:)", throwing: MoldClientError.unreachable("Lost reply"))
        let original = VideoJobCoordinator(defaults: saved)
        await original.submit(request(), owner: owner, backend: backend, current: { true }, unchanged: { true })
        let receipt = try #require(original.pending.first)
        original.suspend()
        backend.stub("videoJob(id:)", returning: try detail(receipt.operationId, state: "paused", cancel: true, resume: true))
        let recovered = VideoJobCoordinator(defaults: saved)
        await recovered.recover(owner: owner, backend: backend, current: { true })
        #expect(recovered.jobs.first?.detail?.id == receipt.operationId)
        #expect(backend.count("createVideoJob(_:operationId:destinationInstanceId:)") == 1)
        recovered.suspend()
    }
    @Test func definitiveAdmissionRejectionRemovesReceiptAndPlaceholder() async throws {
        let (backend, owner, saved) = try setup()
        backend.stub("createVideoJob(_:operationId:destinationInstanceId:)", throwing: MoldClientError.http(status: 422, code: "INVALID_VIDEO", message: "Bad inputs"))
        let coordinator = VideoJobCoordinator(defaults: saved)
        await coordinator.submit(request(), owner: owner, backend: backend, current: { true }, unchanged: { true })
        #expect(coordinator.pending.isEmpty)
        #expect(coordinator.jobs.isEmpty)
        #expect(coordinator.error != nil)
    }

    @Test func transportPreparationLivesAcrossValidationAndIsReleasedAfterRejection() async throws {
        let (backend, owner, saved) = try setup()
        actor Release { var count = 0; func done() { count += 1 } }
        let release = Release()
        backend.stub("prepareVideoJob(_:)") { args in
            var value = args[0] as! VideoJobRequest
            value.retainedMediaSession = "ephemeral-session"
            return PreparedVideoJob(request: value, release: { await release.done() })
        }
        backend.stub("validateVideoJob(_:)") { args in
            #expect((args[0] as? VideoJobRequest)?.retainedMediaSession == "ephemeral-session")
            throw MoldClientError.http(status: 422, code: "INVALID_VIDEO", message: "Refused")
        }
        let coordinator = VideoJobCoordinator(defaults: saved)
        await coordinator.submit(request(), owner: owner, backend: backend, current: { true }, unchanged: { true })
        #expect(backend.count("prepareVideoJob(_:)") == 1)
        #expect(await release.count == 1)
        #expect(coordinator.pending.isEmpty)
    }

    @Test func ambiguousAdmissionCannotMintAnotherUuidUntilExactReconciliation() async throws {
        let (backend, owner, saved) = try setup()
        backend.stub("createVideoJob(_:operationId:destinationInstanceId:)", throwing: MoldClientError.unreachable("Lost reply"))
        let coordinator = VideoJobCoordinator(defaults: saved)
        await coordinator.submit(request(), owner: owner, backend: backend, current: { true }, unchanged: { true })
        let id = try #require(coordinator.pending.first?.operationId)
        await coordinator.submit(request(), owner: owner, backend: backend, current: { true }, unchanged: { true })
        #expect(backend.count("createVideoJob(_:operationId:destinationInstanceId:)") == 1)
        #expect(coordinator.pending.count == 1)
        backend.stub("videoJob(id:)", returning: try detail(id, state: "paused", resume: true))
        await coordinator.refresh(id, owner: owner, backend: backend, current: { true })
        await coordinator.submit(request(), owner: owner, backend: backend, current: { true }, unchanged: { true })
        #expect(backend.count("createVideoJob(_:operationId:destinationInstanceId:)") == 2)
    }

    @Test func ambiguousPostKeepsOnlyLeaseCleanupUntilExactParentIsKnown() async throws {
        let (backend, owner, saved) = try setup()
        actor Release { var count = 0; func done() { count += 1 } }
        let release = Release()
        backend.stub("prepareVideoJob(_:)") { args in
            PreparedVideoJob(request: args[0] as! VideoJobRequest, release: { await release.done() })
        }
        backend.stub("createVideoJob(_:operationId:destinationInstanceId:)", throwing: MoldClientError.unreachable("Lost reply"))
        let coordinator = VideoJobCoordinator(defaults: saved)
        await coordinator.submit(request(), owner: owner, backend: backend, current: { true }, unchanged: { true })
        #expect(await release.count == 0)
        let id = try #require(coordinator.pending.first?.operationId)
        backend.stub("videoJob(id:)", returning: try detail(id, state: "paused", resume: true))
        await coordinator.refresh(id, owner: owner, backend: backend, current: { true })
        #expect(await release.count == 1)
    }

}
