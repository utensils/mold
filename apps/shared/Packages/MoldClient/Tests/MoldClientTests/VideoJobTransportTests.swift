import Foundation
import Testing
@testable import MoldClient

private final class VideoParentTransport: StubTransport {
    nonisolated(unsafe) static var sent: [URLRequest] = []
    nonisolated(unsafe) static var cancelAccepted = true
    nonisolated(unsafe) static var bodies: [Data] = []
    override class var contentType: String { "application/json" }
    override class func response(for path: String) -> (status: Int, body: Data)? {
        if path.hasSuffix("/cancel") { return (202, Data("{\"accepted\":\(cancelAccepted)}".utf8)) }
        if path.hasSuffix("validate") {
            let plan = requestPlan()
            return (200, Data(plan.utf8))
        }
        let job = """
        {"id":"operation","state":"queued","plan":\(VideoJobContractTests.plan()),"completed_clips":"0",
         "active_child_id":null,"output_filename":null,"error":null,"error_detail":null,
         "created_at_ms":1,"updated_at_ms":2,"can_cancel":true,"can_resume":false,"can_delete":false}
        """
        if path == "/api/video-jobs", sent.last?.httpMethod == "GET" {
            let query = URLComponents(url: sent.last!.url!, resolvingAgainstBaseURL: false)?.queryItems
            let filtered = query?.first { $0.name == "recoverable_only" }?.value == "true"
            let older = job.replacingOccurrences(of: "operation", with: "older-failed").replacingOccurrences(of: "queued", with: "failed")
            let completed = (0..<50).map { job.replacingOccurrences(of: "operation", with: "completed-\($0)").replacingOccurrences(of: "queued", with: "completed") }
            let jobs = filtered ? [older] : completed
            return (200, Data(("{\"jobs\":[" + jobs.joined(separator: ",") + "],\"has_more\":false,\"next_cursor\":null}").utf8))
        }
        return (200, Data(job.utf8))
    }
    class func requestPlan() -> String {
        let plan = VideoJobContractTests.plan()
        return sent.last?.url?.query == "planning_only=true"
            ? plan.replacingOccurrences(of: "\"validation_revision\":\"revision\"", with: "\"validation_revision\":\"\"") : plan
    }
    override func startLoading() {
        Self.sent.append(request)
        var bytes = request.httpBody ?? Data()
        if let stream = request.httpBodyStream {
            stream.open(); defer { stream.close() }
            var buffer = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                guard count > 0 else { break }
                bytes.append(contentsOf: buffer.prefix(count))
            }
        }
        Self.bodies.append(bytes)
        super.startLoading()
    }
}

@Suite(.serialized) struct VideoJobTransportTests {
    func draft() -> VideoJobRequest {
        .init(generation: .init(prompt: #"C:\new\render"#, model: "model", width: 768, height: 768,
                                steps: 2, guidance: 1, seed: UInt64.max), durationMs: 30_000)
    }
    @Test func capturedDestinationFencesEveryParentReadAndValidationWithoutLeakingToOrdinaryRoutes() async throws {
        VideoParentTransport.sent = []
        let backend = VideoParentTransport.backend().pinningVideoDestination("captured-installation")
        _ = try await backend.validateVideoJob(draft())
        _ = try await backend.planVideoJob(draft())
        _ = try await backend.videoJob(id: "operation")
        _ = try await backend.listVideoJobs(limit: 50, cursor: nil)
        #expect(VideoParentTransport.sent.count == 4)
        #expect(VideoParentTransport.sent.allSatisfy { $0.value(forHTTPHeaderField: "x-mold-destination-instance") == "captured-installation" })
        #expect(backend.request("/api/status").value(forHTTPHeaderField: "x-mold-destination-instance") == nil)
    }

    @Test func unboundReadsAndConflictingMutationTargetsAreRefusedBeforeNetwork() async throws {
        VideoParentTransport.sent = []
        await #expect(throws: (any Error).self) { try await VideoParentTransport.backend().videoJob(id: "operation") }
        let pinned = VideoParentTransport.backend().pinningVideoDestination("captured")
        await #expect(throws: (any Error).self) { try await pinned.cancelVideoJob(id: "operation", destinationInstanceId: "replacement") }
        #expect(VideoParentTransport.sent.isEmpty)
    }

    @Test func checkLengthIsTheReadOnlyValidationRouteAndPreservesLiteralPrompt() async throws {
        VideoParentTransport.sent = []; VideoParentTransport.bodies = []
        let backend: any MoldBackend = VideoParentTransport.backend().pinningVideoDestination("target")
        let plan = try await backend.validateVideoJob(draft())
        #expect(plan.clipCount == 4)
        #expect(VideoParentTransport.sent.map { $0.url!.path } == ["/api/video-jobs/validate"])
        let body = try #require(JSONSerialization.jsonObject(with: VideoParentTransport.bodies[0]) as? [String: Any])
        let generation = try #require(body["generation"] as? [String: Any])
        #expect(generation["prompt"] as? String == draft().generation.prompt)
        #expect(VideoParentTransport.sent[0].value(forHTTPHeaderField: "x-mold-operation-id") == nil)
    }
    @Test func createRequiresValidationAndCapturesOperationAndDestination() async throws {
        VideoParentTransport.sent = []; VideoParentTransport.bodies = []
        let backend: any MoldBackend = VideoParentTransport.backend().pinningVideoDestination("target")
        await #expect(throws: (any Error).self) {
            try await backend.createVideoJob(draft(), operationId: "operation", destinationInstanceId: "target")
        }
        #expect(VideoParentTransport.sent.isEmpty)
        var request = draft(); request.validationRevision = "revision"
        let job = try await backend.createVideoJob(request, operationId: "operation", destinationInstanceId: "target")
        #expect(job.id == "operation")
        #expect(VideoParentTransport.sent[0].url?.path == "/api/video-jobs")
        #expect(VideoParentTransport.sent[0].value(forHTTPHeaderField: "x-mold-operation-id") == "operation")
        #expect(VideoParentTransport.sent[0].value(forHTTPHeaderField: "x-mold-destination-instance") == "target")
        let body = try #require(JSONSerialization.jsonObject(with: VideoParentTransport.bodies[0]) as? [String: Any])
        #expect(body["validation_revision"] as? String == "revision")
        #expect(body["duration_ms"] as? String == "30000")
    }
    @Test func statusEscapesExactParentIdInsteadOfUsingChildQueue() async throws {
        VideoParentTransport.sent = []
        let backend: any MoldBackend = VideoParentTransport.backend().pinningVideoDestination("target")
        _ = try await backend.videoJob(id: "a/b")
        #expect(VideoParentTransport.sent[0].url?.absoluteString.contains("/api/video-jobs/a%2Fb") == true)
    }
    @Test func automaticPreviewSendsOnlyScalarsAndCannotReturnAnAdmissionProof() async throws {
        VideoParentTransport.sent = []; VideoParentTransport.bodies = []
        let backend: any MoldBackend = VideoParentTransport.backend().pinningVideoDestination("target")
        var request = draft()
        request.generation.sourceImage = "PRIVATE_IMAGE_BYTES"
        request.generation.audioFile = "PRIVATE_AUDIO_BYTES"
        request.generation.sourceVideo = "PRIVATE_VIDEO_BYTES"
        request.generation.pipeline = "text-to-video"
        request.generation.negativePrompt = #"literal\new"#
        request.source = .init(filename: "original.mp4")
        let plan = try await backend.planVideoJob(request)
        let body = try #require(String(data: VideoParentTransport.bodies[0], encoding: .utf8))
        #expect(!body.contains("PRIVATE_"))
        #expect(VideoParentTransport.sent[0].url?.query == "planning_only=true")
        let object = try #require(JSONSerialization.jsonObject(with: VideoParentTransport.bodies[0]) as? [String: Any])
        let generation = try #require(object["generation"] as? [String: Any])
        #expect(generation["pipeline"] as? String == "text-to-video")
        #expect(generation["negative_prompt"] as? String == request.generation.negativePrompt)
        #expect((object["source"] as? [String: Any])?["filename"] as? String == "original.mp4")
        #expect(plan.validationRevision.isEmpty)
        #expect(request.generation.sourceImage == "PRIVATE_IMAGE_BYTES")
    }

    @Test func queueFilteringPrecedesLimitAndOpaqueCursorIsNotAnOffset() async throws {
        VideoParentTransport.sent = []
        let backend: any MoldBackend = VideoParentTransport.backend().pinningVideoDestination("target")
        let cursor = "opaque+/=&position"
        let listing = try await backend.listVideoJobs(limit: 50, cursor: cursor)
        #expect(listing.jobs.map(\.id) == ["older-failed"])
        #expect(!listing.hasMore && listing.nextCursor == nil)
        let url = try #require(VideoParentTransport.sent[0].url)
        let query = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems)
        #expect(query.first { $0.name == "recoverable_only" }?.value == "true")
        #expect(query.first { $0.name == "cursor" }?.value == cursor)
        #expect(query.allSatisfy { $0.name != "offset" })
    }

    @Test func retainedInputSessionRemainsHeaderOnlyForValidationAndCreate() async throws {
        VideoParentTransport.sent = []; VideoParentTransport.bodies = []
        let backend: any MoldBackend = VideoParentTransport.backend().pinningVideoDestination("target")
        var request = draft(); request.retainedMediaSession = "private-session"
        _ = try await backend.validateVideoJob(request)
        request.validationRevision = "revision"
        _ = try await backend.createVideoJob(request, operationId: "operation", destinationInstanceId: "target")
        #expect(VideoParentTransport.sent.allSatisfy { $0.value(forHTTPHeaderField: "x-mold-retained-media-session") == "private-session" })
        #expect(VideoParentTransport.bodies.allSatisfy { !String(decoding: $0, as: UTF8.self).contains("private-session") })
        #expect(request.planningProjection.retainedMediaSession == nil)
    }

    @Test func literalParentReuseSessionProtectsItsNestedLegacyWireOnly() async throws {
        VideoParentTransport.sent = []; VideoParentTransport.bodies = []
        let backend = VideoParentTransport.backend().pinningVideoDestination("target").preparingLiteralVideoInputs()
        var generation = draft().generation
        generation.prompt = "literal\\new\r\nnext"
        generation.negativePrompt = "avoid\\render\r\nnext"
        generation.originalPrompt = "original\\root"
        generation.promptTransform = .init(operation: .expand, rootPrompt: "root\\name", sourcePrompt: "source\\name", task: .textToVideo)
        // This transport deliberately has no reuse-session response; capture
        // the outbound boundary rather than minting a fictitious credential.
        await #expect(throws: (any Error).self) {
            try await backend.retainedMediaReuseSession(for: "movie.mp4", members: ["member"], target: generation)
        }
        let body = try #require(JSONSerialization.jsonObject(with: VideoParentTransport.bodies[0]) as? [String: Any])
        let scoped = try #require(body["target_request"] as? [String: Any])
        #expect(scoped["prompt"] as? String == generation.protectedForLegacyPromptWire.prompt)
        #expect(scoped["negative_prompt"] as? String == generation.protectedForLegacyPromptWire.negativePrompt)
        #expect(scoped["original_prompt"] as? String == generation.protectedForLegacyPromptWire.originalPrompt)
        let transform = try #require(scoped["prompt_transform"] as? [String: Any])
        #expect(transform["root_prompt"] as? String == generation.protectedForLegacyPromptWire.promptTransform?.rootPrompt)
        #expect(transform["source_prompt"] as? String == generation.protectedForLegacyPromptWire.promptTransform?.sourcePrompt)
        #expect(generation.prompt == "literal\\new\r\nnext")
    }

    @Test func stopRequiresExplicitAcceptanceBeforeItsReceiptCanBeAcknowledged() async throws {
        VideoParentTransport.sent = []; VideoParentTransport.cancelAccepted = false
        defer { VideoParentTransport.cancelAccepted = true }
        let backend = VideoParentTransport.backend().pinningVideoDestination("target")
        await #expect(throws: (any Error).self) {
            try await backend.cancelVideoJob(id: "operation", destinationInstanceId: "installation")
        }
    }

}
