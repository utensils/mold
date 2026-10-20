import Foundation
import Testing
@testable import MoldClient

private final class VideoParentTransport: StubTransport {
    nonisolated(unsafe) static var sent: [URLRequest] = []
    nonisolated(unsafe) static var bodies: [Data] = []
    override class var contentType: String { "application/json" }
    override class func response(for path: String) -> (status: Int, body: Data)? {
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
    @Test func checkLengthIsTheReadOnlyValidationRouteAndPreservesLiteralPrompt() async throws {
        VideoParentTransport.sent = []; VideoParentTransport.bodies = []
        let backend: any MoldBackend = VideoParentTransport.backend()
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
        let backend: any MoldBackend = VideoParentTransport.backend()
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
        let backend: any MoldBackend = VideoParentTransport.backend()
        _ = try await backend.videoJob(id: "a/b")
        #expect(VideoParentTransport.sent[0].url?.absoluteString.contains("/api/video-jobs/a%2Fb") == true)
    }
    @Test func automaticPreviewSendsOnlyScalarsAndCannotReturnAnAdmissionProof() async throws {
        VideoParentTransport.sent = []; VideoParentTransport.bodies = []
        let backend: any MoldBackend = VideoParentTransport.backend()
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
        let backend: any MoldBackend = VideoParentTransport.backend()
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

}
