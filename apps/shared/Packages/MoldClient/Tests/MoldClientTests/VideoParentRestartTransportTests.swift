import Foundation
import Testing

@testable import MoldClient

private final class ParentRestartTransport: StubTransport {
    nonisolated(unsafe) static var sent: [URLRequest] = []
    nonisolated(unsafe) static var bodies: [Data] = []
    override func startLoading() {
        Self.sent.append(request)
        var bytes = request.httpBody ?? Data()
        if let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var chunk = [UInt8](repeating: 0, count: 8192)
            while stream.hasBytesAvailable {
                let read = stream.read(&chunk, maxLength: chunk.count)
                if read <= 0 { break }
                bytes.append(contentsOf: chunk.prefix(read))
            }
        }
        Self.bodies.append(bytes)
        super.startLoading()
    }
    override class var contentType: String { "application/json" }
    override class func response(for path: String) -> (status: Int, body: Data)? {
        if path.hasSuffix("restart-source") { return (200, Data([1, 2, 3])) }
        if path.hasSuffix("validate") { return (200, Data(VideoJobContractTests.plan().utf8)) }
        if path == "/api/video-jobs" {
            return (
                202,
                Data(
                    """
                    {"id":"operation","state":"queued","plan":\(VideoJobContractTests.plan()),"completed_clips":"0","created_at_ms":1,"updated_at_ms":2,"can_cancel":true,"can_resume":false,"can_delete":false}
                    """.utf8)
            )
        }
        if path.hasSuffix("complete") {
            return (
                200, Data(#"{"handle":"upload","sha256":"\#(String(repeating:"b",count:64))","size_bytes":"3"}"#.utf8)
            )
        }
        return (200, Data(#"{"handle":"upload","offset_bytes":"3","completed":false}"#.utf8))
    }
}

@Suite(.serialized) struct VideoParentRestartTransportTests {
    @Test func rawParentValidationAndAdmissionKeepLiteralU64JSONAndHeaderAuthority() async throws {
        ParentRestartTransport.sent = []
        ParentRestartTransport.bodies = []
        let backend: any MoldBackend = ParentRestartTransport.backend()
        let raw = #"{"generation":{"seed":18446744073709551615,"prompt":"literal\\new"},"duration_ms":"30000"}"#
        let upload = VideoSourceUploadRef(
            handle: "upload", sha256: String(repeating: "b", count: 64), sizeBytes: UInt64.max)
        _ = try await backend.validateVideoRestart(
            requestJSON: raw, destinationInstanceId: "target", sourceUpload: upload)
        _ = try await backend.createVideoJob(
            requestJSON: raw, validationRevision: "proof", operationId: "operation", destinationInstanceId: "target",
            sourceUpload: upload)
        #expect(ParentRestartTransport.bodies == [Data(raw.utf8), Data(raw.utf8)])
        #expect(ParentRestartTransport.sent.map { $0.url!.path } == ["/api/video-jobs/validate", "/api/video-jobs"])
        #expect(
            ParentRestartTransport.sent.allSatisfy {
                $0.value(forHTTPHeaderField: "x-mold-destination-instance") == "target"
            })
        #expect(
            ParentRestartTransport.sent.last?.value(forHTTPHeaderField: "x-mold-video-validation-revision") == "proof")
        #expect(ParentRestartTransport.sent.last?.value(forHTTPHeaderField: "x-mold-operation-id") == "operation")
        let header = try #require(
            ParentRestartTransport.sent.last?.value(forHTTPHeaderField: "x-mold-video-source-upload"))
        #expect(header.contains("18446744073709551615"))
    }
    @Test func prefixRelayForwardsOpaqueSpecAndBoundedChunkBytesToExactOwnerRoutes() async throws {
        ParentRestartTransport.sent = []
        ParentRestartTransport.bodies = []
        let backend: any MoldBackend = ParentRestartTransport.backend()
        let spec = #"{"metadata":{"seed":18446744073709551615,"prompt":"literal\\new"}}"#
        _ = try await backend.beginVideoSourceUpload(id: "upload", specJSON: spec, destinationInstanceId: "target")
        _ = try await backend.appendVideoSourceUpload(
            id: "upload", offsetBytes: 4_294_967_296, bytes: Data([1, 2, 3]), destinationInstanceId: "target")
        _ = try await backend.completeVideoSourceUpload(id: "upload", destinationInstanceId: "target")
        let chunk = try await backend.videoRestartSource(
            id: "source/id", offsetBytes: UInt64.max, sourceInstanceId: "source-instance", sourceRef: "opaque-ref")
        #expect(chunk == Data([1, 2, 3]))
        #expect(ParentRestartTransport.bodies[0] == Data(spec.utf8))
        #expect(ParentRestartTransport.bodies[1] == chunk)
        let read = try #require(ParentRestartTransport.sent.last)
        #expect(
            read.url?.absoluteString.contains("source%2Fid/restart-source?offset_bytes=18446744073709551615") == true)
        #expect(read.value(forHTTPHeaderField: "x-mold-source-instance") == "source-instance")
        #expect(read.value(forHTTPHeaderField: "x-mold-video-source-ref") == "opaque-ref")
    }
}
