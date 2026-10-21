import Foundation
import Testing
@testable import MoldClient

private final class VideoSnapshotTransport: StubTransport {
    nonisolated(unsafe) static var responseId = "parent"
    nonisolated(unsafe) static var instance: String?
    override func startLoading() {
        Self.instance = request.value(forHTTPHeaderField: "x-mold-destination-instance")
        super.startLoading()
    }
    override class func response(for path: String) -> (status: Int, body: Data)? {
        let json = """
        {"id":"\(responseId)","state":"completed","plan":\(VideoJobContractTests.plan()),"completed_clips":"4",
        "active_child_id":null,"output_filename":"saved.mp4","error":null,"error_detail":null,
        "created_at_ms":1,"updated_at_ms":2,"can_cancel":false,"can_resume":false,"can_delete":true}
        """.replacingOccurrences(of: "\n", with: "")
        return (200, Data(("event: snapshot\ndata: " + json + "\n\n").utf8))
    }
}

@Suite(.serialized) struct VideoJobStreamTests {
    @Test func terminalSnapshotForExactParentIsDelivered() async throws {
        VideoSnapshotTransport.responseId = "parent"
        let backend: any MoldBackend = VideoSnapshotTransport.backend().pinningVideoDestination("installation")
        var values: [VideoJob] = []
        for try await value in backend.videoJobEvents(id: "parent") { values.append(value) }
        #expect(VideoSnapshotTransport.instance == "installation")
        #expect(values.count == 1)
        #expect(values[0].outputFilename == "saved.mp4")
    }
    @Test func wrongParentSnapshotCannotPublishOrSettleTheReceipt() async throws {
        VideoSnapshotTransport.responseId = "different-parent"
        let backend: any MoldBackend = VideoSnapshotTransport.backend().pinningVideoDestination("installation")
        var delivered = 0, refused = false
        do {
            for try await _ in backend.videoJobEvents(id: "parent") { delivered += 1 }
        } catch { refused = true }
        #expect(delivered == 0)
        #expect(refused)
    }
}
