import Foundation
import Testing
@testable import MoldClient
import MoldClientTesting

private actor CancellationBytes: MoldBodyReader {
    var data: Data?
    init(_ data: Data) { self.data = data.isEmpty ? nil : data }
    func read() -> Data? { defer { data = nil }; return data }
    func cancel() { data = nil }
}
private struct CancellationTransport: MoldHTTPTransport {
    let status: Int
    let body: String
    func response(for request: URLRequest, body: MoldRequestBody) async throws -> (MoldResponseBytes, HTTPURLResponse) {
        #expect(request.httpMethod == "DELETE")
        #expect(request.url?.path == "/api/queue/job")
        return (MoldResponseBytes(reader: CancellationBytes(Data(self.body.utf8))), HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!)
    }
}
@Suite struct QueueCancellationReceiptTests {
    private func backend(_ status: Int, _ body: String = "") -> any MoldBackend {
        HTTPBackend(host: MoldHost(name: "host", baseURL: URL(string: "http://host")!), transport: CancellationTransport(status: status, body: body))
    }
    @Test func acceptedCancellationIsPendingWhileTheDeviceIsHeld() async throws {
        let receipt = try await backend(202, #"{"cancel_requested":true,"device_still_held":true,"stall":null}"#).requestJobCancellation(id: "job")
        #expect(receipt == .pending(stall: nil))
    }
    @Test func queuedTerminalCancellationRemainsCompatible() async throws {
        #expect(try await backend(204).requestJobCancellation(id: "job") == .cancelled)
        try await backend(204).cancelJob(id: "job")
    }
    @Test func malformedPendingResponseCannotClaimTerminalCancellation() async {
        await #expect(throws: (any Error).self) {
            try await backend(202, #"{"cancel_requested":true,"device_still_held":false}"#).requestJobCancellation(id: "job")
        }
    }
    @Test func sourceCompatibleBackendFallbackDoesNotInventTerminalProof() async throws {
        let fake = FakeBackend()
        fake.stub("cancelJob(id:)", returning: ())
        #expect(try await fake.requestJobCancellation(id: "job") == .unconfirmed)
    }
    @Test func stageCountersRepresentCompletedWorkWithinCurrentPhase() throws {
        let progress = try MoldJSON.decoder.decode(JobProgress.self, from: Data(#"{"stage":"Decode QKV chunks","stage_current":2,"stage_total":9}"#.utf8))
        #expect(progress.stageCurrent == 2)
        #expect(progress.stageTotal == 9)
    }
}
