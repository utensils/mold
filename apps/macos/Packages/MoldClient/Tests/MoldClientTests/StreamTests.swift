import Foundation
import Testing

@testable import MoldClient

// Pins the SSE routes through one stub transport: a stream should check its
// status before the first byte and split frames on the accumulator that keeps
// empty lines, not `URLSession`'s `.lines`, which drops the blank line that
// ends an SSE frame.

/// Serves canned responses by path, so a test never opens a real socket.
final class StubURLProtocol: URLProtocol {
    nonisolated(unsafe) static var responses: [String: (status: Int, body: Data)] = [:]

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let path = request.url?.path ?? ""
        guard let fixture = Self.responses[path] else {
            client?.urlProtocol(self, didFailWithError: MoldClientError.malformedResponse)
            return
        }
        let response = HTTPURLResponse(
            url: request.url!, statusCode: fixture.status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: fixture.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private func stubbedBackend() -> HTTPBackend {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [StubURLProtocol.self]
    return HTTPBackend(
        host: MoldHost(name: "stub", baseURL: URL(string: "http://stub:7680")!),
        session: URLSession(configuration: config))
}

// Serialized: every test stubs the same static `responses` table, and this
// suite is the one place that mutates it.
@Suite(.serialized)
struct StreamTests {

    @Test func aDownloadFrameArrivesThroughTheStream() async throws {
        let frame = #"{"type":"job_progress","id":"d1","model":"flux-dev","bytes_done":10,"bytes_total":20}"#
        let body = "event: download\r\ndata: \(frame)\r\n\r\n"
        StubURLProtocol.responses["/api/downloads/stream"] = (200, Data(body.utf8))
        let backend = stubbedBackend()

        var received: [DownloadEvent] = []
        for try await event in backend.downloadEvents() {
            received.append(event)
            break
        }
        #expect(received.count == 1)
        #expect(received.first?.id == "d1")
        #expect(received.first?.model == "flux-dev")
    }

    @Test func aRefusedStreamThrowsUnauthorized() async throws {
        StubURLProtocol.responses["/api/events"] = (401, Data())
        let backend = stubbedBackend()

        await #expect {
            for try await _ in backend.events() {}
        } throws: { error in
            guard case MoldClientError.unauthorized = error else { return false }
            return true
        }
    }

    @Test func theResourceStreamTakesSnapshotFramesAndDropsPings() async throws {
        let snapshot = #"{"hostname":"h","timestamp":1,"gpus":[],"system_ram":{"total":1,"used":1,"used_by_mold":0}}"#
        let body = ": ping\r\n\r\n" + "event: snapshot\r\ndata: \(snapshot)\r\n\r\n"
        StubURLProtocol.responses["/api/resources/stream"] = (200, Data(body.utf8))
        let backend = stubbedBackend()

        var received: [ResourceSnapshot] = []
        for try await sample in backend.resourceStream() { received.append(sample) }
        #expect(received.count == 1)
        #expect(received.first?.hostname == "h")
    }

    @Test func aBatchStreamStopsAtTheFirstSettledFrame() async throws {
        let complete = #"{"id":"b1","client_batch_id":"c1","instance_id":"i","durable":true,"children":[{"index":0,"job_id":"j1","state":"complete","revision":1}]}"#
        let running = #"{"id":"b1","client_batch_id":"c1","instance_id":"i","durable":true,"children":[{"index":0,"job_id":"j1","state":"running","revision":2}]}"#
        let body = "event: generation_batch\r\ndata: \(complete)\r\n\r\n" +
            "event: generation_batch\r\ndata: \(running)\r\n\r\n"
        StubURLProtocol.responses["/api/generation-batches/b1/events"] = (200, Data(body.utf8))
        let backend = stubbedBackend()

        var received: [BatchStatus] = []
        for try await status in backend.batchEvents(id: "b1") {
            received.append(status)
        }
        #expect(received.count == 1)
        #expect(received.first?.isSettled == true)
    }
}
