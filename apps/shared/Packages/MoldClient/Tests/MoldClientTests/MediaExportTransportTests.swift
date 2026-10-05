import Foundation
import Testing
@testable import MoldClient

private final class ExportTransport: StubTransport {
    nonisolated(unsafe) static var seen: URLRequest?
    nonisolated(unsafe) static var body: Data?
    override class func response(for path: String) -> (status: Int, body: Data)? { (200, Data([1, 2, 3])) }
    override func startLoading() {
        Self.seen = request
        Self.body = request.httpBody
        if let stream = request.httpBodyStream {
            stream.open(); defer { stream.close() }
            var bytes = Data(); var buffer = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count <= 0 { break }
                bytes.append(buffer, count: count)
            }
            Self.body = bytes
        }
        super.startLoading()
    }
}

@Suite(.serialized) struct MediaExportTransportTests {
    @Test func videoExportUsesTheHoldingHostWithAnAuthenticatedZeroPauseBody() async throws {
        let backend = ExportTransport.backend(apiKey: "fixture-key", host: "rendering-host")
        _ = try await backend.export("loop #1.mp4", request: VideoExportRequest(pauseMs: 0))
        let request = try #require(ExportTransport.seen)
        #expect(request.url?.absoluteString == "http://rendering-host:7680/api/gallery/export/loop%20%231.mp4")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "X-Api-Key") == "fixture-key")
        #expect(request.timeoutInterval == 300)
        let body = try #require(ExportTransport.body)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(json["pause_ms"] as? Int == 0)
        #expect(json["repeat"] as? String == "forever")
        #expect(json["transparent"] == nil && json["frames"] == nil)
    }
    @Test func assetComponentsStayEncodedAndUseTheLargeMediaTimeout() async throws {
        let backend = ExportTransport.backend(apiKey: "fixture-key")
        _ = try await backend.generationAsset("mesh #1.glb", assetID: "base/color?1")
        let request = try #require(ExportTransport.seen)
        #expect(request.url?.absoluteString == "http://stub:7680/api/gallery/assets/mesh%20%231.glb/base%2Fcolor%3F1")
        #expect(request.value(forHTTPHeaderField: "X-Api-Key") == "fixture-key")
        #expect(request.httpMethod == "GET")
        #expect(request.timeoutInterval == 300)
        #expect(request.url?.query == nil)
    }
}
