import Foundation
import Testing
@testable import MoldClient

@Test func queuedSourceThumbnailRejectsDeclaredOversizeBeforeCollection() throws {
    let response = try #require(HTTPURLResponse(url: URL(string: "http://box/api/queue/j/input-thumbnail")!,
        statusCode: 200, httpVersion: nil, headerFields: ["Content-Length": "2097153"]))
    #expect(throws: ResponseCeiling.Exceeded.self) { try HTTPBackend.validateQueueThumbnail(response) }
}

@Test func queuedSourceThumbnailPreservesAuthenticationAndMissingJobs() throws {
    for status in [401, 404] {
        let response = try #require(HTTPURLResponse(url: URL(string: "http://box")!, statusCode: status,
            httpVersion: nil, headerFields: nil))
        #expect(throws: MoldClientError.self) { try HTTPBackend.validateQueueThumbnail(response) }
    }
}

@Test func queuedSourceThumbnailUsesOneEscapedJobComponentAndHostPrefix() {
    let backend = HTTPBackend(host: MoldHost(name: "box", baseURL: URL(string: "http://box/mold")!, apiKey: "secret"))
    let request = backend.request(backend.queueInputThumbnailPath("job #1"))
    #expect(request.url?.absoluteString == "http://box/mold/api/queue/job%20%231/input-thumbnail")
    #expect(request.value(forHTTPHeaderField: "X-Api-Key") == "secret")
}
