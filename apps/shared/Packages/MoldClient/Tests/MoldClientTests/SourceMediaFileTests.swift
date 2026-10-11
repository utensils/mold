import Foundation
import Testing
@testable import MoldClient

struct SourceMediaFileTests {
    @Test func previewCeilingRejectsOversizedOriginalButAllowsBoundary() throws {
        try SourceMediaFile.checkPreviewSize(SourceMediaFile.previewLimit)
        #expect(throws: (any Error).self) { try SourceMediaFile.checkPreviewSize(SourceMediaFile.previewLimit + 1) }
    }
    @Test func originalFileIsPrivateAndRemovedWithViewerLifetime() throws {
        var file: SourceMediaFile? = try SourceMediaFile(extension: "wav")
        let url = try #require(file?.url)
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
        let directory = try FileManager.default.attributesOfItem(atPath: url.deletingLastPathComponent().path)
        #expect((directory[.posixPermissions] as? NSNumber)?.intValue == 0o700)
        file = nil
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }
}

private final class SourceOriginalProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let oversized = request.url!.path.contains("oversized")
        let body = Data([1, 2, 3])
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Length": oversized ? String(SourceMediaFile.previewLimit + 1) : "3"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if !oversized { client?.urlProtocol(self, didLoad: body) }
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

extension SourceMediaFileTests {
    @Test func originalHTTPStreamsToPrivateFileAndRejectsDeclaredOversize() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [SourceOriginalProtocol.self]
        let backend = HTTPBackend.testing(host: MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture")!),
                                  session: URLSession(configuration: configuration))
        let member = RetainedSourceMedia.Member(memberId: "m", role: "audio_file", displayName: "audio.wav", sizeBytes: 3)
        let file = try await backend.retainedSourceMediaFile(for: "owner.png", member: member)
        #expect(try Data(contentsOf: file.url) == Data([1, 2, 3]))
        await #expect(throws: (any Error).self) { try await backend.retainedSourceMediaFile(for: "oversized.png", member: member) }
    }
}

extension SourceMediaFileTests {
    @Test func cancelledDecodeDoesNotPublishPixels() async throws {
        let file = try SourceMediaFile(extension: "png")
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await SourceMediaOriginalImage.decode(file)
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }
}

extension SourceMediaFileTests {
    @Test func queueOriginalUsesBoundedPrivateStreamingAndRequiresIndex() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [SourceOriginalProtocol.self]
        let backend = HTTPBackend(host: MoldHost(name: "Fixture", baseURL: URL(string: "http://fixture/prefix")!),
                                  session: URLSession(configuration: configuration))
        let file = try await backend.queueInputOriginal(id: "job #1", index: 2)
        #expect(try Data(contentsOf: file.url) == Data([1, 2, 3]))
        await #expect(throws: (any Error).self) { try await backend.queueInputOriginal(id: "oversized", index: 2) }
        await #expect(throws: MoldClientError.self) { try await backend.queueInputOriginal(id: "job", index: -1) }
    }
}
