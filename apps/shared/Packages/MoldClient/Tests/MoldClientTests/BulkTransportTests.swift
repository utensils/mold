import Foundation
import Testing
@testable import MoldClient

private final class BulkTransport: StubTransport {
    nonisolated(unsafe) static var timeouts: [TimeInterval] = []
    nonisolated(unsafe) static var modelLoadRequest: URLRequest?
    nonisolated(unsafe) static var modelLoadBody: Data?
    override class func response(for path: String) -> (status: Int, body: Data)? {
        (204, Data())
    }
    override func startLoading() {
        Self.timeouts.append(request.timeoutInterval)
        if request.url?.path == "/api/models/load" {
            Self.modelLoadRequest = request
            Self.modelLoadBody = request.httpBody ?? request.httpBodyStream.map(Self.read)
        }
        super.startLoading()
    }

    private static func read(_ stream: InputStream) -> Data {
        stream.open(); defer { stream.close() }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while stream.hasBytesAvailable {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count <= 0 { break }
            data.append(buffer, count: count)
        }
        return data
    }
}

@Suite(.serialized)
struct BulkTransportTests {
    @Test func modelLoadAllowsColdStartWhileOtherRequestsKeepShortTimeout() async throws {
        BulkTransport.modelLoadRequest = nil
        BulkTransport.modelLoadBody = nil
        let backend = BulkTransport.backend(apiKey: "test-key")
        try await backend.loadModel("z-image-turbo:bf16", gpu: 1)

        let request = try #require(BulkTransport.modelLoadRequest)
        #expect(request.url?.path == "/api/models/load")
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        #expect(request.value(forHTTPHeaderField: "X-Api-Key") == "test-key")
        let body = try #require(BulkTransport.modelLoadBody)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(json["model"] as? String == "z-image-turbo:bf16")
        #expect(json["gpu"] as? Int == 1)
        #expect(request.timeoutInterval == 300)
        #expect(backend.request("/api/status").timeoutInterval == 10)
    }

    @Test func destructiveRequestsAllowFilesystemWorkWithoutChangingProbeTimeouts() async throws {
        BulkTransport.timeouts = []
        let backend = BulkTransport.backend()
        try await backend.trash(["a.png"])
        try await backend.restoreFromTrash(["a.png"])
        try await backend.deleteForever(["a.png"])
        try await backend.emptyTrash()
        #expect(BulkTransport.timeouts == [300, 300, 300, 300])
        #expect(backend.request("/api/status").timeoutInterval == 10)
    }

    @Test func busyLifecycleActionsAreDisabledInBothMenuScopes() {
        for scope in [LibraryScopeKind.prints, .trash] {
            let plan = LibraryMenuPlan(scope: scope, count: 3, trashCount: 10, lifecycleBusy: true)
            let removals = plan.items.filter {
                switch $0.kind {
                case .trash?, .putBack?, .deleteForever?, .emptyTrash?: true
                default: false
                }
            }
            #expect(!removals.isEmpty)
            #expect(removals.allSatisfy { $0.isDisabled })
        }
    }
}
