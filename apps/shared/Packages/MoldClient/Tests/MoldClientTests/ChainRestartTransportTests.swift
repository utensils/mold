import Foundation
import Testing
@testable import MoldClient

private final class ChainRestartTransport: StubTransport {
    nonisolated(unsafe) static var sent: [URLRequest] = []
    nonisolated(unsafe) static var bodies: [Data] = []
    override class var contentType: String { "application/json" }
    override class func response(for path: String) -> (status: Int, body: Data)? {
        if path.hasSuffix("restart-options") {
            return (200, Data(#"{"source_instance_id":"source","source_revision":"revision","restart":{"supported":true},"send":{"supported":false,"reason":"Local adapter"}}"#.utf8))
        }
        if path.hasSuffix("restart-export") {
            return (200, Data(#"{"source_instance_id":"source","source_revision":"revision","destination_instance_id":"target","request_json":"{\"seed\":18446744073709551615}"}"#.utf8))
        }
        if path == "/api/chain-jobs" { return (202, Data(#"{"job_id":"operation"}"#.utf8)) }
        return (204, Data())
    }
    override func startLoading() {
        Self.sent.append(request)
        var data = request.httpBody ?? Data()
        if let stream = request.httpBodyStream {
            stream.open(); defer { stream.close() }
            var bytes = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let count = stream.read(&bytes, maxLength: bytes.count)
                guard count > 0 else { break }
                data.append(contentsOf: bytes.prefix(count))
            }
        }
        Self.bodies.append(data)
        super.startLoading()
    }
}

@Suite(.serialized) struct ChainRestartTransportTests {
    @Test func exportedWireBodyIsForwardedWithoutSeedOrBackslashReencoding() async throws {
        ChainRestartTransport.sent = []; ChainRestartTransport.bodies = []
        let backend: any MoldBackend = ChainRestartTransport.backend(apiKey: "test-key")
        let raw = #"{"seed":18446744073709551615,"stages":[{"prompt":"C:\\\\new\\\\render"}]}"#
        let accepted = try await backend.createChainJob(requestJSON: raw, operationId: "operation", destinationInstanceId: "target")
        #expect(accepted.jobId == "operation")
        #expect(ChainRestartTransport.bodies.last == Data(raw.utf8))
        let request = try #require(ChainRestartTransport.sent.last)
        #expect(request.value(forHTTPHeaderField: "x-mold-operation-id") == "operation")
        #expect(request.value(forHTTPHeaderField: "x-mold-destination-instance") == "target")
        #expect(request.value(forHTTPHeaderField: "X-Api-Key") == "test-key")
    }

    @Test func restartExportKeepsExactRevisionAndEscapesJobId() async throws {
        ChainRestartTransport.sent = []; ChainRestartTransport.bodies = []
        let backend: any MoldBackend = ChainRestartTransport.backend()
        let options = try await backend.chainRestartOptions(id: "a/b")
        #expect(options.sourceRevision == "revision")
        #expect(options.restart.supported)
        #expect(options.send.reason == "Local adapter")
        let exported = try await backend.exportChainRestart(id: "a/b", request: .init(
            sourceInstanceId: "source", sourceRevision: options.sourceRevision, destinationInstanceId: "target"))
        #expect(exported.requestJSON == #"{"seed":18446744073709551615}"#)
        #expect(ChainRestartTransport.sent.last?.url?.absoluteString.contains("a%2Fb/restart-export") == true)
    }

    @Test func targetRestartCapabilityRequiresPositiveAdvertisement() throws {
        for fixture in [#"{}"#, #"{"queue":{}}"#, #"{"queue":{"chain_restart":false}}"#] {
            #expect(try MoldJSON.decoder.decode(Capabilities.self, from: Data(fixture.utf8)).canRestartChain == false)
        }
        #expect(try MoldJSON.decoder.decode(Capabilities.self, from: Data(#"{"queue":{"chain_restart":true}}"#.utf8)).canRestartChain)
    }
    @Test func stopCarriesCapturedInstallationAndDiscardCarriesExactSourceRevision() async throws {
        ChainRestartTransport.sent = []; ChainRestartTransport.bodies = []
        let backend: any MoldBackend = ChainRestartTransport.backend()
        try await backend.cancelChainOperation(id: "a/b", operationId: "op/one", destinationInstanceId: "target")
        #expect(ChainRestartTransport.sent[0].url?.absoluteString.contains("a%2Fb/operations/op%2Fone/cancel") == true)
        #expect(ChainRestartTransport.sent[0].value(forHTTPHeaderField: "x-mold-destination-instance") == "target")
        #expect(ChainRestartTransport.bodies[0].isEmpty)
        try await backend.discardChainJob(id: "original", request: .init(sourceInstanceId: "source", sourceRevision: "revision"))
        let body = try #require(JSONSerialization.jsonObject(with: ChainRestartTransport.bodies[1]) as? [String: String])
        #expect(body == ["source_instance_id": "source", "source_revision": "revision"])
        #expect(ChainRestartTransport.sent[1].url?.path == "/api/chain-jobs/original/discard")
    }

}
