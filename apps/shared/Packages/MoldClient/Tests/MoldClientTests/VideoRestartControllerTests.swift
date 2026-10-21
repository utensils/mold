import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

@MainActor struct VideoRestartControllerTests {
    func decode<T: Decodable>(_ text: String, as: T.Type = T.self) throws -> T {
        try MoldJSON.decoder.decode(T.self, from: Data(text.utf8))
    }
    func backend(_ name: String) throws -> FakeBackend {
        let backend = FakeBackend(host: MoldHost(name: name, baseURL: URL(string: "http://\(name):7680")!))
        backend.stub("status()", returning: try decode(#"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"INSTANCE"}"#.replacingOccurrences(of: "INSTANCE", with: name), as: ServerStatus.self))
        backend.stub("capabilities()", returning: try decode(#"{"queue":{"chain_restart":true}}"#, as: Capabilities.self))
        return backend
    }
    func source() throws -> FakeBackend {
        let backend = try backend("source")
        backend.stub("chainRestartOptions(id:)", returning: try decode(#"{"source_instance_id":"source","source_revision":"revision","restart":{"supported":true},"send":{"supported":true}}"#, as: ChainRestartOptions.self))
        backend.stub("exportChainRestart(id:request:)", returning: try decode(#"{"source_instance_id":"source","source_revision":"revision","destination_instance_id":"target","request_json":"{\"seed\":18446744073709551615}"}"#, as: ChainRestartExport.self))
        return backend
    }
    func target(_ backend: FakeBackend) -> VideoRestartTarget { .init(host: backend.host, instanceId: backend.host.name) }
    func defaults() -> UserDefaults { UserDefaults(suiteName: "restart-test-\(UUID())")! }

    @Test func absentTargetCapabilityDoesNotExportOrSubmit() async throws {
        let source = try source(), destination = try backend("target")
        destination.stub("capabilities()", returning: try decode("{}", as: Capabilities.self))
        let controller = VideoRestartController(defaults: defaults())
        await controller.open(source: target(source), jobId: "original", backend: source, current: { true })
        await controller.start(destination: target(destination), backend: destination, current: { true })
        #expect(source.count("exportChainRestart(id:request:)") == 0)
        #expect(destination.count("createChainJob(requestJSON:operationId:destinationInstanceId:)") == 0)
        #expect(controller.error != nil)
    }

    @Test func ambiguousReplyPersistsExactTargetAndNeverRepostsOnRecovery() async throws {
        let source = try source(), destination = try backend("target"), saved = defaults()
        destination.stub("createChainJob(requestJSON:operationId:destinationInstanceId:)", throwing: MoldClientError.unreachable("Lost reply"))
        let controller = VideoRestartController(defaults: saved)
        await controller.open(source: target(source), jobId: "original", backend: source, current: { true })
        await controller.start(destination: target(destination), backend: destination, current: { true })
        let record = try #require(controller.admission)
        #expect(record.destinationHostId == destination.host.id)
        #expect(record.sourceJobId == "original")
        #expect(controller.pending.count == 1)
        let sent = try #require(destination.calls.last { $0.route == "createChainJob(requestJSON:operationId:destinationInstanceId:)" })
        #expect(sent.arguments.first as? String == #"{"seed":18446744073709551615}"#)
        let recovered = VideoRestartController(defaults: saved)
        await recovered.open(source: target(source), jobId: "original", backend: source, current: { true })
        #expect(recovered.admission?.operationId == record.operationId)
        #expect(source.count("exportChainRestart(id:request:)") == 1)
        #expect(destination.count("createChainJob(requestJSON:operationId:destinationInstanceId:)") == 1)
    }

    @Test func definitiveRefusalReleasesJournalButUnknownStatusDoesNot() async throws {
        for status in [422, 425, 499] {
            let source = try source(), destination = try backend("target")
            destination.stub("createChainJob(requestJSON:operationId:destinationInstanceId:)", throwing: MoldClientError.http(status: status, code: nil, message: "Rejected"))
            let controller = VideoRestartController(defaults: defaults())
            await controller.open(source: target(source), jobId: "original", backend: source, current: { true })
            await controller.start(destination: target(destination), backend: destination, current: { true })
            #expect(controller.pending.count == (status == 422 ? 0 : 1))
        }
    }

    @Test func corruptJournalBlocksNewAdmissionInsteadOfOverwritingIt() async throws {
        let saved = defaults(); saved.set(Data("broken".utf8), forKey: VideoRestartController.journalKey)
        let source = try source(), destination = try backend("target")
        let controller = VideoRestartController(defaults: saved)
        await controller.open(source: target(source), jobId: "original", backend: source, current: { true })
        await controller.start(destination: target(destination), backend: destination, current: { true })
        #expect(destination.calls.isEmpty)
        #expect(saved.data(forKey: VideoRestartController.journalKey) == Data("broken".utf8))
    }

    @Test func ownerChangeDuringExportCannotCreateDestinationWork() async throws {
        let source = try source(), destination = try backend("target")
        let exported: ChainRestartExport = try decode(#"{"source_instance_id":"source","source_revision":"revision","destination_instance_id":"target","request_json":"{}"}"#)
        let gate = AsyncStream<Void>.makeStream()
        source.stub("exportChainRestart(id:request:)") { _ in
            for await _ in gate.stream { break }
            return exported
        }
        let controller = VideoRestartController(defaults: defaults())
        await controller.open(source: target(source), jobId: "original", backend: source, current: { true })
        var current = true
        let run = Task { await controller.start(destination: target(destination), backend: destination, current: { current }) }
        for _ in 0..<100 where source.count("exportChainRestart(id:request:)") == 0 { await Task.yield() }
        current = false
        gate.continuation.finish()
        await run.value
        #expect(destination.count("createChainJob(requestJSON:operationId:destinationInstanceId:)") == 0)
        #expect(controller.pending.isEmpty)
        #expect(!controller.isBusy)
    }

    @Test func destinationReplacementAndExportEchoMismatchRefuseBeforePost() async throws {
        let source = try source(), destination = try backend("target")
        destination.stub("status()", returning: try decode(#"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"replacement"}"#, as: ServerStatus.self))
        let controller = VideoRestartController(defaults: defaults())
        await controller.open(source: target(source), jobId: "original", backend: source, current: { true })
        await controller.start(destination: target(destination), backend: destination, current: { true })
        #expect(source.count("exportChainRestart(id:request:)") == 0)
        #expect(controller.pending.isEmpty)
        let request = ChainRestartExportRequest(sourceInstanceId: "source", sourceRevision: "revision", destinationInstanceId: "target")
        let wrong: ChainRestartExport = try decode(#"{"source_instance_id":"source","source_revision":"changed","destination_instance_id":"target","request_json":"{}"}"#)
        #expect(throws: SequenceDraftError.self) { try wrong.validated(for: request) }
    }

    @Test func operationIsDurableBeforeDestinationPostAndDoesNotPersistMedia() async throws {
        let source = try source(), destination = try backend("target"), saved = defaults()
        let readSaved: @MainActor @Sendable () -> Data? = { saved.data(forKey: VideoRestartController.journalKey) }
        destination.stub("createChainJob(requestJSON:operationId:destinationInstanceId:)") { args in
            let persisted = await readSaved()
            let bytes = try #require(persisted)
            let records = try MoldJSON.decoder.decode([VideoRestartAdmission].self, from: bytes)
            #expect(records.last?.operationId == args[1] as? String)
            #expect(String(data: bytes, encoding: .utf8)?.contains("request_json") == false)
            #expect(String(data: bytes, encoding: .utf8)?.contains("18446744073709551615") == false)
            return try MoldJSON.decoder.decode(CreateChainJobResponse.self,
                from: Data(("{\"job_id\":\"" + (args[1] as! String) + "\"}").utf8))
        }
        let controller = VideoRestartController(defaults: saved)
        await controller.open(source: target(source), jobId: "original", backend: source, current: { true })
        await controller.start(destination: target(destination), backend: destination, current: { true })
        #expect(controller.error == nil)
        #expect(controller.pending.count == 1)
    }

    @Test func mismatchedStatusCannotPublishOrForgetTheKnownOperation() async throws {
        let source = try source(), destination = try backend("target")
        destination.stub("createChainJob(requestJSON:operationId:destinationInstanceId:)", throwing: MoldClientError.unreachable("Lost reply"))
        destination.stub("chainJob(id:)", returning: try decode(#"{"id":"different-operation","state":"completed","model":"model","stage_count":1,"current_stage":1}"#, as: ChainJobDetail.self))
        let controller = VideoRestartController(defaults: defaults())
        await controller.open(source: target(source), jobId: "original", backend: source, current: { true })
        await controller.start(destination: target(destination), backend: destination, current: { true })
        await controller.refresh(destination: target(destination), backend: destination, current: { true })
        #expect(controller.detail == nil)
        #expect(controller.pending.count == 1)
        #expect(controller.error != nil)
    }

}
