import Foundation
import MoldClientTesting
import Testing

@testable import MoldClient

@MainActor struct VideoParentRestartControllerTests {
    let helpers = VideoRestartControllerTests()
    func backend(_ name: String, transfer: Bool = true) throws -> FakeBackend {
        let backend = try helpers.backend(name)
        backend.stub(
            "capabilities()",
            returning: try helpers.decode(
                "{\"video_jobs\":{\"supported\":true,\"literal_prompts\":true,\"library_extend\":true,\"source_transfer\":\(transfer),\"restart\":true}}",
                as: Capabilities.self))
        return backend
    }
    @Test func parentLostReplyReopensSameOperationThroughOnlyParentProtocol() async throws {
        let source = try backend("source")
        let destination = try backend("target")
        let defaults = helpers.defaults()
        let raw = #"{"generation":{"seed":18446744073709551615,"prompt":"literal\\new"},"duration_ms":"30000"}"#
        source.stub(
            "videoRestartOptions(id:sourceInstanceId:)",
            returning: try helpers.decode(
                #"{"source_instance_id":"source","source_revision":"revision","restart":{"supported":true},"send":{"supported":true}}"#,
                as: ChainRestartOptions.self))
        let exportData = try JSONSerialization.data(withJSONObject: [
            "source_instance_id": "source", "source_revision": "revision", "destination_instance_id": "target",
            "request_json": raw,
        ])
        source.stub(
            "exportVideoRestart(id:request:)",
            returning: try MoldJSON.decoder.decode(VideoParentRestartExport.self, from: exportData))
        destination.stub(
            "validateVideoRestart(requestJSON:destinationInstanceId:sourceUpload:)",
            returning: try helpers.decode(VideoJobContractTests.plan(), as: VideoJobPlan.self))
        destination.stub(
            "createVideoJob(requestJSON:validationRevision:operationId:destinationInstanceId:sourceUpload:)",
            throwing: URLError(.timedOut))
        let controller = VideoRestartController(defaults: defaults)
        await controller.open(
            source: helpers.target(source), jobId: "original", protocolKind: .videoJob, backend: source,
            current: { true })
        await controller.start(destination: helpers.target(destination), backend: destination, current: { true })
        let record = try #require(controller.admission)
        #expect(record.protocolKind == .videoJob)
        #expect(
            destination.calls.first(where: {
                $0.route
                    == "createVideoJob(requestJSON:validationRevision:operationId:destinationInstanceId:sourceUpload:)"
            })?.arguments.first as? String == raw)
        let restored = VideoRestartController(defaults: defaults)
        await restored.open(
            source: helpers.target(source), jobId: "original", protocolKind: .videoJob, backend: source,
            current: { true })
        #expect(restored.admission?.operationId == record.operationId)
        destination.stub(
            "videoJob(id:)",
            returning: try helpers.decode(
                """
                {"id":"\(record.operationId)","state":"queued","plan":\(VideoJobContractTests.plan()),"completed_clips":"0","created_at_ms":1,"updated_at_ms":2,"can_cancel":true,"can_resume":false,"can_delete":false}
                """, as: VideoJob.self))
        await restored.refresh(destination: helpers.target(destination), backend: destination, current: { true })
        #expect(restored.parentDetail?.id == record.operationId)
        #expect(
            destination.count(
                "createVideoJob(requestJSON:validationRevision:operationId:destinationInstanceId:sourceUpload:)") == 1)
        #expect(source.count("exportChainRestart(id:request:)") == 0)
        #expect(destination.count("chainJob(id:)") == 0)
    }
    @Test func orphanedReceiptReopensWithoutSourceAndCannotStartAgain() async throws {
        let source = try backend("source")
        let target = try backend("target")
        let defaults = helpers.defaults()
        let record = VideoRestartAdmission(
            operationId: "same-operation", destinationHostId: target.host.id,
            destinationOrigin: target.host.baseURL.absoluteString, destinationInstanceId: "target",
            destinationMode: target.host.connectionMode, sourceHostId: source.host.id,
            sourceJobId: "removed-source", sourceInstanceId: "source", sourceRevision: "revision",
            protocolKind: .videoJob)
        defaults.set(try MoldJSON.encoder.encode([record]), forKey: VideoRestartController.journalKey)
        let controller = VideoRestartController(defaults: defaults)
        #expect(controller.restorePending(record))
        #expect(controller.protocolKind == .videoJob)
        #expect(controller.admission?.operationId == "same-operation")
        #expect(controller.source == nil)
        await controller.start(destination: helpers.target(target), backend: target, current: { true })
        #expect(
            target.count(
                "createVideoJob(requestJSON:validationRevision:operationId:destinationInstanceId:sourceUpload:)") == 0)
        var invented = record
        invented.cancelRequested = true
        #expect(!controller.restorePending(invented))
    }

    @Test func localPrefixUsesCompletedGrantAndNeverRelaysBytes() async throws {
        let source = try backend("same")
        let controller = VideoRestartController(defaults: helpers.defaults())
        let sha = String(repeating: "b", count: 64)
        let wire: [String: Any] = [
            "source_instance_id": "same", "source_revision": "revision", "destination_instance_id": "same",
            "request_json": "{}",
            "prefix": [
                "filename": "original.mp4", "archive_identity_sha256": String(repeating: "a", count: 64), "sha256": sha,
                "size_bytes": "3", "source_ref": "source", "upload_spec_json": "{}",
                "retained_upload": ["handle": "completed", "sha256": sha, "size_bytes": "3"],
            ],
        ]
        let exported = try MoldJSON.decoder.decode(
            VideoParentRestartExport.self, from: JSONSerialization.data(withJSONObject: wire))
        let grant = try await controller.transferPrefix(
            exported, source: helpers.target(source), sourceBackend: source,
            sourceJobId: "original", destination: helpers.target(source), backend: source, current: { true })
        #expect(grant?.handle == "completed")
        #expect(source.count("beginVideoSourceUpload(id:specJSON:destinationInstanceId:)") == 0)
        #expect(source.count("videoRestartSource(id:offsetBytes:sourceInstanceId:sourceRef:)") == 0)
        source.stub(
            "capabilities()",
            returning: try helpers.decode(
                #"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true,"restart":true,"source_transfer":false}}"#,
                as: Capabilities.self))
        let local = try await controller.transferPrefix(
            exported, source: helpers.target(source), sourceBackend: source,
            sourceJobId: "original", destination: helpers.target(source), backend: source, current: { true })
        #expect(local?.handle == "completed")
        var remoteWire = wire
        var remotePrefix = try #require(remoteWire["prefix"] as? [String: Any])
        remotePrefix["retained_upload"] = nil
        remoteWire["prefix"] = remotePrefix
        remoteWire["destination_instance_id"] = "target"
        let remote = try MoldJSON.decoder.decode(
            VideoParentRestartExport.self, from: JSONSerialization.data(withJSONObject: remoteWire))
        let destination = try backend("target")
        await #expect(throws: (any Error).self) {
            try await controller.transferPrefix(
                remote, source: self.helpers.target(source), sourceBackend: source,
                sourceJobId: "original", destination: self.helpers.target(destination), backend: destination,
                current: { true })
        }
        #expect(destination.count("beginVideoSourceUpload(id:specJSON:destinationInstanceId:)") == 0)
    }

    @Test func parentLostStopReplyReplaysOnlySameCancellationBeforeExactGet() async throws {
        let target = try backend("target")
        let defaults = helpers.defaults()
        let record = VideoRestartAdmission(
            operationId: "same-operation", destinationHostId: target.host.id,
            destinationOrigin: target.host.baseURL.absoluteString, destinationInstanceId: "target",
            destinationMode: target.host.connectionMode,
            sourceHostId: UUID(), sourceJobId: "original", sourceInstanceId: "source", sourceRevision: "revision",
            protocolKind: .videoJob)
        defaults.set(try MoldJSON.encoder.encode([record]), forKey: VideoRestartController.journalKey)
        let first = VideoRestartController(defaults: defaults)
        #expect(first.restorePending(record))
        target.stub("cancelVideoJob(id:destinationInstanceId:)", throwing: URLError(.networkConnectionLost))
        await first.cancel(destination: helpers.target(target), backend: target, current: { true })
        #expect(first.pending.first?.cancelRequested == true)
        #expect(first.pending.first?.cancelAcknowledged == false)
        let restored = VideoRestartController(defaults: defaults)
        #expect(restored.restorePending(try #require(restored.pending.first)))
        target.stub("cancelVideoJob(id:destinationInstanceId:)", returning: ())
        target.stub(
            "videoJob(id:)",
            throwing: MoldClientError.http(status: 410, code: "VIDEO_OPERATION_CANCELLED", message: "cancelled"))
        await restored.refresh(destination: helpers.target(target), backend: target, current: { true })
        #expect(restored.cancelledBeforeAdmission)
        #expect(restored.pending.isEmpty)
        #expect(target.count("cancelVideoJob(id:destinationInstanceId:)") == 2)
        #expect(
            target.calls.filter { $0.route == "cancelVideoJob(id:destinationInstanceId:)" }.allSatisfy {
                $0.arguments[0] as? String == record.operationId
            })
        #expect(
            target.count(
                "createVideoJob(requestJSON:validationRevision:operationId:destinationInstanceId:sourceUpload:)") == 0)
        #expect(target.count("cancelChainOperation(id:operationId:)") == 0)
    }

    @Test func durationCapabilityAloneCannotAuthorizeRestart() async throws {
        for flag in ["", ",\"restart\":false"] {
            let source = try backend("source")
            let controller = VideoRestartController(defaults: helpers.defaults())
            source.stub(
                "capabilities()",
                returning: try helpers.decode(
                    "{\"video_jobs\":{\"supported\":true,\"literal_prompts\":true,\"library_extend\":true\(flag)}}",
                    as: Capabilities.self))
            source.stub(
                "videoRestartOptions(id:sourceInstanceId:)",
                returning: try helpers.decode(
                    #"{"source_instance_id":"source","source_revision":"revision","restart":{"supported":true},"send":{"supported":true}}"#,
                    as: ChainRestartOptions.self))
            await controller.open(
                source: helpers.target(source), jobId: "original", protocolKind: .videoJob, backend: source,
                current: { true })
            #expect(controller.options == nil)
            #expect(source.count("videoRestartOptions(id:sourceInstanceId:)") == 0)
        }
    }

    @Test func exactAcceptedParentPrunesOnlyReceiptWithoutPendingStop() async throws {
        for stopDuringPost in [false,true] {
            let source=try backend("source"), destination=try backend("target")
            let controller=VideoRestartController(defaults:helpers.defaults())
            source.stub("videoRestartOptions(id:sourceInstanceId:)",returning:try helpers.decode(#"{"source_instance_id":"source","source_revision":"revision","restart":{"supported":true},"send":{"supported":true}}"#,as:ChainRestartOptions.self))
            source.stub("exportVideoRestart(id:request:)",returning:try helpers.decode(#"{"source_instance_id":"source","source_revision":"revision","destination_instance_id":"target","request_json":"{}"}"#,as:VideoParentRestartExport.self))
            destination.stub("validateVideoRestart(requestJSON:destinationInstanceId:sourceUpload:)",returning:try helpers.decode(VideoJobContractTests.plan(),as:VideoJobPlan.self))
            destination.stub("cancelVideoJob(id:destinationInstanceId:)",throwing:URLError(.timedOut))
            let target=helpers.target(destination)
            destination.stub("createVideoJob(requestJSON:validationRevision:operationId:destinationInstanceId:sourceUpload:)") { args in
                if stopDuringPost {await controller.cancel(destination:target,backend:destination,current:{true})}
                let id=args[2] as! String
                return try MoldJSON.decoder.decode(VideoJob.self,from:Data("{\"id\":\"\(id)\",\"state\":\"queued\",\"plan\":\(VideoJobContractTests.plan()),\"completed_clips\":\"0\",\"created_at_ms\":1,\"updated_at_ms\":2,\"can_cancel\":true,\"can_resume\":false,\"can_delete\":false}".utf8))
            }
            await controller.open(source:helpers.target(source),jobId:"original",protocolKind:.videoJob,backend:source,current:{true})
            await controller.start(destination:target,backend:destination,current:{true})
            #expect(controller.parentDetail?.id==controller.admission?.operationId)
            #expect(controller.progress == nil, "Preparation must stop animating after the attempt is acknowledged")
            #expect(controller.pending.count==(stopDuringPost ? 1:0))
            if stopDuringPost {#expect(controller.pending.first?.cancelRequested==true && controller.pending.first?.cancelAcknowledged==false)}
        }
    }

}
