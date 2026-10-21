import Foundation
import MoldClientTesting
import Synchronization
import Testing

@testable import MoldClient

@MainActor struct VideoParentPrefixRelayTests {
    @Test func relayUsesExactBoundedOffsetsAndStopsBeforeWriteWhenOwnerChanges() async throws {
        let h = VideoParentRestartControllerTests()
        for ownerChanges in [false, true] {
            let source = try h.backend("source")
            let target = try h.backend("target")
            let current = Mutex(true)
            let size = UInt64(4 * 1024 * 1024 + 3)
            let sha = String(repeating: "b", count: 64)
            let wire: [String: Any] = [
                "source_instance_id": "source", "source_revision": "revision", "destination_instance_id": "target",
                "request_json": "{}",
                "prefix": [
                    "filename": "original.mp4", "archive_identity_sha256": String(repeating: "a", count: 64),
                    "sha256": sha, "size_bytes": String(size), "source_ref": "opaque", "upload_spec_json": "{}",
                ],
            ]
            let exported = try MoldJSON.decoder.decode(
                VideoParentRestartExport.self, from: JSONSerialization.data(withJSONObject: wire))
            target.stub("beginVideoSourceUpload(id:specJSON:destinationInstanceId:)") { args in
                let id = args[0] as! String
                return try MoldJSON.decoder.decode(
                    VideoSourceUploadStatus.self,
                    from: Data("{\"handle\":\"\(id)\",\"offset_bytes\":\"0\",\"completed\":false}".utf8))
            }
            source.stub("videoRestartSource(id:offsetBytes:sourceInstanceId:sourceRef:)") { args in
                let offset = args[1] as! UInt64
                if ownerChanges { current.withLock { $0 = false } }
                return Data(repeating: 7, count: Int(min(UInt64(4 * 1024 * 1024), size - offset)))
            }
            target.stub("appendVideoSourceUpload(id:offsetBytes:bytes:destinationInstanceId:)") { args in
                let id = args[0] as! String
                let offset = args[1] as! UInt64
                let bytes = args[2] as! Data
                return try MoldJSON.decoder.decode(
                    VideoSourceUploadStatus.self,
                    from: Data(
                        "{\"handle\":\"\(id)\",\"offset_bytes\":\"\(offset + UInt64(bytes.count))\",\"completed\":false}"
                            .utf8))
            }
            target.stub("completeVideoSourceUpload(id:destinationInstanceId:)") { args in
                VideoSourceUploadRef(handle: args[0] as! String, sha256: sha, sizeBytes: size)
            }
            let controller = VideoRestartController(defaults: h.helpers.defaults())
            do {
                let grant = try await controller.transferPrefix(
                    exported, source: h.helpers.target(source), sourceBackend: source,
                    sourceJobId: "original", destination: h.helpers.target(target), backend: target,
                    current: { current.withLock { $0 } })
                #expect(!ownerChanges)
                #expect(grant?.sizeBytes == size)
                let writes = target.calls.filter {
                    $0.route == "appendVideoSourceUpload(id:offsetBytes:bytes:destinationInstanceId:)"
                }
                #expect(writes.compactMap { $0.arguments[1] as? UInt64 } == [0, 4 * 1024 * 1024])
                #expect(writes.compactMap { ($0.arguments[2] as? Data)?.count } == [4 * 1024 * 1024, 3])
            } catch {
                #expect(ownerChanges)
                #expect(target.count("appendVideoSourceUpload(id:offsetBytes:bytes:destinationInstanceId:)") == 0)
                #expect(target.count("completeVideoSourceUpload(id:destinationInstanceId:)") == 0)
            }
            #expect(
                target.count(
                    "createVideoJob(requestJSON:validationRevision:operationId:destinationInstanceId:sourceUpload:)")
                    == 0)
        }
    }
}
