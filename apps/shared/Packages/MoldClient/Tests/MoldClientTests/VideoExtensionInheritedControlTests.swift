import Foundation
import MoldClientTesting
import Testing
@testable import MoldClient

@MainActor struct VideoExtensionInheritedControlTests {
    @Test func unsupportedHistoricalControlsRefuseInsteadOfBecomingUncontrolledVideo() async throws {
        for extra in [#", "ic_lora_control":"depth""#, #", "control_model":"controlnet:fp16""#, #", "control_scale":0.5"#] {
            let backend = FakeBackend()
            let owner = VideoJobOwner(host: backend.host, instanceId: "original")
            let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data("{\"model\":\"video\",\"prompt\":\"tail\"\(extra)}".utf8))
            backend.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self, from: Data(#"{"version":"test","busy":false,"uptime_secs":1,"instance_id":"original"}"#.utf8)))
            backend.stub("capabilities()", returning: try MoldJSON.decoder.decode(Capabilities.self, from: Data(#"{"video_jobs":{"supported":true,"literal_prompts":true,"library_extend":true}}"#.utf8)))
            backend.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .unavailableLegacy))
            do {
                _ = try await VideoLibraryExtensionSource.inspect(filename: "original.mp4", title: "Original", metadata: metadata,
                    owner: owner, backend: backend, current: { true })
                Issue.record("Historical control settings were silently dropped")
            } catch {
                #expect(error.localizedDescription.contains("controls"))
                #expect(error.localizedDescription.contains("Start again"))
                #expect(error.localizedDescription.contains("CLI"))
            }
            #expect(backend.count("retainedSourceMediaBytes(for:member:)") == 0)
            #expect(backend.count("createVideoJob(_:operationId:destinationInstanceId:)") == 0)
        }
    }
}
