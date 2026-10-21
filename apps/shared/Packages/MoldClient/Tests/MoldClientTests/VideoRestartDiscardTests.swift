import Foundation
import MoldClientTesting
import Testing

@testable import MoldClient

@MainActor struct VideoRestartDiscardTests {
    let h = VideoRestartControllerTests()
    @Test func deletionRequiresExplicitOfferAndForwardsExactRevisionWithoutRestarting() async throws {
        for supported in [false, true] {
            let backend = try h.source()
            backend.stub(
                "chainRestartOptions(id:)",
                returning: try h.decode(
                    "{\"source_instance_id\":\"source\",\"source_revision\":\"captured\",\"restart\":{\"supported\":true},\"send\":{\"supported\":true},\"discard\":{\"supported\":\(supported)}}",
                    as: ChainRestartOptions.self))
            backend.stub("discardChainJob(id:request:)", returning: ())
            let controller = VideoRestartController(defaults: h.defaults())
            await controller.open(source: h.target(backend), jobId: "original", backend: backend, current: { true })
            #expect(await controller.discardSource() == supported)
            #expect(backend.count("discardChainJob(id:request:)") == (supported ? 1 : 0))
            if supported {
                let sent = try #require(backend.calls.first(where: { $0.route == "discardChainJob(id:request:)" }))
                let authority = try #require(sent.arguments[1] as? ChainDiscardRequest)
                #expect(authority.sourceRevision == "captured" && authority.sourceInstanceId == "source")
            }
            #expect(backend.count("exportChainRestart(id:request:)") == 0)
        }
    }
    @Test func lostDeleteReplyOnlyAcknowledgesExactMissingJobOnSameInstallation() async throws {
        let backend = try h.source()
        backend.stub(
            "chainRestartOptions(id:)",
            returning: try h.decode(
                #"{"source_instance_id":"source","source_revision":"captured","restart":{"supported":true},"send":{"supported":true},"discard":{"supported":true}}"#,
                as: ChainRestartOptions.self))
        backend.stub("discardChainJob(id:request:)", throwing: URLError(.networkConnectionLost))
        backend.stub(
            "chainJob(id:)",
            throwing: MoldClientError.http(status: 404, code: "CHAIN_JOB_NOT_FOUND", message: "missing"))
        let controller = VideoRestartController(defaults: h.defaults())
        await controller.open(source: h.target(backend), jobId: "original", backend: backend, current: { true })
        #expect(await controller.discardSource())
        #expect(backend.count("discardChainJob(id:request:)") == 1)
        #expect(backend.count("chainJob(id:)") == 1)
    }
    @Test func rejectedDeleteCannotHideExistingJobEvenWhenErrorWas404() async throws {
        let backend = try h.source()
        backend.stub(
            "chainRestartOptions(id:)",
            returning: try h.decode(
                #"{"source_instance_id":"source","source_revision":"captured","restart":{"supported":true},"send":{"supported":true},"discard":{"supported":true}}"#,
                as: ChainRestartOptions.self))
        backend.stub(
            "discardChainJob(id:request:)",
            throwing: MoldClientError.http(status: 404, code: "NOT_FOUND", message: "route missing"))
        backend.stub(
            "chainJob(id:)",
            returning: try h.decode(
                #"{"id":"original","state":"failed","model":"model","stage_count":1,"current_stage":0,"stages":[]}"#,
                as: ChainJobDetail.self))
        let controller = VideoRestartController(defaults: h.defaults())
        await controller.open(source: h.target(backend), jobId: "original", backend: backend, current: { true })
        #expect(!(await controller.discardSource()))
        #expect(controller.error != nil)
    }

}
