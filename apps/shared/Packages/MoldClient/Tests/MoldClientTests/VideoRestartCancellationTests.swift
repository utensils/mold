import Foundation
import Testing
import MoldClientTesting
@testable import MoldClient

extension VideoRestartControllerTests {
    @Test func lostStopReplyIsReconciledOnReopenUsingOnlyTheSameKnownOperation() async throws {
        let source = try source(), destination = try backend("target"), saved = defaults()
        destination.stub("capabilities()", returning: try decode(#"{"queue":{"chain_restart":true,"chain_restart_controls":true}}"#, as: Capabilities.self))
        destination.stub("createChainJob(requestJSON:operationId:destinationInstanceId:)", throwing: MoldClientError.unreachable("Lost reply"))
        destination.stub("cancelChainOperation(id:operationId:destinationInstanceId:)", throwing: MoldClientError.unreachable("Lost Stop reply"))
        let controller = VideoRestartController(defaults: saved)
        await controller.open(source: target(source), jobId: "original", backend: source, current: { true })
        await controller.start(destination: target(destination), backend: destination, current: { true })
        let id = try #require(controller.admission?.operationId)
        await controller.cancel(destination: target(destination), backend: destination, current: { true })
        #expect(controller.pending.first?.cancelRequested == true)
        #expect(controller.pending.first?.cancelAcknowledged == false)
        destination.stub("cancelChainOperation(id:operationId:destinationInstanceId:)", returning: ())
        destination.stub("chainJob(id:)", throwing: MoldClientError.http(status: 404, code: "CHAIN_JOB_NOT_FOUND", message: "No job"))
        let recovered = VideoRestartController(defaults: saved)
        await recovered.open(source: target(source), jobId: "original", backend: source, current: { true })
        await recovered.refresh(destination: target(destination), backend: destination, current: { true })
        #expect(recovered.cancelledBeforeAdmission)
        #expect(recovered.pending.isEmpty)
        let stops = destination.calls.filter { $0.route == "cancelChainOperation(id:operationId:destinationInstanceId:)" }
        #expect(stops.count == 2)
        #expect(stops.allSatisfy { $0.arguments[0] as? String == id && $0.arguments[1] as? String == id && $0.arguments[2] as? String == "target" })
        #expect(destination.count("createChainJob(requestJSON:operationId:destinationInstanceId:)") == 1)
        #expect(source.count("exportChainRestart(id:request:)") == 1)
    }

    @Test func stopRequiresAffirmativeControlCapabilityAndCurrentDestination() async throws {
        let source = try source(), destination = try backend("target")
        destination.stub("createChainJob(requestJSON:operationId:destinationInstanceId:)", throwing: MoldClientError.unreachable("Lost reply"))
        let controller = VideoRestartController(defaults: defaults())
        await controller.open(source: target(source), jobId: "original", backend: source, current: { true })
        await controller.start(destination: target(destination), backend: destination, current: { true })
        await controller.cancel(destination: target(destination), backend: destination, current: { true })
        #expect(controller.pending.count == 1)
        #expect(controller.error != nil)
        #expect(destination.count("cancelChainOperation(id:operationId:destinationInstanceId:)") == 0)
        #expect(destination.count("cancelChainOperation(id:operationId:)") == 0)
        destination.stub("capabilities()", returning: try decode(#"{"queue":{"chain_restart":true,"chain_restart_controls":true}}"#, as: Capabilities.self))
        await controller.refresh(destination: target(destination), backend: destination, current: { false })
        #expect(destination.count("cancelChainOperation(id:operationId:destinationInstanceId:)") == 0)
    }
}
