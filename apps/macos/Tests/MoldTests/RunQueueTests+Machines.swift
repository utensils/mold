import Foundation
import MoldClient
import Testing

@testable import Mold

/// The queue across two machines, and a refused second admission -- split
/// from `RunQueueTests.swift` for size.
@MainActor
struct RunQueueMachinesTests {
    private func machine(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func running(_ id: String) -> BatchStatus {
        FakeFixtures.batchStatus(id: id, clientBatchId: "client-\(id)", [.init(1, state: "running")])
    }

    /// The Machine control can be moved between two presses, so the second
    /// batch can belong to another machine -- and it must be followed, and
    /// stopped, on THAT machine's connection, never the first one's.
    @Test func aBatchQueuedOnAnotherMachineIsFollowedThere() async {
        let plato = machine("plato"), hal = machine("hal9000")
        let onPlato = FakeBackend(host: plato), onHal = FakeBackend(host: hal)
        let hosts = HostStore(hosts: [plato, hal]) { $0.id == plato.id ? onPlato : onHal }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.modelName = "flux-dev:q4"
        controller.draft.prompt = "a cat"
        onPlato.submitAnswer = running("batch-1")
        onPlato.batchEventsHeldOpen.insert("batch-1")
        onHal.submitAnswer = running("batch-2")
        onHal.batchStatusAnswers["batch-2"] = running("batch-2")

        controller.submit(on: plato, backend: onPlato)
        await settle { onPlato.calls.contains("batchEvents") }
        controller.submit(on: hal, backend: onHal)
        await settle { controller.queuedCount == 1 }

        onPlato.emitBatchEvent(
            FakeFixtures.batchStatus(id: "batch-1", clientBatchId: "client-batch-1", [.init(1, state: "complete", seed: 1)]),
            for: "batch-1")
        onPlato.finishBatchEvents(for: "batch-1")
        // What the canvas does once the outcome is really on screen; without
        // it the queue waits out `ResultHandoff`'s grace instead (02#9).
        await settle { controller.handoff.isHolding }
        controller.handoff.acknowledge()
        await settle { onHal.calls.contains("batchEvents") }

        #expect(controller.activeBatch?.host == hal.id)
        #expect(onPlato.calls.filter { $0 == "batchEvents" }.count == 1)

        controller.stop()
        await settle { onHal.calls.contains("cancelBatch") }
        #expect(!onPlato.calls.contains("cancelBatch"))
    }

    // MARK: - A refused second admission

    @Test func aRefusedSecondAdmissionDoesNotDisturbTheRenderOnScreen() async {
        let plato = machine("plato")
        let backend = FakeBackend(host: plato)
        let hosts = HostStore(hosts: [plato]) { _ in backend }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.modelName = "flux-dev:q4"
        controller.hostID = plato.id
        controller.draft.prompt = "a cat"
        let first = FakeFixtures.batchStatus(id: "batch-1", clientBatchId: "client-1", [.init(1, state: "running")])
        backend.submitAnswer = first
        backend.batchStatusAnswers["batch-1"] = first

        controller.submit(on: plato, backend: backend)
        await settle { backend.calls.contains("batchStatus") }

        backend.plantedErrors["submit"] = MoldClientError.http(status: 409, code: nil, message: "Refused.")
        controller.submit(on: plato, backend: backend)
        await settle { backend.calls.filter { $0 == "submit" }.count == 2 }
        await settle { !controller.hosts.failures.isEmpty }

        #expect(controller.run.isBusy)
        #expect(controller.activeBatch?.id == "batch-1")
        #expect(controller.queuedCount == 0)
        #expect(controller.hosts.failures.contains { $0.host == plato.id && $0.verb == "queue that render" })
    }
}
