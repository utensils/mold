import Foundation
import MoldClient
import Testing

@testable import Mold

/// Stop while an admission is still in the air, and the beat a settled
/// outcome gets before the next batch takes the canvas (findings 02#2, 02#9).
@MainActor
struct RunStopFenceTests {
    private func machine() -> MoldHost {
        MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
    }

    private func makeController(
        _ backend: FakeBackend, host: MoldHost, handoff: ResultHandoff = ResultHandoff()
    ) -> GenerateController {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let controller = GenerateController(
            hosts: hosts, defaults: ConfigStore(hosts: hosts), handoff: handoff)
        controller.modelName = "flux-dev:q4"
        controller.hostID = host.id
        controller.draft.prompt = "a cat"
        return controller
    }

    private func status(_ id: String, _ clientId: String) -> BatchStatus {
        FakeFixtures.batchStatus(id: id, clientBatchId: clientId, [.init(1, state: "running")])
    }

    /// **Fails today**: `stop()` guards on `activeBatch`, which on a
    /// first-ever render is nil -- so Stop did nothing at all while the POST
    /// already on its way was admitted, rendered to completion and was never
    /// cancelled.
    @Test func stopOnAFirstEverRenderCancelsTheBatchTheHostAdmits() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato)
        backend.submitAnswers = [status("batch-1", "client-1")]
        backend.holdsSubmit = true

        controller.submit(on: plato, backend: backend)
        await settle { backend.calls.contains("submit") }
        controller.stop()
        // The button stops being a Stop the moment it is pressed, even though
        // the id it is aimed at does not exist yet.
        #expect(!controller.run.isBusy)

        backend.releaseSubmit()
        await settle { backend.cancelledBatchIds.count == 1 }
        #expect(backend.cancelledBatchIds == ["batch-1"])
        #expect(controller.activeBatch == nil)
        #expect(!backend.calls.contains("batchEvents"))
    }

    /// **Fails today**: Stop during `.submitting` cancelled the PREVIOUS,
    /// already-settled batch and forgot ITS recovery record, while the new
    /// admission rendered on unwatched.
    @Test func stopDuringASecondSubmissionNeverCancelsTheBatchBefore() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato)
        let first = status("batch-1", "client-1")
        backend.submitAnswers = [first, status("batch-2", "client-2")]
        backend.batchStatusAnswers["batch-1"] = FakeFixtures.batchStatus(
            id: "batch-1", clientBatchId: "client-1", [.init(1, state: "complete", seed: 7)])

        // One render, settled, so `run` is `.finished` and nothing is live.
        controller.submit(on: plato, backend: backend)
        await settle { !controller.run.isBusy }

        backend.holdsSubmit = true
        controller.submit(on: plato, backend: backend)
        await settle { backend.calls.filter { $0 == "submit" }.count == 2 }
        controller.stop()
        backend.releaseSubmit()
        await settle { backend.cancelledBatchIds.count == 1 }

        // The batch that was in the air, never the one that already settled.
        #expect(backend.cancelledBatchIds == ["batch-2"])
    }

    /// A second press while a Stop-ed submission is still in the air takes the
    /// canvas; the earlier one queues rather than being cancelled or stomping
    /// over what replaced it (M8 decision 8).
    @Test func aSubmissionSupersededWhileInFlightQueuesInstead() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato)
        backend.submitAnswers = [status("batch-1", "client-1"), status("batch-2", "client-2")]
        backend.batchEventsHeldOpen.insert("batch-2")
        backend.holdsSubmit = true

        controller.submit(on: plato, backend: backend)
        await settle { backend.calls.contains("submit") }
        controller.stop()
        // Stop cleared the canvas, so this second press follows at once.
        controller.submit(on: plato, backend: backend)
        await settle { backend.calls.filter { $0 == "submit" }.count == 2 }
        backend.releaseSubmit()

        await settle { controller.queuedCount == 1 }
        #expect(backend.cancelledBatchIds.isEmpty)
        #expect(controller.queuedCount == 1)
    }

    /// **Fails today**: `settle` called `followNext()` in the same turn, so
    /// the first batch's picture, result bar and failure summary could be
    /// replaced before any of it was ever drawn.
    @Test func theNextBatchWaitsUntilTheCanvasHasTheResult() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        // A grace long enough that only an explicit acknowledgement can
        // release the queue inside this test.
        let handoff = ResultHandoff(grace: .seconds(30))
        let controller = makeController(backend, host: plato, handoff: handoff)
        backend.submitAnswers = [status("batch-1", "client-1"), status("batch-2", "client-2")]
        backend.batchEventsHeldOpen.insert("batch-1")

        controller.submit(on: plato, backend: backend)
        await settle { backend.calls.contains("batchEvents") }
        controller.submit(on: plato, backend: backend)
        await settle { controller.queuedCount == 1 }

        backend.emitBatchEvent(FakeFixtures.batchStatus(
            id: "batch-1", clientBatchId: "client-1",
            [.init(1, state: "complete", seed: 1)]), for: "batch-1")
        await settle { handoff.isHolding }

        // Settled -- and the queue has NOT moved on.
        #expect(controller.queuedCount == 1)
        guard case .finished = controller.run else {
            Issue.record("expected .finished, got \(controller.run)")
            return
        }

        handoff.acknowledge()
        await settle { controller.queuedCount == 0 }
        #expect(controller.activeBatch?.id == "batch-2")
    }

    /// The pane can be off screen entirely, so the grace period is the belt:
    /// a queue must never wait for a view nobody is looking at.
    @Test func theQueueMovesOnAnywayWhenNobodyIsLooking() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato, handoff: ResultHandoff(grace: .zero))
        backend.submitAnswers = [status("batch-1", "client-1"), status("batch-2", "client-2")]
        backend.batchEventsHeldOpen.insert("batch-1")

        controller.submit(on: plato, backend: backend)
        await settle { backend.calls.contains("batchEvents") }
        controller.submit(on: plato, backend: backend)
        await settle { controller.queuedCount == 1 }

        backend.emitBatchEvent(FakeFixtures.batchStatus(
            id: "batch-1", clientBatchId: "client-1",
            [.init(1, state: "complete", seed: 1)]), for: "batch-1")

        await settle { controller.activeBatch?.id == "batch-2" }
        #expect(controller.queuedCount == 0)
    }
}
