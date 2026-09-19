import Foundation
import MoldClient
import Testing

@testable import Mold

/// M8 decision 8: Generate never turns into Stop. A second admission while
/// one render is on screen still goes to the host -- its queue is durable --
/// and waits in `queued` rather than being followed at once. These pin the
/// FIFO: a second `submit` is admitted and queued, the queue advances when
/// the followed batch settles or is stopped, and a refused second admission
/// never disturbs what is already on screen.
@MainActor
struct RunQueueTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func makeController(_ backend: FakeBackend, host: MoldHost) -> GenerateController {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.modelName = "flux-dev:q4"
        controller.hostID = host.id
        controller.draft.prompt = "a cat"
        return controller
    }

    // MARK: - Admitting a second batch while one runs

    @Test func aSecondGenerateWhileOneRunsIsAdmittedAndQueued() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        let first = FakeFixtures.batchStatus(id: "batch-1", clientBatchId: "client-1", [.init(1, state: "running")])
        let second = FakeFixtures.batchStatus(id: "batch-2", clientBatchId: "client-2", [.init(1, state: "running")])
        backend.submitAnswers = [first, second]
        // Backend tasks may enter in either order, so either answer can
        // belong to the first click. Keep both streams genuinely running.
        backend.batchEventsHeldOpen.formUnion(["batch-1", "batch-2"])
        defer { controller.stopAll() }

        controller.submit(on: workstation, backend: backend)
        // `run = .submitting` is set synchronously, before either `Task`
        // has run -- so this holds true no matter how the two `Task`s
        // launched below later interleave.
        #expect(controller.run.isBusy)
        controller.draft.prompt = "a dog"
        controller.submit(on: workstation, backend: backend)

        // Wait for adopted state: recording a POST precedes adopting its answer.
        await settle { controller.activeBatch != nil && controller.queuedCount == 1 }

        #expect(backend.calls.filter { $0 == "submit" }.count == 2)
        #expect(controller.run.isBusy)
        let firstAdmission = backend.submittedAdmissions.first { $0.requests.first?.prompt == "a cat" }
        let secondAdmission = backend.submittedAdmissions.first { $0.requests.first?.prompt == "a dog" }
        #expect(firstAdmission != nil && secondAdmission != nil)
        #expect(controller.activeBatch?.clientBatchId == firstAdmission?.clientBatchId)
        #expect(controller.queuedCount == 1)
        if case let .batch(queued) = controller.queued.first {
            #expect(queued.clientBatchId == secondAdmission?.clientBatchId)
        } else {
            Issue.record("expected the second click to be queued")
        }
    }

    // MARK: - Advancing the queue

    @Test func theNextQueuedBatchIsFollowedWhenTheFirstSettles() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        let first = FakeFixtures.batchStatus(id: "batch-1", clientBatchId: "client-1", [.init(1, state: "running")])
        let second = FakeFixtures.batchStatus(id: "batch-2", clientBatchId: "client-2", [.init(1, state: "running")])
        backend.submitAnswers = [first, second]
        // Held open so the test decides exactly when `batch-1` settles,
        // rather than the fake's default stream finishing before the second
        // `submit` below has had a chance to land in `queued`.
        backend.batchEventsHeldOpen.insert("batch-1")

        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.contains("batchEvents") }
        controller.submit(on: workstation, backend: backend)
        await settle { controller.queuedCount == 1 }

        let settled = FakeFixtures.batchStatus(
            id: "batch-1", clientBatchId: "client-1", [.init(1, state: "complete", seed: 1)])
        backend.emitBatchEvent(settled, for: "batch-1")
        backend.finishBatchEvents(for: "batch-1")

        // What the canvas does once the outcome is really on screen; without
        // it the queue waits out `ResultHandoff`'s grace instead (02#9).
        await settle { controller.handoff.isHolding }
        controller.handoff.acknowledge()
        await settle { controller.queuedCount == 0 }

        #expect(controller.activeBatch?.id == "batch-2")
        #expect(controller.queuedCount == 0)
    }

    @Test func stopMovesOnToTheNextQueuedBatch() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        let first = FakeFixtures.batchStatus(id: "batch-1", clientBatchId: "client-1", [.init(1, state: "running")])
        let second = FakeFixtures.batchStatus(id: "batch-2", clientBatchId: "client-2", [.init(1, state: "running")])
        backend.submitAnswers = [first, second]
        backend.batchStatusAnswers["batch-1"] = first

        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.contains("batchStatus") }
        controller.submit(on: workstation, backend: backend)
        await settle { controller.queuedCount == 1 }

        controller.stop()
        await settle { backend.calls.filter { $0 == "cancelBatch" }.count == 1 }

        #expect(backend.calls.filter { $0 == "cancelBatch" }.count == 1)
        #expect(controller.activeBatch?.id == "batch-2")
    }

    @Test func stopAllCancelsEverythingThisPaneAdmitted() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        let first = FakeFixtures.batchStatus(id: "batch-1", clientBatchId: "client-1", [.init(1, state: "running")])
        let second = FakeFixtures.batchStatus(id: "batch-2", clientBatchId: "client-2", [.init(1, state: "running")])
        let third = FakeFixtures.batchStatus(id: "batch-3", clientBatchId: "client-3", [.init(1, state: "running")])
        backend.submitAnswers = [first, second, third]
        backend.batchStatusAnswers["batch-1"] = first

        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.contains("batchStatus") }
        controller.submit(on: workstation, backend: backend)
        await settle { controller.queuedCount == 1 }
        controller.submit(on: workstation, backend: backend)
        await settle { controller.queuedCount == 2 }

        controller.stopAll()
        await settle { backend.calls.filter { $0 == "cancelBatch" }.count == 3 }

        #expect(backend.calls.filter { $0 == "cancelBatch" }.count == 3)
        #expect(controller.queuedCount == 0)
        guard case .idle = controller.run else {
            Issue.record("expected .idle, got \(controller.run)")
            return
        }
    }
}
