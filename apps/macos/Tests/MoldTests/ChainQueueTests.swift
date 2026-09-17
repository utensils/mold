import Foundation
import MoldClient
import Testing

@testable import Mold

/// M8 decision 8 applies to a chain too: a press while something is on screen
/// ADMITS the work — the host's queue is durable — and waits its turn.
///
/// **Fails today**: `ChainSubmission.take` ran before `followingNow` and
/// cancelled `runTask` unconditionally, so a chain press wiped a live batch
/// off the canvas, a second chain press orphaned the first on the GPU, and a
/// running batch was dropped and never re-queued.
@MainActor
struct ChainQueueTests {
    private func machine() -> MoldHost {
        MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
    }

    private func makeController(_ backend: FakeBackend, host: MoldHost) -> GenerateController {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.modelName = "ltx-2-19b:fp8"
        controller.modelFamily = "ltx2"
        controller.hostID = host.id
        controller.draft.prompt = "a tin robot"
        controller.draft.frames = 249
        return controller
    }

    private func chainAnswer(_ id: String) -> CreateChainJobResponse {
        try! MoldJSON.decoder.decode(
            CreateChainJobResponse.self, from: Data(#"{"job_id": "\#(id)"}"#.utf8))
    }

    private func event(_ json: String) -> ChainJobEvent {
        try! MoldJSON.decoder.decode(ChainJobEvent.self, from: Data(json.utf8))
    }

    private let routing = ChainRouting.Decision.chain(
        clipFrames: 97, motionTail: 17, stageCount: 3)

    /// Image, then a long clip. The batch keeps the canvas; the chain is
    /// admitted and waits. The batch's POST is never cancelled.
    @Test func aChainPressBehindARunningBatchWaitsItsTurn() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato)
        let first = FakeFixtures.batchStatus(
            id: "batch-1", clientBatchId: "client-1", [.init(1, state: "running")])
        backend.submitAnswers = [first]
        backend.batchEventsHeldOpen.insert("batch-1")
        backend.chainJobAnswer = chainAnswer("chain-q1")

        controller.submit(on: plato, backend: backend)
        await settle { backend.calls.contains("batchEvents") }

        controller.submit(on: plato, backend: backend, routing: routing)
        await settle { controller.queuedCount == 1 }

        // The batch is untouched and still on the canvas.
        #expect(controller.activeBatch?.id == "batch-1")
        #expect(controller.run.isBusy)
        #expect(controller.run.stage == nil, "the chain did not take the canvas")
        #expect(backend.cancelledBatchIds.isEmpty)
        // And the chain IS admitted -- the host's queue is durable, so a press
        // is never thrown away just because something else is showing.
        #expect(backend.calls.contains("createChainJob"))

        // When the batch settles, the chain is followed in its turn.
        backend.chainEventsHeldOpen.insert("chain-q1")
        let settled = FakeFixtures.batchStatus(
            id: "batch-1", clientBatchId: "client-1", [.init(1, state: "complete", seed: 1)])
        backend.emitBatchEvent(settled, for: "batch-1")
        backend.finishBatchEvents(for: "batch-1")
        await settle { controller.run.stage == "Clip 1 of 3" }
        #expect(controller.queuedCount == 0)
    }

    /// A long clip, then an image. The chain keeps the canvas and the batch
    /// queues -- the mirror of the case above.
    @Test func abatchPressBehindARunningChainWaitsItsTurn() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato)
        backend.chainJobAnswer = chainAnswer("chain-q2")
        backend.chainEventsHeldOpen.insert("chain-q2")
        backend.submitAnswers = [FakeFixtures.batchStatus(
            id: "batch-2", clientBatchId: "client-2", [.init(1, state: "running")])]

        controller.submit(on: plato, backend: backend, routing: routing)
        await settle { controller.run.stage == "Clip 1 of 3" }

        controller.submit(on: plato, backend: backend)
        await settle { controller.queuedCount == 1 }
        #expect(controller.run.stage == "Clip 1 of 3", "the batch did not take the canvas")

        backend.emitChainEvent(
            event(#"{"type":"finalized","gallery_filename":"long.mp4"}"#), for: "chain-q2")
        await settle { controller.activeBatch?.id == "batch-2" }
        #expect(controller.queuedCount == 0)
    }

    /// Two long clips. The second waits; the FIRST is never orphaned on the
    /// GPU and never loses its recovery record.
    @Test func asecondChainNeverOrphansTheFirst() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato)
        backend.chainJobAnswer = chainAnswer("chain-q3")
        backend.chainEventsHeldOpen.insert("chain-q3")

        controller.submit(on: plato, backend: backend, routing: routing)
        await settle { controller.run.stage == "Clip 1 of 3" }

        backend.chainJobAnswer = chainAnswer("chain-q4")
        controller.submit(on: plato, backend: backend, routing: routing)
        await settle { controller.queuedCount == 1 }

        #expect(backend.cancelledChainJobIds.isEmpty, "the first chain was cancelled")
        #expect(PendingChain.all()["chain-q3"] == plato.id.uuidString)
        #expect(controller.run.stage == "Clip 1 of 3")
    }

    /// Stop All withdraws the waiting chain on its own machine too -- a queued
    /// chain is a REAL job on the host, not a local intention.
    @Test func stopAllWithdrawsAWaitingChain() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato)
        backend.chainJobAnswer = chainAnswer("chain-q5")
        backend.chainEventsHeldOpen.insert("chain-q5")

        controller.submit(on: plato, backend: backend, routing: routing)
        await settle { controller.run.stage == "Clip 1 of 3" }
        backend.chainJobAnswer = chainAnswer("chain-q6")
        controller.submit(on: plato, backend: backend, routing: routing)
        await settle { controller.queuedCount == 1 }

        controller.stopAll()
        await settle { backend.cancelledChainJobIds.count == 2 }
        #expect(Set(backend.cancelledChainJobIds) == ["chain-q5", "chain-q6"])
        #expect(controller.queuedCount == 0)
        #expect(PendingChain.all()["chain-q6"] == nil)
    }
}
