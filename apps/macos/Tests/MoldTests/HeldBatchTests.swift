import Foundation
import MoldClient
import Testing

@testable import Mold

/// A HELD batch is not a render in progress: the machine has parked it until
/// a person retries, moves or cancels it in the Queue. The pane used to treat
/// it as live -- `isSettled` is false for a hold -- so a followed batch that
/// went on hold spun "Getting ready…" for ever, and every launch after that
/// re-attached to it and spun again, which read as the app generating on its
/// own at startup.
@MainActor
struct HeldBatchTests {
    private func machine() -> MoldHost {
        MoldHost(name: "plato", baseURL: URL(string: "http://plato")!)
    }

    private func makeController(_ backend: FakeBackend, host: MoldHost) -> GenerateController {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.modelName = "flux-dev:q4"
        controller.hostID = host.id
        controller.draft.prompt = "a cat"
        return controller
    }

    private let reason = "needs more device memory than cuda:0 has free"

    /// **Fails today**: recovery follows anything unsettled, a hold included.
    @Test func aHeldBatchIsNotReattachedAtLaunch() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato)
        PendingBatch.remember("client-held", host: plato.id)
        backend.batchStatusByClientId["client-held"] = FakeFixtures.batchStatus(
            id: "batch-held", clientBatchId: "client-held", [.init(1, state: "held", error: reason)])

        await controller.recoverPending()

        #expect(!controller.run.isBusy)
        #expect(controller.activeBatch == nil)
        #expect(PendingBatch.all()["client-held"] == nil)
        #expect(backend.callCount("batchEvents") == 0)
    }

    /// **Fails today**: the follow loop only ends on a settled frame.
    @Test func aFollowedBatchThatGoesOnHoldStopsSpinningAndPointsAtTheQueue() async {
        let plato = machine()
        let backend = FakeBackend(host: plato)
        let controller = makeController(backend, host: plato)
        backend.submitAnswers = [FakeFixtures.batchStatus(
            id: "batch-1", clientBatchId: "client-1", [.init(1, state: "accepted")])]
        backend.batchEventsHeldOpen = ["batch-1"]

        controller.submit(on: plato, backend: backend)
        await settle { backend.batchEventsContinuations["batch-1"] != nil }
        backend.emitBatchEvent(FakeFixtures.batchStatus(
            id: "batch-1", clientBatchId: "client-1", [.init(1, state: "held", error: reason)]), for: "batch-1")
        await settle { !controller.run.isBusy }

        guard case let .failed(sentence) = controller.run else {
            Issue.record("expected the hold to end the run, got \(controller.run)")
            return
        }
        #expect(sentence.contains(reason))
        #expect(sentence.contains("Queue"))
    }

    /// Pictures that DID arrive are still shown when a sibling is held.
    @Test func aPartlyHeldBatchShowsWhatArrived() {
        let status = FakeFixtures.batchStatus([
            .init(1, state: "complete"), .init(2, state: "held", error: reason),
        ])
        let outcome = BatchOutcome(settling: status)
        #expect(outcome?.results.count == 1)
        #expect(outcome?.failures.count == 1)
    }

    /// Anything still able to move on its own keeps the pane following.
    @Test func aBatchWithARunningChildIsNotAtRest() {
        let status = FakeFixtures.batchStatus([
            .init(1, state: "running"), .init(2, state: "held", error: reason),
        ])
        #expect(!status.isAtRest)
        #expect(BatchOutcome(settling: status) == nil)
    }
}
