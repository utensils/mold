import Foundation
import MoldClient
import Testing

@testable import Mold

/// Stop while an admission is still in the air, and the beat a settled
/// outcome gets before the next batch takes the canvas (findings 02#2, 02#9).
@MainActor
struct RunStopFenceTests {
    private func machine() -> MoldHost {
        MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
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

    /// The client batch id the pane minted for its nth submission, so a test
    /// plants an answer the fake will hand back for that exact admission.
    private func clientId(_ backend: FakeBackend, _ index: Int) -> String {
        backend.submittedAdmissions[index].clientBatchId
    }

    // MARK: - Stop before the answer

    /// **Fails today**: `stop()` guards on `activeBatch`, which on a
    /// first-ever render is nil -- so Stop did nothing at all while the POST
    /// already on its way was admitted, rendered to completion and was never
    /// cancelled.
    @Test func stopOnAFirstEverRenderCancelsTheBatchTheHostAdmits() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.submitAnswers = [status("batch-1", "client-1")]
        backend.holdsSubmit = true

        controller.submit(on: workstation, backend: backend)
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
        #expect(!PendingBatch.all().keys.contains(clientId(backend, 0)))
    }

    /// **Fails today**: Stop during `.submitting` cancelled the PREVIOUS,
    /// already-settled batch and forgot ITS recovery record, while the new
    /// admission rendered on unwatched.
    @Test func stopDuringASecondSubmissionNeverCancelsTheBatchBefore() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.submitAnswers = [status("batch-1", "client-1"), status("batch-2", "client-2")]
        backend.batchStatusAnswers["batch-1"] = FakeFixtures.batchStatus(
            id: "batch-1", clientBatchId: "client-1", [.init(1, state: "complete", seed: 7)])

        // One render, settled, so `run` is `.finished` and nothing is live.
        controller.submit(on: workstation, backend: backend)
        await settle { !controller.run.isBusy }

        backend.holdsSubmit = true
        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.filter { $0 == "submit" }.count == 2 }
        controller.stop()
        backend.releaseSubmit()
        await settle { backend.cancelledBatchIds.count == 1 }

        // The batch that was in the air, never the one that already settled.
        #expect(backend.cancelledBatchIds == ["batch-2"])
    }

    // MARK: - Stop, then Generate again before the answer

    /// **Fails today**: the fence held ONE `stopRequested` flag, and `begin`
    /// reset it -- so pressing Generate again while the stopped POST was still
    /// in the air discarded the Stop entirely. The withdrawn batch landed as
    /// an ordinary queued one, was never cancelled, and took the canvas later.
    /// A user who presses Stop has withdrawn that render.
    @Test func aStoppedSubmissionIsStillCancelledWhenAnotherTakesTheCanvas() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.submitAnswers = [status("batch-1", "client-1"), status("batch-2", "client-2")]
        backend.batchEventsHeldOpen.insert("batch-2")
        backend.holdsSubmit = true

        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.contains("submit") }
        controller.stop()
        // Stop cleared the canvas, so this second press follows at once.
        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.filter { $0 == "submit" }.count == 2 }
        backend.releaseSubmit()

        await settle { backend.cancelledBatchIds.count == 1 }
        #expect(backend.cancelledBatchIds == ["batch-1"])
        // Withdrawn, so it never queues and never takes the canvas.
        #expect(controller.queuedCount == 0)
        #expect(controller.activeBatch?.id != "batch-1")
        #expect(!PendingBatch.all().keys.contains(clientId(backend, 0)))
    }

    /// The second press must not abort the first POST either: `runTask` is the
    /// whole submit-and-follow task, and cancelling it would leave the host
    /// running a batch nobody holds the id of.
    @Test func aSecondPressNeverAbortsAnUnansweredPost() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.submitAnswers = [status("batch-1", "client-1"), status("batch-2", "client-2")]
        backend.batchEventsHeldOpen.insert("batch-2")
        backend.holdsSubmit = true

        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.contains("submit") }
        controller.stop()
        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.filter { $0 == "submit" }.count == 2 }
        backend.releaseSubmit()

        // It answered rather than throwing a cancellation, which is what lets
        // the cancel above reach the host at all.
        await settle { backend.cancelledBatchIds.count == 1 }
        #expect(backend.submittedAdmissions.count == 2)
    }

    // MARK: - Stop, then the submission FAILS

    /// A POST that is refused after a Stop reports nothing and replaces
    /// nothing: the user already withdrew it.
    @Test func aStoppedSubmissionThatFailsIsSilent() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.refuses = ["submit"]
        backend.holdsSubmit = true

        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.contains("submit") }
        controller.stop()
        backend.releaseSubmit()
        await settle { !controller.submissions.hasUnansweredPost }

        guard case .idle = controller.run else {
            Issue.record("expected .idle, got \(controller.run)")
            return
        }
        #expect(controller.hosts.failures.isEmpty)
    }

    // MARK: - Two stops

    /// Two withdrawn submissions are two cancels, in whatever order they land.
    @Test func twoStoppedSubmissionsAreBothCancelled() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.submitAnswers = [status("batch-1", "client-1"), status("batch-2", "client-2")]

        backend.holdsSubmit = true
        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.contains("submit") }
        controller.stop()
        backend.releaseSubmit()
        await settle { backend.cancelledBatchIds.count == 1 }

        backend.holdsSubmit = true
        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.filter { $0 == "submit" }.count == 2 }
        controller.stop()
        backend.releaseSubmit()
        await settle { backend.cancelledBatchIds.count == 2 }

        #expect(backend.cancelledBatchIds == ["batch-1", "batch-2"])
        #expect(controller.queuedCount == 0)
    }

    // MARK: - The fence itself

    /// The keyed rule, without a backend: a stop belongs to the ID it was
    /// aimed at and survives any number of later submissions.
    @Test func theFenceRemembersAStopPerClientBatchId() {
        let fence = SubmissionFence()
        fence.begin("a")
        #expect(fence.requestStop())
        fence.begin("b")
        #expect(fence.land("a") == .cancel)
        #expect(fence.land("b") == .follow)

        // Nothing in the air is the caller's cue to stop what is on screen.
        #expect(!fence.requestStop())

        // A superseded submission queues; it was never withdrawn.
        fence.begin("c")
        fence.begin("d")
        #expect(fence.land("c") == .queue)
        #expect(fence.land("d") == .follow)

        // An unanswered POST is what forbids cancelling the submit task.
        fence.begin("e")
        #expect(fence.hasUnansweredPost)
        _ = fence.land("e")
        #expect(!fence.hasUnansweredPost)
        // Including one already stopped -- it still has to reach its landing.
        fence.begin("f")
        _ = fence.requestStop()
        #expect(fence.hasUnansweredPost)
        #expect(fence.land("f") == .cancel)
        #expect(!fence.hasUnansweredPost)
    }

    // MARK: - The beat before the next batch (02#9)

    /// **Fails today**: `settle` called `followNext()` in the same turn, so
    /// the first batch's picture, result bar and failure summary could be
    /// replaced before any of it was ever drawn.
    @Test func theNextBatchWaitsUntilTheCanvasHasTheResult() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        // A grace long enough that only an explicit acknowledgement can
        // release the queue inside this test.
        let handoff = ResultHandoff(grace: .seconds(30))
        let controller = makeController(backend, host: workstation, handoff: handoff)
        backend.submitAnswers = [status("batch-1", "client-1"), status("batch-2", "client-2")]
        backend.batchEventsHeldOpen.insert("batch-1")

        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.contains("batchEvents") }
        controller.submit(on: workstation, backend: backend)
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
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation, handoff: ResultHandoff(grace: .zero))
        backend.submitAnswers = [status("batch-1", "client-1"), status("batch-2", "client-2")]
        backend.batchEventsHeldOpen.insert("batch-1")

        controller.submit(on: workstation, backend: backend)
        await settle { backend.calls.contains("batchEvents") }
        controller.submit(on: workstation, backend: backend)
        await settle { controller.queuedCount == 1 }

        backend.emitBatchEvent(FakeFixtures.batchStatus(
            id: "batch-1", clientBatchId: "client-1",
            [.init(1, state: "complete", seed: 1)]), for: "batch-1")

        await settle { controller.activeBatch?.id == "batch-2" }
        #expect(controller.queuedCount == 0)
    }
}
