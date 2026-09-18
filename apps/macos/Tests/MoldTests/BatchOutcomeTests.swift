import Foundation
import MoldClient
import Testing

@testable import Mold

/// `GenerateController.submit` sent one request no matter how many copies
/// were asked for, and the server refuses any `batch_size != 1` per child --
/// so "Batch 4" was a 422, never four pictures. These pin the fan-out
/// (`RenderDraft.requests(model:copies:randomBase:)`, wired in at S6) all the
/// way through settling every child, not just one.
@MainActor
struct BatchOutcomeTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func makeController(_ backend: FakeBackend, host: MoldHost) -> GenerateController {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let controller = GenerateController(
            hosts: hosts, defaults: ConfigStore(hosts: hosts), probe: PlacementProbe(debounce: .zero))
        controller.modelName = "flux-dev:q4"
        controller.hostID = host.id
        controller.draft.prompt = "a cat"
        return controller
    }

    // MARK: - The fan-out

    /// **Fails today**: `submit` builds `[draft.request(model:)]`, always one
    /// request regardless of `batchSize`.
    @Test func aBatchOfFourIsSubmittedAsFourChildren() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        controller.draft.batchSize = 4

        controller.submit(on: workstation, backend: backend)
        // `submit` fires an unstructured `Task` and returns immediately --
        // awaiting that same task (rather than polling a call count) is what
        // actually waits for `backend.submit` to have run.
        await controller.runTask?.value

        let requests = backend.submittedAdmissions.last?.requests ?? []
        #expect(requests.count == 4)
        #expect(requests.allSatisfy { $0.batchSize == 1 })
        #expect(Set(requests.compactMap(\.batchId)).count == 1)
        #expect(requests.map(\.batchIndex) == [1, 2, 3, 4])
        #expect(requests.map(\.batchCount) == [4, 4, 4, 4])
    }

    @Test func aBatchOfOneIsUnchanged() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        controller.draft.batchSize = 1

        controller.submit(on: workstation, backend: backend)
        await controller.runTask?.value

        let requests = backend.submittedAdmissions.last?.requests ?? []
        #expect(requests.count == 1)
        #expect(requests.first?.batchId == nil)
        #expect(requests.first?.batchIndex == nil)
        #expect(requests.first?.batchCount == nil)
    }

    // MARK: - Settling every child

    /// Out-of-index-order on the wire, on purpose: the outcome sorts, it
    /// doesn't just echo the array back.
    @Test func everyChildThatMadeSomethingIsInTheOutcome() {
        let status = FakeFixtures.batchStatus([
            .init(4, state: "complete", seed: 4),
            .init(1, state: "complete", seed: 1),
            .init(3, state: "complete", seed: 3),
            .init(2, state: "complete", seed: 2),
        ])
        let outcome = BatchOutcome(settling: status)
        #expect(outcome?.results.map(\.seed) == [1, 2, 3, 4])
        #expect(outcome?.failures.isEmpty == true)
    }

    /// The canvas and the result bar fetch a finished render from the
    /// machine it RAN on. They read the pane's current machine, so switching
    /// machines after Generate -- or a default changing underneath -- turned
    /// a finished render into "That didn't arrive · Image not found" with
    /// Save and Copy still offered (2026-09-17).
    ///
    /// **Fails today**: `RunState` has no such reading.
    @Test func aFinishedRunRemembersTheMachineItRanOn() {
        let ran = UUID()
        let status = FakeFixtures.batchStatus([.init(1, state: "complete", seed: 1)])
        let outcome = try! #require(BatchOutcome(settling: status))
        #expect(RunState.finished(outcome, host: ran).finishedHost == ran)
        #expect(RunState.idle.finishedHost == nil)
        #expect(RunState.failed("x").finishedHost == nil)
    }

    @Test func aPartlyFinishedBatchShowsWhatItMadeAndSaysWhatItDidNot() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        controller.draft.batchSize = 4

        let final = FakeFixtures.batchStatus([
            .init(1, state: "complete", seed: 1),
            .init(2, state: "complete", seed: 2),
            .init(3, state: "complete", seed: 3),
            .init(4, state: "failed", error: "Ran out of memory."),
        ])
        backend.submitAnswer = final
        backend.batchStatusAnswers[final.id] = final

        controller.submit(on: workstation, backend: backend)
        await controller.runTask?.value

        guard case let .finished(outcome, _) = controller.run else {
            Issue.record("expected .finished, got \(controller.run)")
            return
        }
        #expect(outcome.results.count == 3)
        #expect(outcome.failures == ["Ran out of memory."])
    }

    @Test func aBatchWhereNothingFinishedIsAFailure() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        controller.draft.batchSize = 4

        let final = FakeFixtures.batchStatus([
            .init(1, state: "failed", error: "Ran out of memory."),
            .init(2, state: "failed", error: "The host went away."),
            .init(3, state: "cancelled"),
            .init(4, state: "cancelled"),
        ])
        backend.submitAnswer = final
        backend.batchStatusAnswers[final.id] = final

        controller.submit(on: workstation, backend: backend)
        await controller.runTask?.value

        guard case let .failed(message) = controller.run else {
            Issue.record("expected .failed, got \(controller.run)")
            return
        }
        #expect(message == "Ran out of memory.")
    }

    /// **Fails today**: `settle` forgets the batch the moment the FIRST
    /// child settles, which with four children drops the idempotency fence
    /// while three are still running.
    @Test func theFenceIsHeldUntilTheWholeBatchSettles() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        controller.draft.batchSize = 4

        // Never settles in this test -- only the fence is being pinned.
        let midway = FakeFixtures.batchStatus([
            .init(1, state: "complete", seed: 1),
            .init(2, state: "running"), .init(3, state: "running"), .init(4, state: "running"),
        ])
        backend.submitAnswer = midway
        backend.batchStatusAnswers[midway.id] = midway

        controller.submit(on: workstation, backend: backend)
        await controller.runTask?.value

        let clientBatchId = backend.submittedAdmissions.last?.clientBatchId
        #expect(clientBatchId != nil)
        #expect(PendingBatch.all().keys.contains(clientBatchId ?? ""))

        // This batch deliberately never settles, so nothing else would ever
        // forget it -- clean up rather than leaking it into a real launch's
        // recovery list.
        if let clientBatchId { PendingBatch.forget(clientBatchId) }
    }

    @Test func thePreviewFollowsAChildThatIsStillRunning() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        controller.draft.batchSize = 4

        // Never settles in this test -- only the preview poll is being pinned.
        let midway = FakeFixtures.batchStatus([
            .init(1, jobId: "job-1", state: "complete", seed: 1),
            .init(2, jobId: "job-2", state: "running"),
            .init(3, jobId: "job-3", state: "running"),
            .init(4, jobId: "job-4", state: "running"),
        ])
        backend.submitAnswer = midway
        backend.batchStatusAnswers[midway.id] = midway

        controller.submit(on: workstation, backend: backend)
        // Waits for `follow` to fully return, by which point its `defer`
        // has already cancelled the preview poll -- so what it recorded
        // before that cancellation is exactly what it will ever record.
        await controller.runTask?.value

        #expect(backend.jobPreviewCalls.first == "job-2")

        if let clientBatchId = backend.submittedAdmissions.last?.clientBatchId {
            PendingBatch.forget(clientBatchId)
        }
    }

    // MARK: - Placement

    @Test func aPlacementPreviewOfFourAsksForFourCopiesOfOneOutput() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        controller.draft.batchSize = 4

        controller.refreshPlacement(on: workstation)
        // `PlacementProbe` debounces before it calls out -- a constructor
        // parameter, so this waits on the call rather than on a clock.
        await settle { backend.callCount("placementPreview") == 1 }

        #expect(backend.placementCopiesRequested.last == 4)
    }

    /// The probe is a planning READ: it prices a render, so it carries no
    /// prompt, no media bytes and no filing (finding 02#5).
    @Test func aPlacementPreviewSendsARedactedRequest() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        controller.draft.prompt = "a tin robot"
        controller.draft.tags = ["unannounced"]
        controller.draft.media.sourceImage = "SOURCEBYTES"

        controller.refreshPlacement(on: workstation)
        await settle { backend.callCount("placementPreview") == 1 }

        let sent = backend.placementRequests.last
        #expect(sent?.prompt == "")
        #expect(sent?.sourceImage == "")
        #expect(sent?.tags == nil)
    }
}
