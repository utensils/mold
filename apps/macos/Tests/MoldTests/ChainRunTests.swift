import Foundation
import MoldClient
import Testing

@testable import Mold

/// A clip too long for one denoise, from the press to the print.
/// **Fails today**: the app admitted a batch whatever the length was.
@MainActor
struct ChainRunTests {
    private func machine(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    private func makeController(
        _ backend: FakeBackend, host: MoldHost, reconnect: Duration = .milliseconds(1)
    ) -> GenerateController {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts),
                                            chain: ChainRun(firstBackoff: reconnect))
        controller.modelName = "ltx-2-19b:fp8"
        controller.modelFamily = "ltx2"
        controller.hostID = host.id
        controller.draft.prompt = "a tin robot"
        controller.draft.frames = 249
        return controller
    }

    private func event(_ json: String) -> ChainJobEvent {
        try! MoldJSON.decoder.decode(ChainJobEvent.self, from: Data(json.utf8))
    }

    private let routing = ChainRouting.Decision.chain(
        clipFrames: 97, motionTail: 17, stageCount: 3)

    /// The whole sequence: created as an EPHEMERAL chain job, followed on its
    /// own stream, settling into ONE print -- never a batch.
    @Test func aLongClipBecomesOneEphemeralChainJobAndOnePrint() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.chainJobAnswer = try! MoldJSON.decoder.decode(
            CreateChainJobResponse.self, from: Data(#"{"job_id": "chain-1"}"#.utf8))
        backend.chainEventsHeldOpen.insert("chain-1")

        controller.submit(on: workstation, backend: backend, routing: routing)
        await settle { controller.run.stage == "Clip 1 of 3" }

        // No batch was admitted -- the two doors are different routes and a
        // chain that also submitted one would render the whole thing twice.
        #expect(backend.calls.contains("createChainJob"))
        #expect(backend.calls.contains("submit") == false)
        let body = try! #require(backend.chainJobRequests.first)
        #expect(body.ephemeral)
        #expect(body.totalFrames == 249)
        #expect(body.clipFrames == 97)
        #expect(body.motionTailFrames == 17)
        // A stitched long video is still ONE print: the id it is recovered by
        // is the CHAIN's, and it is remembered before anything is followed.
        #expect(PendingChain.all()["chain-1"] == workstation.id.uuidString)

        backend.emitChainEvent(event(#"{"type":"stage_start","stage_idx":1}"#), for: "chain-1")
        backend.emitChainEvent(
            event(#"{"type":"denoise_step","stage_idx":1,"step":4,"total":8}"#), for: "chain-1")
        // Settled on the STATE the assertion reads, never on the first of two
        // frames landing: `stage_start` alone satisfies the stage label while
        // the step counter is still empty.
        await settle { controller.run.steps?.done == 4 }
        #expect(controller.run.stage == "Clip 2 of 3")

        backend.emitChainEvent(
            event(#"{"type":"finalized","gallery_filename":"long.mp4"}"#), for: "chain-1")
        await settle { if case .finished = controller.run { return true } else { return false } }
        guard case let .finished(outcome, host) = controller.run else {
            Issue.record("the chain did not settle into a print: \(controller.run)")
            return
        }
        #expect(outcome.results.map(\.filename) == ["long.mp4"])
        #expect(host == workstation.id)
        // Settled, so there is nothing left to recover on the next launch.
        #expect(PendingChain.all()["chain-1"] == nil)
    }

    /// A new clip RESETS the step counter. The old one belonged to the clip
    /// before it and would read as progress that had already happened.
    @Test func aNewClipStartsItsStepCounterOver() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.chainJobAnswer = try! MoldJSON.decoder.decode(
            CreateChainJobResponse.self, from: Data(#"{"job_id": "chain-2"}"#.utf8))
        backend.chainEventsHeldOpen.insert("chain-2")

        controller.submit(on: workstation, backend: backend, routing: routing)
        await settle { controller.run.stage == "Clip 1 of 3" }
        backend.emitChainEvent(
            event(#"{"type":"denoise_step","stage_idx":0,"step":7,"total":8}"#), for: "chain-2")
        await settle { controller.run.steps?.done == 7 }
        backend.emitChainEvent(event(#"{"type":"stage_start","stage_idx":1}"#), for: "chain-2")
        await settle { controller.run.stage == "Clip 2 of 3" }
        #expect(controller.run.steps == nil)
    }

    /// Stop goes through the CHAIN's own route. A chain id is not a batch id,
    /// and `DELETE /api/generation-batches/chain-1` would have 404'd while the
    /// GPU kept going.
    @Test func stopCancelsTheChainThroughItsOwnRouteAndNotTheQueue() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.chainJobAnswer = try! MoldJSON.decoder.decode(
            CreateChainJobResponse.self, from: Data(#"{"job_id": "chain-3"}"#.utf8))
        backend.chainEventsHeldOpen.insert("chain-3")

        controller.submit(on: workstation, backend: backend, routing: routing)
        await settle { controller.run.isBusy && backend.calls.contains("chainJobEvents") }

        controller.stop()
        await settle { backend.cancelledChainJobIds == ["chain-3"] }
        #expect(backend.cancelledBatchIds.isEmpty)
        #expect(controller.run.isBusy == false)
        // Cancelled by the user, not lost: nothing to recover on relaunch.
        #expect(PendingChain.all()["chain-3"] == nil)
    }

    /// Stop pressed while the CREATE is still in the air. The task is not
    /// cancelled there -- the POST has very likely already reached the host --
    /// so the landing withdraws the job the host just minted. Without this the
    /// create answered, the follow started, and a render the user had stopped
    /// took the canvas and ran to completion.
    @Test func stopDuringAnUnansweredCreateWithdrawsTheJobTheHostMints() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.chainJobAnswer = try! MoldJSON.decoder.decode(
            CreateChainJobResponse.self, from: Data(#"{"job_id": "chain-4"}"#.utf8))
        backend.holdsChainCreate = true

        controller.submit(on: workstation, backend: backend, routing: routing)
        await settle { backend.chainCreatesWaiting == 1 }
        #expect(controller.run.isBusy)

        controller.stop()
        #expect(controller.run.isBusy == false)
        backend.releaseChainCreate()

        await settle { backend.cancelledChainJobIds == ["chain-4"] }
        // Never followed, and never recoverable -- the user withdrew it.
        #expect(backend.calls.contains("chainJobEvents") == false)
        #expect(PendingChain.all()["chain-4"] == nil)
        #expect(controller.run.isBusy == false)
    }

    /// Generate, Stop, Generate. **Fails today**: `creating` and `withdrawn`
    /// are instance state shared by every start, so the FIRST task's landing
    /// clears the SECOND's flags -- and the second Stop then found nothing to
    /// stop, fell through to the batch path, and the chain the user had just
    /// withdrawn rendered to completion.
    @Test func aStaleStartNeverClobbersTheOneAfterIt() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.chainJobAnswer = try! MoldJSON.decoder.decode(
            CreateChainJobResponse.self, from: Data(#"{"job_id": "chain-5"}"#.utf8))
        backend.holdsChainCreate = true

        // Settled on what a release ACTS ON, never on the call count: the
        // route is recorded before the create suspends, so a count of one can
        // be satisfied with nothing parked on the gate yet.
        controller.submit(on: workstation, backend: backend, routing: routing)
        await settle { backend.chainCreatesWaiting == 1 }
        controller.stop()

        // A second press while the FIRST create is still in the air. Both are
        // now parked, in order, so `releaseChainCreate` is exact.
        backend.holdsChainCreate = true
        controller.submit(on: workstation, backend: backend, routing: routing)
        await settle { backend.chainCreatesWaiting == 2 }
        // The first create lands now, and must touch nothing of the second's.
        backend.releaseChainCreate()
        await settle { backend.cancelledChainJobIds.count == 1 }

        // Stop, aimed at the SECOND chain, which is still unanswered.
        #expect(controller.run.isBusy)
        controller.stop()
        #expect(controller.run.isBusy == false)
        backend.releaseChainCreate()
        await settle { backend.cancelledChainJobIds.count == 2 }
        #expect(backend.calls.contains("chainJobEvents") == false)
    }

    /// A dropped stream is not a settlement. **Fails today**: the catch ran
    /// `settle()`, which FORGOT the job -- so a thirty-second Wi-Fi blip in
    /// the middle of a twelve-clip render discarded the app's only record of a
    /// job that kept burning GPU, and there was no reconnect to replace it.
    @Test func adroppedStreamReconnectsAndKeepsTheJob() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.chainJobAnswer = try! MoldJSON.decoder.decode(
            CreateChainJobResponse.self, from: Data(#"{"job_id": "chain-6"}"#.utf8))
        backend.chainEventsHeldOpen.insert("chain-6")
        backend.chainJobDetails["chain-6"] = try! MoldJSON.decoder.decode(
            ChainJobDetail.self, from: Data(#"""
            {"id": "chain-6", "state": "running", "model": "m", "stage_count": 3,
             "current_stage": 1, "error": null, "finalizes": []}
            """#.utf8))

        controller.submit(on: workstation, backend: backend, routing: routing)
        await settle { backend.calls.contains("chainJobEvents") }

        backend.failChainEvents(for: "chain-6")
        // BOTH facts, because neither implies the other and each is monotonic:
        // the re-read lands before the reconnect, and the reconnect is
        // recorded before anything it carries is applied. Settling on one and
        // asserting the other is the shape that flaked on a slower machine.
        await settle {
            controller.run.stage == "Clip 2 of 3"
                && backend.calls.filter { $0 == "chainJobEvents" }.count == 2
        }

        // Still on the canvas, still recoverable, and the stage counter was
        // re-read from the host rather than invented.
        #expect(controller.run.isBusy)
        #expect(PendingChain.all()["chain-6"] == workstation.id.uuidString)
        #expect(backend.calls.contains("chainJob"))
    }

    /// A refusal is the SERVER's sentence and nothing is submitted at all.
    @Test func aRefusedLengthNeverReachesEitherDoor() async {
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)

        controller.submit(on: workstation, backend: backend, routing: .reject("Nope, too long."))
        await settle { if case .failed = controller.run { return true } else { return false } }
        #expect(controller.run == RunState.failed("Nope, too long."))
        #expect(backend.calls.contains("createChainJob") == false)
        #expect(backend.calls.contains("submit") == false)
    }
}

extension RunState: @retroactive Equatable {
    /// Only the two arms these tests compare. Enough to assert on a failure
    /// sentence without a `guard case` at every site.
    public static func == (lhs: RunState, rhs: RunState) -> Bool {
        switch (lhs, rhs) {
        case let (.failed(left), .failed(right)): left == right
        case (.idle, .idle): true
        default: false
        }
    }
}
