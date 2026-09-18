import Foundation
import MoldClient
import Testing

@testable import Mold

/// A chain job outlives a quit. **Fails today**: `PendingChain` was
/// WRITE-ONLY and `ChainRun.reattach` had no caller at all, so a relaunch
/// showed an empty pane while the machine kept rendering, and the id stayed in
/// preferences for the life of the install.
@MainActor
struct ChainRecoveryTests {
    private func machine() -> MoldHost {
        MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
    }

    private func detail(_ json: String) -> ChainJobDetail {
        try! MoldJSON.decoder.decode(ChainJobDetail.self, from: Data(json.utf8))
    }

    private func makeController(_ backend: FakeBackend, host: MoldHost) -> GenerateController {
        let hosts = HostStore(hosts: [host]) { _ in backend }
        return GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts),
                                  chain: ChainRun(firstBackoff: .milliseconds(1)))
    }

    private func clearPending() {
        for (id, _) in PendingChain.all() { PendingChain.forget(id) }
    }

    @Test func aliveJobIsFollowedAgainAfterARelaunch() async {
        clearPending()
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.chainJobDetails["chain-r1"] = detail(#"""
        {"id": "chain-r1", "state": "running", "model": "m", "stage_count": 4,
         "current_stage": 2, "error": null, "finalizes": []}
        """#)
        backend.chainEventsHeldOpen.insert("chain-r1")
        PendingChain.remember("chain-r1", host: workstation.id)

        await controller.recoverPending()
        await settle { backend.calls.contains("chainJobEvents") }

        // The stage count comes from the JOB, never from a routing decision
        // this launch does not have.
        #expect(controller.run.stage == "Clip 3 of 4")
        #expect(PendingChain.all()["chain-r1"] == workstation.id.uuidString)
    }

    /// A job PARKED by a host restart is not over, and says so.
    @Test func aparkedJobIsShownAsPausedAndCanBeResumed() async {
        clearPending()
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.chainJobDetails["chain-r2"] = detail(#"""
        {"id": "chain-r2", "state": "paused", "model": "m", "stage_count": 3,
         "current_stage": 1, "error": null, "finalizes": []}
        """#)
        backend.chainEventsHeldOpen.insert("chain-r2")
        PendingChain.remember("chain-r2", host: workstation.id)

        await controller.recoverPending()
        await settle { controller.chain.active?.isPaused == true }
        #expect(controller.run.stage == "Paused after clip 2 of 3")

        controller.chain.resume(backend: { controller.hosts.backend(for: $0) })
        await settle { backend.resumedChainJobIds == ["chain-r2"] }
        #expect(controller.chain.active?.isPaused == false)
    }

    /// Finished while the app was closed: the record goes, and the print is
    /// SHOWN -- the render happened.
    @Test func ajobThatFinishedWhileClosedIsShownAndForgotten() async {
        clearPending()
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.chainJobDetails["chain-r3"] = detail(#"""
        {"id": "chain-r3", "state": "completed", "model": "m", "stage_count": 3,
         "current_stage": 3, "error": null,
         "finalizes": [{"gallery_filename": "long.mp4"}]}
        """#)
        PendingChain.remember("chain-r3", host: workstation.id)

        await controller.recoverPending()
        #expect(PendingChain.all()["chain-r3"] == nil)
        #expect(backend.calls.contains("chainJobEvents") == false)
        guard case let .finished(outcome, _) = controller.run else {
            Issue.record("the finished chain was not shown: \(controller.run)")
            return
        }
        #expect(outcome.results.map(\.filename) == ["long.mp4"])
    }

    /// A job the host has never heard of is DROPPED -- ids used to accumulate
    /// in preferences for the life of the install.
    @Test func ajobTheHostDoesNotKnowIsDropped() async {
        clearPending()
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.plantedErrors["chainJob"] = MoldClientError.http(
            status: 404, code: nil, message: "no such job")
        PendingChain.remember("chain-r4", host: workstation.id)

        await controller.recoverPending()
        #expect(PendingChain.all()["chain-r4"] == nil)
        #expect(controller.run.isBusy == false)
    }

    /// A bad minute on the network is left for the NEXT launch rather than
    /// guessed to be gone -- `PendingRecovery`'s own rule.
    @Test func atransientFailureKeepsTheRecordForNextTime() async {
        clearPending()
        let workstation = machine()
        let backend = FakeBackend(host: workstation)
        let controller = makeController(backend, host: workstation)
        backend.plantedErrors["chainJob"] = MoldClientError.unreachable("down")
        PendingChain.remember("chain-r5", host: workstation.id)

        await controller.recoverPending()
        #expect(PendingChain.all()["chain-r5"] == workstation.id.uuidString)
        clearPending()
    }
}
