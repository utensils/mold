import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor
struct PlacementProbeRaceTests {
    @Test func cancelledOldHostErrorCannotReplaceCurrentHostResult() async {
        let old = MoldHost(name: "Old", baseURL: URL(string: "http://old")!)
        let current = MoldHost(name: "Current", baseURL: URL(string: "http://current")!)
        let first = FakeBackend(host: old), second = FakeBackend(host: current)
        var finishOld: CheckedContinuation<PlacementPreview, Error>?
        first.placementResponder = { _ in try await withCheckedThrowingContinuation { finishOld = $0 } }
        second.placementResponder = { _ in throw MoldClientError.unreachable("Current host explanation") }
        let hosts = HostStore(hosts: [old, current]) { $0.id == old.id ? first : second }
        let probe = PlacementProbe(debounce: .zero)
        probe.refresh(draft: RenderDraft(), model: "same-model", on: old, hosts: hosts)
        await settle { finishOld != nil }
        probe.refresh(draft: RenderDraft(), model: "same-model", on: current, hosts: hosts)
        #expect(probe.error == nil)
        await settle { probe.error != nil }
        let currentError = probe.error
        finishOld?.resume(throwing: MoldClientError.unreachable("Old host explanation"))
        await settle { first.placementRequests.count == 1 }
        for _ in 0..<10 { await Task.yield() }
        #expect(probe.error == currentError)
    }
}
