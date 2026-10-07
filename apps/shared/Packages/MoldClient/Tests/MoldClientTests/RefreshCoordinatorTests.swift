import Testing
@testable import MoldClient

@MainActor
struct RefreshCoordinatorTests {
    @Test func overlappingRefreshesNeverRunConcurrentlyAndRecheckOnce() async {
        let coordinator = RefreshCoordinator()
        var running = 0
        var maximum = 0
        var runs = 0
        var release: CheckedContinuation<Void, Never>?
        let first = Task {
            await coordinator.run("host") {
                running += 1; maximum = max(maximum, running); runs += 1
                if runs == 1 { await withCheckedContinuation { release = $0 } }
                running -= 1
            }
        }
        while release == nil { await Task.yield() }
        let second = Task { await coordinator.run("host") { Issue.record("Must reuse the active operation") } }
        // Let the overlapping request join before releasing the first read.
        while !coordinator.hasPending("host") { await Task.yield() }
        release?.resume()
        await first.value; await second.value
        #expect(maximum == 1)
        #expect(runs == 2)
    }
}
