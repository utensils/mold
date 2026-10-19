import Testing
@testable import Mold

struct PerformancePolicyTests {
    @Test func explicitEnvironmentOverridesNightly() {
        #expect(PerformancePolicy.enabled(environment: [:], version: "0.36.1-nightly.1"))
        #expect(!PerformancePolicy.enabled(environment: [:], version: "0.36.1"))
        #expect(PerformancePolicy.enabled(environment: ["MOLD_PERFORMANCE_DIAGNOSTICS": "1"], version: "0.36.1"))
        #expect(!PerformancePolicy.enabled(environment: ["MOLD_PERFORMANCE_DIAGNOSTICS": "0"], version: "nightly"))
    }

    @Test func slowEventsAreThresholdedAndRateLimitedIndependently() {
        var policy = PerformancePolicy()
        let result1 = policy.shouldReport(.libraryCounts, seconds: 0.05, now: 0)
        #expect(!result1)
        let result2 = policy.shouldReport(.libraryCounts, seconds: 0.1, now: 1)
        #expect(result2)
        let result3 = policy.shouldReport(.libraryCounts, seconds: 1, now: 2)
        #expect(!result3)
        let result4 = policy.shouldReport(.sourceInventory, seconds: 1, now: 2)
        #expect(result4)
        let result5 = policy.shouldReport(.libraryCounts, seconds: 1, now: 31)
        #expect(result5)
    }

    @Test func heartbeatDoesNotQueueMoreWorkWhileMainThreadIsBlocked() {
        var probe = PerformanceHeartbeat()
        #expect(probe.tick(now: 1) == .enqueue)
        #expect(probe.tick(now: 2) == .none)
        #expect(probe.tick(now: 3) == .stalled(seconds: 2))
        #expect(probe.tick(now: 4) == .none)
        for second in 5..<33 { _ = probe.tick(now: Double(second)) }
        #expect(probe.tick(now: 33) == .stalled(seconds: 32))
        probe.acknowledge()
        #expect(probe.tick(now: 34) == .enqueue)
    }

    @Test func sleepAndSchedulingGapsDoNotMasqueradeAsHangs() {
        var probe = PerformanceHeartbeat()
        #expect(probe.tick(now: 1) == .enqueue)
        #expect(probe.tick(now: 100) == .none)
        #expect(probe.tick(now: 101) == .none)
        probe.acknowledge()
        #expect(probe.tick(now: 102) == .enqueue)
    }
}
