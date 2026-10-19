import Testing
@testable import MoldClient

struct SyncBatchedValueTests {
    @Test func followsExactlyWhenNotBatching() {
        var value = SyncBatchedValue<Int>()
        for (now, live) in [(0.0, 1), (0.1, 2), (0.2, 3)] {
            #expect(value.read(live, batching: false, now: now) == .init(value: live, dueIn: nil))
        }
    }

    @Test func publishesTheFirstChangeAtOnceThenAtMostOncePerInterval() {
        var value = SyncBatchedValue<Int>(interval: 1)
        #expect(value.read(1, batching: true, now: 10).value == 1)
        #expect(value.read(2, batching: true, now: 12).value == 2, "a quiet second has passed: leading edge")
        let held = value.read(3, batching: true, now: 12.25)
        #expect(held.value == 2)
        #expect(held.dueIn == 0.75)
        #expect(value.read(4, batching: true, now: 12.5).value == 2)
        #expect(value.read(4, batching: true, now: 13).value == 4)
        #expect(value.read(4, batching: true, now: 13.1) == .init(value: 4, dueIn: nil))
    }

    @Test func stoppingFlushesTheLiveValue() {
        var value = SyncBatchedValue<Int>(interval: 1)
        _ = value.read(1, batching: true, now: 0)
        #expect(value.read(2, batching: true, now: 0.5).value == 1)
        #expect(value.read(2, batching: false, now: 0.6) == .init(value: 2, dueIn: nil))
        #expect(value.read(3, batching: true, now: 0.7).value == 2, "restarting within the interval still holds")
    }
}
