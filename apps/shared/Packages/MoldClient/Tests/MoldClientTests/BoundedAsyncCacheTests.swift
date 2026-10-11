import Foundation
import Testing
@testable import MoldClient

private actor PreviewCounter {
    var requests = 0
    var active = 0
    var peak = 0
    func load() async throws -> Data {
        requests += 1; active += 1; peak = max(peak, active)
        defer { active -= 1 }
        try await Task.sleep(for: .milliseconds(20))
        return Data(repeating: 1, count: 4)
    }
}
private actor PreviewGate {
    var requests = 0
    var waiting: CheckedContinuation<Data, Never>?
    func load() async -> Data {
        requests += 1
        return await withCheckedContinuation { waiting = $0 }
    }
    func release() { waiting?.resume(returning: Data(repeating: 1, count: 4)); waiting = nil }
}
struct BoundedAsyncCacheTests {
    @Test func coalescesCapsEvictsAndInvalidates() async {
        let cache = BoundedAsyncCache<String, Data>(maximumCost: 8, concurrency: 2, cost: { $0.count })
        let counter = PreviewCounter()
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<20 { group.addTask { _ = await cache.value(for: "same") { try await counter.load() } } }
        }
        #expect(await counter.requests == 1)
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<8 { group.addTask { _ = await cache.value(for: "\(i)") { try await counter.load() } } }
        }
        #expect(await counter.peak <= 2)
        _ = await cache.value(for: "same") { try await counter.load() }
        #expect(await counter.requests == 10)
        await cache.invalidate { $0 == "same" }
        _ = await cache.value(for: "same") { try await counter.load() }
        #expect(await counter.requests == 11)
    }
    @Test func oneCancelledSubscriberDoesNotCancelAnother() async {
        let cache = BoundedAsyncCache<String, Data>(maximumCost: 8, concurrency: 1, cost: { $0.count })
        let gate = PreviewGate()
        let first = Task { await cache.value(for: "same") { await gate.load() } }
        let second = Task { await cache.value(for: "same") { await gate.load() } }
        while await cache.subscriberCount < 2 { await Task.yield() }
        while await gate.requests < 1 { await Task.yield() }
        first.cancel()
        #expect(await first.value == nil)
        await gate.release()
        #expect(await second.value?.count == 4)
        #expect(await gate.requests == 1)
    }
}

extension BoundedAsyncCacheTests {
    @Test func revokedInFlightValueCannotReenterCache() async {
        let cache = BoundedAsyncCache<String, Data>(maximumCost: 8, concurrency: 1, cost: { $0.count })
        let counter = PreviewCounter()
        let pending = Task { await cache.value(for: "revoked") { try await counter.load() } }
        try? await Task.sleep(for: .milliseconds(5))
        await cache.invalidate { $0 == "revoked" }
        #expect(await pending.value == nil)
        _ = await cache.value(for: "revoked") { try await counter.load() }
        #expect(await counter.requests == 2)
    }
}

@Test func expiredPreviewBytesAreReloadedForUnchangedLegacyMember() async {
    let cache = BoundedAsyncCache<String, Int>(maximumCost: 8, concurrency: 1, maximumAge: 0, cost: { _ in 1 })
    #expect(await cache.value(for: "legacy") { 1 } == 1)
    #expect(await cache.value(for: "legacy") { 2 } == 2)
}
