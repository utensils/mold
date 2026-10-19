import Foundation
import Testing
@testable import MoldClient

private actor WarmPreviewCounter {
    var requests = 0
    func load(_ value: Data) -> Data { requests += 1; return value }
}
private actor WarmPreviewGate {
    var requests = 0
    var continuation: CheckedContinuation<Data, Never>?
    func load() async -> Data {
        requests += 1
        return await withCheckedContinuation { continuation = $0 }
    }
    func open() { continuation?.resume(returning: Data([1, 2, 3])); continuation = nil }
}
@MainActor struct SourceMediaPreviewCacheTests {
    private func key(_ name: String) -> SourcePreviewKey {
        let owner = PrintFixtures.entry(name, host: UUID(uuidString: "11111111-1111-1111-1111-111111111111")!)
        return .init(authority: .init(owner: owner, route: nil, instance: nil),
                     member: .init(memberId: name, role: "source_image", displayName: name, sizeBytes: 3))
    }
    @Test func warmPeekAndAsyncHitDoNotRunAnotherLoad() async {
        let cache = SourceMediaPreviewCache<Data>(maximumCost: 8, cost: { $0.count }), counter = WarmPreviewCounter(), item = key("one")
        #expect(cache.peek(for: item) == nil)
        #expect(await cache.value(for: item) { await counter.load(Data([1, 2, 3])) } == Data([1, 2, 3]))
        #expect(cache.peek(for: item) == Data([1, 2, 3]))
        #expect(await cache.value(for: item) { await counter.load(Data([9])) } == Data([1, 2, 3]))
        #expect(await counter.requests == 1)
    }
    @Test func leastRecentlyUsedPixelsEvictWithinTheSingleBudget() async {
        let cache = SourceMediaPreviewCache<Data>(maximumCost: 6, cost: { $0.count })
        let a = key("a"), b = key("b"), c = key("c")
        _ = await cache.value(for: a) { Data([1, 1, 1]) }
        _ = await cache.value(for: b) { Data([2, 2, 2]) }
        #expect(cache.peek(for: a) != nil)
        _ = await cache.value(for: c) { Data([3, 3, 3]) }
        #expect(cache.peek(for: a) != nil && cache.peek(for: b) == nil && cache.peek(for: c) != nil)
        #expect(await cache.value(for: b) { Data([8, 8, 8]) } == Data([8, 8, 8]))
        #expect(await cache.value(for: key("oversize")) { Data(repeating: 4, count: 7) }?.count == 7)
        #expect(cache.peek(for: key("oversize")) == nil)
    }
    @Test func invalidationRemovesWarmPixelsAndFencesALateNoncooperativeLoad() async {
        let cache = SourceMediaPreviewCache<Data>(cost: { $0.count }), gate = WarmPreviewGate(), item = key("late")
        let pending = Task { await cache.value(for: item) { await gate.load() } }
        while await gate.requests == 0 { await Task.yield() }
        await cache.invalidate { $0 == item }
        #expect(await pending.value == nil)
        await gate.open()
        #expect(cache.peek(for: item) == nil)
        #expect(await cache.value(for: item) { Data([9]) } == Data([9]))
        await cache.invalidate { $0 == item }
        #expect(cache.peek(for: item) == nil)
    }
    @Test func cancelledSubscriberLeavesAnotherSubscriberAndItsWarmHitIntact() async {
        let cache = SourceMediaPreviewCache<Data>(cost: { $0.count }), gate = WarmPreviewGate(), item = key("shared")
        let first = Task { await cache.value(for: item) { await gate.load() } }
        while await gate.requests == 0 { await Task.yield() }
        let second = Task { await cache.value(for: item) { await gate.load() } }
        // Let both main-actor tasks enter their shared pipeline wait.
        try? await Task.sleep(for: .milliseconds(20))
        first.cancel()
        #expect(await first.value == nil)
        await gate.open()
        #expect(await second.value == Data([1, 2, 3]))
        #expect(cache.peek(for: item) == Data([1, 2, 3]))
        #expect(await gate.requests == 1)
    }
}
