import Foundation

/// Main-actor warm previews can be read during view construction without a
/// placeholder frame; bounded asynchronous work still coalesces cold requests.
@MainActor public final class SourceMediaPreviewCache<Value: Sendable> {
    private struct Hit { let value: Value; let cost: Int; var access: UInt64 }
    private struct Pending { let token: UUID; var waiters: Int }
    private let pipeline: BoundedAsyncCache<SourcePreviewKey, Value>
    private let maximumCost: Int
    private let cost: @Sendable (Value) -> Int
    private var hits: [SourcePreviewKey: Hit] = [:]
    private var pending: [SourcePreviewKey: Pending] = [:]
    private var totalCost = 0
    private var clock: UInt64 = 0

    public init(maximumCost: Int = 32 * 1024 * 1024, concurrency: Int = 4,
                cost: @escaping @Sendable (Value) -> Int) {
        self.maximumCost = max(0, maximumCost); self.cost = cost
        // The synchronous front cache owns the only retained-pixel budget.
        // A positive cost guarantees that the zero-budget pipeline keeps no hits.
        pipeline = BoundedAsyncCache(maximumCost: 0, concurrency: concurrency, cost: { _ in 1 })
    }

    public func peek(for key: SourcePreviewKey) -> Value? {
        guard var hit = hits[key] else { return nil }
        clock &+= 1; hit.access = clock; hits[key] = hit
        return hit.value
    }

    public func value(for key: SourcePreviewKey, load: @escaping @Sendable () async throws -> Value) async -> Value? {
        guard !Task.isCancelled else { return nil }
        if let hit = peek(for: key) { return hit }
        var flight = pending[key] ?? Pending(token: UUID(), waiters: 0)
        flight.waiters += 1; pending[key] = flight
        let token = flight.token
        defer { release(key, token: token) }
        let value = await pipeline.value(for: key, load: load)
        guard !Task.isCancelled, pending[key]?.token == token, let value else { return nil }
        store(value, for: key)
        return value
    }

    private func release(_ key: SourcePreviewKey, token: UUID) {
        guard var flight = pending[key], flight.token == token else { return }
        flight.waiters -= 1
        pending[key] = flight.waiters == 0 ? nil : flight
    }

    private func store(_ value: Value, for key: SourcePreviewKey) {
        let bytes = max(0, cost(value))
        guard bytes <= maximumCost else { return }
        if let old = hits.removeValue(forKey: key) { totalCost -= old.cost }
        while bytes > maximumCost - totalCost,
              let oldest = hits.min(by: { $0.value.access < $1.value.access })?.key {
            totalCost -= hits.removeValue(forKey: oldest)!.cost
        }
        clock &+= 1
        hits[key] = Hit(value: value, cost: bytes, access: clock); totalCost += bytes
    }

    public func invalidate(where invalid: @escaping @Sendable (SourcePreviewKey) -> Bool) async {
        // Fence results synchronously before awaiting cancellation of cold work.
        // A noncooperative old loader cannot repopulate the warm cache afterward.
        for key in Array(hits.keys) where invalid(key) { totalCost -= hits.removeValue(forKey: key)!.cost }
        for key in Array(pending.keys) where invalid(key) { pending[key] = nil }
        await pipeline.invalidate(where: invalid)
    }
}
