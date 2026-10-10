import Foundation

/// Transient, cost-bounded work sharing. Every waiter owns its cancellation;
/// only the final waiter cancels work, and distinct misses share a fixed limit.
public actor BoundedAsyncCache<Key: Hashable & Sendable, Value: Sendable> {
    private struct Hit { let value: Value; let cost: Int; var access: UInt64; let fetched: Date }
    private struct Job {
        let token: UUID
        let load: @Sendable () async throws -> Value
        var waiters: [UUID: CheckedContinuation<Value?, Never>]
    }
    private let maximumCost: Int
    private let maximumAge: TimeInterval
    private let concurrency: Int
    private let cost: @Sendable (Value) -> Int
    private var hits: [Key: Hit] = [:]
    private var jobs: [Key: Job] = [:]
    private var queue: [Key] = []
    private var running: [UUID: Task<Void, Never>] = [:]
    private var totalCost = 0
    private var clock: UInt64 = 0
    public init(maximumCost: Int, concurrency: Int, maximumAge: TimeInterval = .infinity, cost: @escaping @Sendable (Value) -> Int) {
        self.maximumAge = maximumAge
        self.maximumCost = maximumCost; self.concurrency = max(1, concurrency); self.cost = cost
    }
    public func value(for key: Key, load: @escaping @Sendable () async throws -> Value) async -> Value? {
        guard !Task.isCancelled else { return nil }
        if let hit = hits[key], Date().timeIntervalSince(hit.fetched) >= maximumAge {
            totalCost -= hit.cost; hits[key] = nil
        }
        if var hit = hits[key] { clock &+= 1; hit.access = clock; hits[key] = hit; return hit.value }
        let waiter = UUID()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                if Task.isCancelled { continuation.resume(returning: nil); return }
                if jobs[key] != nil { jobs[key]!.waiters[waiter] = continuation }
                else {
                    jobs[key] = Job(token: UUID(), load: load, waiters: [waiter: continuation])
                    queue.append(key)
                }
                pump()
            }
        } onCancel: { Task { await self.cancel(key: key, waiter: waiter) } }
    }
    public func invalidate(where invalid: @Sendable (Key) -> Bool) {
        for key in Array(hits.keys) where invalid(key) { totalCost -= hits.removeValue(forKey: key)!.cost }
        for key in Array(jobs.keys) where invalid(key) {
            guard let job = jobs.removeValue(forKey: key) else { continue }
            for waiter in job.waiters.values { waiter.resume(returning: nil) }
            running[job.token]?.cancel()
        }
        queue.removeAll(where: invalid)
    }
    private func cancel(key: Key, waiter: UUID) {
        guard var job = jobs[key], let continuation = job.waiters.removeValue(forKey: waiter) else { return }
        continuation.resume(returning: nil)
        if job.waiters.isEmpty {
            jobs[key] = nil; queue.removeAll { $0 == key }; running[job.token]?.cancel()
        } else { jobs[key] = job }
    }
    private func pump() {
        while running.count < concurrency, !queue.isEmpty {
            let key = queue.removeFirst()
            guard let job = jobs[key] else { continue }
            running[job.token] = Task {
                let result = try? await job.load()
                self.finish(key: key, token: job.token, value: Task.isCancelled ? nil : result)
            }
        }
    }
    private func finish(key: Key, token: UUID, value: Value?) {
        running[token] = nil
        if let job = jobs[key], job.token == token {
            jobs[key] = nil
            if let value {
                let bytes = max(0, cost(value))
                if bytes <= maximumCost {
                    while totalCost + bytes > maximumCost, let oldest = hits.min(by: { $0.value.access < $1.value.access })?.key {
                        totalCost -= hits.removeValue(forKey: oldest)!.cost
                    }
                    clock &+= 1; hits[key] = Hit(value: value, cost: bytes, access: clock, fetched: Date()); totalCost += bytes
                }
            }
            for waiter in job.waiters.values { waiter.resume(returning: value) }
        }
        pump()
    }
}

public struct SourcePreviewKey: Hashable, Sendable {
    public let authority: SourceMediaAuthority
    public let member: RetainedSourceMedia.Member
    public init(authority: SourceMediaAuthority, member: RetainedSourceMedia.Member) {
        self.authority = authority; self.member = member
    }
}
