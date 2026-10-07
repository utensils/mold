import Foundation

/// Coalesce overlapping reads, then reconcile once more with current host state.
/// Mutations never pass through this coordinator and are never replayed.
@MainActor
public final class RefreshCoordinator {
    private var tasks: [String: Task<Void, Never>] = [:]
    private var pending = Set<String>()
    public init() {}
    func hasPending(_ key: String) -> Bool { pending.contains(key) }
    public func run(_ key: String, operation: @escaping @MainActor () async -> Void) async {
        if let task = tasks[key] {
            pending.insert(key)
            await task.value
            return
        }
        let task = Task {
            repeat {
                pending.remove(key)
                await operation()
            } while !Task.isCancelled && pending.contains(key)
            tasks.removeValue(forKey: key)
            pending.remove(key)
        }
        tasks[key] = task
        await task.value
    }
    public func cancelAll() {
        for task in tasks.values { task.cancel() }
        pending.removeAll()
    }
}
