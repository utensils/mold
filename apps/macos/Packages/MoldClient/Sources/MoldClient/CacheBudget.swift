import Foundation

/// Which cached files have to go for the rest to fit a budget.
///
/// Arithmetic, kept away from the file system so it can be tested without one.
/// Least-recently-used first, because the thing you looked at longest ago is
/// the thing you are least likely to look at next.
public enum CacheBudget {

    public struct File: Hashable, Sendable {
        public let name: String
        public let bytes: Int
        public let lastUsed: Date

        public init(name: String, bytes: Int, lastUsed: Date) {
            self.name = name
            self.bytes = bytes
            self.lastUsed = lastUsed
        }
    }

    /// The names to delete, oldest first, so that what remains fits `cap`.
    ///
    /// A file bigger than the whole cap goes regardless of how new it is:
    /// keeping it would mean evicting everything else and still not fitting,
    /// which is the worst of both. A cap of zero is a real setting -- someone
    /// with no disk to spare turning the cache off -- and not a disabled one.
    public static func evictions(from files: [File], cap: Int) -> [String] {
        var doomed = files.filter { $0.bytes > cap }.map(\.name)
        var remaining = files.filter { $0.bytes <= cap }.sorted { $0.lastUsed < $1.lastUsed }
        var total = remaining.reduce(0) { $0 + $1.bytes }
        while total > cap, !remaining.isEmpty {
            let oldest = remaining.removeFirst()
            total -= oldest.bytes
            doomed.append(oldest.name)
        }
        return doomed
    }
}
