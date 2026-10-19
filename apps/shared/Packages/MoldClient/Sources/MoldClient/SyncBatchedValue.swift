import Foundation

/// A derived value's key that follows its source exactly, except while a sync
/// runs: then the first change publishes at once and later ones at most once
/// per interval, so a whole-library derivation keyed on it runs once a second
/// rather than once per copied print. Batching off publishes the live value.
public struct SyncBatchedValue<Value: Equatable & Sendable>: Sendable {
    public struct Read: Equatable, Sendable {
        public let value: Value
        /// Seconds until the held live value is due; nil when nothing is held.
        public let dueIn: TimeInterval?
    }

    public let interval: TimeInterval
    private var published: Value?
    private var publishedAt: TimeInterval = -.infinity

    public init(interval: TimeInterval = 1) {
        self.interval = interval
    }

    /// `now` is a monotonic clock in seconds (e.g. `systemUptime`).
    public mutating func read(_ live: Value, batching: Bool, now: TimeInterval) -> Read {
        if let published, published == live { return Read(value: live, dueIn: nil) }
        let elapsed = now - publishedAt
        if let published, batching, elapsed < interval {
            return Read(value: published, dueIn: interval - elapsed)
        }
        published = live
        publishedAt = now
        return Read(value: live, dueIn: nil)
    }
}
