import Foundation

/// A transient thumbnail miss must recover without a scroll/recreated view.
/// Permanent failure stays bounded; cancellation and content changes stop old work.
@MainActor public enum SourcePreviewRetry {
    public static func load<Value>(delay: Duration = .milliseconds(300),
                                    isCurrent: () -> Bool,
                                    attempt: () async -> Value?) async -> Value? {
        for index in 0..<3 {
            guard !Task.isCancelled, isCurrent() else { return nil }
            let value = await attempt()
            guard !Task.isCancelled, isCurrent() else { return nil }
            if let value { return value }
            guard index < 2 else { break }
            do { try await Task.sleep(for: delay) } catch { return nil }
        }
        return nil
    }
}
