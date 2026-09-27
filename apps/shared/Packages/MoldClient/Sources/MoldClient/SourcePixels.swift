import Foundation

/// A picked picture's own dimensions.
///
/// A named pair rather than a tuple because it is STORED on the draft: a
/// tuple is not `Hashable`, and the draft is compared whole (the persistence
/// watcher, `.task(id:)` keys) on every change.
public struct SourcePixels: Hashable, Sendable {
    public let width: Int
    public let height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }
}
