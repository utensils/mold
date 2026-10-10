import Foundation

/// Geometry updates are observations, not view state invalidations. Freeze the
/// exact viewport while navigation covers it, including a partial tile/header.
@MainActor public final class LibraryViewport {
    public init() {}
    public func reset() { offset = 0; covered = false }
    private var offset: CGFloat = 0
    private var covered = false
    public func report(offset: CGFloat) { if !covered { self.offset = offset } }
    public func cover() { covered = true }
    public func uncover() -> CGFloat { covered = false; return offset }
}
