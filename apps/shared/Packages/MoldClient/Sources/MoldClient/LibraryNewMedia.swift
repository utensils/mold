import Foundation

/// Session-only Library visits, matching the desktop client's filename baseline.
/// A visit keeps its own snapshot so marking loaded rows seen never removes its badges.
public struct LibraryNewMedia {
    private var visited = false
    private var seen: Set<String> = []

    public init() {}

    public struct Visit: Sendable {
        private let visited: Bool
        private let seen: Set<String>
        fileprivate init(visited: Bool, seen: Set<String>) {
            self.visited = visited
            self.seen = seen
        }
        public func contains(_ filename: String) -> Bool { visited && !seen.contains(filename) }
    }

    public mutating func beginVisit() -> Visit {
        Visit(visited: visited, seen: seen)
    }

    /// Only called while Library is open, using the whole active pool rather than a filter.
    public mutating func markSeen(_ filenames: [String]) {
        seen.formUnion(filenames)
        visited = true
    }
}
