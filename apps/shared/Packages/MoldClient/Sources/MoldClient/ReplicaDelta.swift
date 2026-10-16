import Foundation

/// What one `GalleryReplica.pull()` changed, in the shape every caller reads
/// whether the machine has a change feed or only the full listing.
///
/// `reset` means "replace what you show for `scope` with `upserted`": the
/// first delta a process receives, a machine that changed instance or epoch,
/// or a caller that asked for a full repair. Otherwise `upserted` is every row
/// whose value changed (live and trash together -- `trashedAt` says which) and
/// `removed` every name that left `scope`.
public struct ReplicaDelta: Sendable, Equatable {
    /// The two partitions a machine's gallery is split into on screen.
    public struct Scope: OptionSet, Sendable, Hashable {
        public let rawValue: Int
        public init(rawValue: Int) { self.rawValue = rawValue }
        public static let live = Scope(rawValue: 1)
        public static let trash = Scope(rawValue: 2)
        public static let all: Scope = [.live, .trash]

        func covers(_ print: GalleryPrint) -> Bool {
            contains(print.trashedAt == nil ? .live : .trash)
        }
    }

    public let upserted: [GalleryPrint]
    public let removed: [String]
    public let reset: Bool
    /// Increments once per delta the replica hands out, so a consumer can key
    /// derived state on it instead of comparing rows.
    public let revision: Int
    /// What this delta speaks for. The feed always answers for both; a legacy
    /// pull answers for the listings it actually read.
    public let scope: Scope

    public init(upserted: [GalleryPrint], removed: [String], reset: Bool, revision: Int,
                scope: Scope = .all) {
        self.upserted = upserted
        self.removed = removed
        self.reset = reset
        self.revision = revision
        self.scope = scope
    }

    /// Changed rows that belong in the live timeline.
    public var live: [GalleryPrint] { upserted.filter { $0.trashedAt == nil } }
    /// Changed rows that belong in Recently Deleted.
    public var trashed: [GalleryPrint] { upserted.filter { $0.trashedAt != nil } }
}

public extension GalleryReplica {
    /// What a caller wants from a pull. Concurrent requests merge into one.
    struct Pull: Sendable, Equatable {
        /// The machine advertises `gallery.changes_feed`. Even then a 404 or a
        /// `503 GALLERY_FEED_NOT_READY` falls back to the listing this time.
        public var useFeed: Bool
        /// Which legacy listings to read. The feed always covers both.
        public var scope: ReplicaDelta.Scope
        /// Deliver the whole state as a reset even if nothing changed: the
        /// repair after an optimistic edit the machine refused.
        public var full: Bool

        public init(useFeed: Bool, scope: ReplicaDelta.Scope = .all, full: Bool = false) {
            self.useFeed = useFeed
            self.scope = scope
            self.full = full
        }

        public static let feed = Pull(useFeed: true)

        public static func legacy(_ scope: ReplicaDelta.Scope, full: Bool = false) -> Pull {
            Pull(useFeed: false, scope: scope, full: full)
        }

        func merged(with other: Pull) -> Pull {
            Pull(useFeed: useFeed || other.useFeed, scope: scope.union(other.scope),
                 full: full || other.full)
        }
    }
}
