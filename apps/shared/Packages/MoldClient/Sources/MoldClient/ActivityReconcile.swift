import Foundation

/// What this app believes one machine is doing, between two answers.
public struct ActivityHostSnapshot: Hashable, Sendable {
    /// The address the last answer came from -- a fence, so a machine that
    /// has been re-pointed at a different box does not keep another one's
    /// work on screen. Never carries the API key.
    public let routeURL: String
    public let instanceId: String?
    public let observedAtUnixMs: Int?
    public let items: [ActiveWorkItem]
    public let unavailableKinds: [String]
    /// The last answer failed. The rows are the last VERIFIED ones: a machine
    /// that cannot be reached is not evidence that its work has vanished.
    public let stale: Bool
    public let error: String?

    public init(routeURL: String, instanceId: String? = nil, observedAtUnixMs: Int? = nil,
                items: [ActiveWorkItem] = [], unavailableKinds: [String] = [],
                stale: Bool = false, error: String? = nil) {
        self.routeURL = routeURL
        self.instanceId = instanceId
        self.observedAtUnixMs = observedAtUnixMs
        self.items = items
        self.unavailableKinds = unavailableKinds
        self.stale = stale
        self.error = error
    }
}

/// The reconciliation rules, ported from `studio/api/activity.ts:139-184` and
/// `:196-229`, which web and desktop both read.
///
/// Pure, so "what happens when the machine goes away" is a test rather than
/// something you unplug a cable to find out.
public enum ActivityReconcile {

    /// A good answer REPLACES wholesale. A failure keeps the last verified
    /// rows and marks them stale -- unless the machine's address moved, in
    /// which case those rows described a different box and are dropped.
    public static func host(
        routeURL: String, previous: ActivityHostSnapshot?,
        result: Result<ActiveWorkSnapshot, Error>
    ) -> ActivityHostSnapshot {
        switch result {
        case let .failure(error):
            let routeChanged = previous != nil && previous?.routeURL != routeURL
            return ActivityHostSnapshot(
                routeURL: routeURL,
                instanceId: previous?.instanceId,
                observedAtUnixMs: previous?.observedAtUnixMs,
                items: routeChanged ? [] : (previous?.items ?? []),
                unavailableKinds: previous?.unavailableKinds ?? [],
                stale: true,
                error: error.activitySentence)
        case let .success(snapshot):
            let unavailable = Set(snapshot.unavailableKinds)
            // Retaining across a different machine would be keeping another
            // box's work: the address AND the instance id both have to match.
            let sameAuthority = previous?.routeURL == routeURL
                && previous?.instanceId == snapshot.instanceId
            let retained = sameAuthority
                ? (previous?.items.filter { unavailable.contains($0.authorityKind) } ?? [])
                : []
            return ActivityHostSnapshot(
                routeURL: routeURL,
                instanceId: snapshot.instanceId,
                observedAtUnixMs: snapshot.observedAtUnixMs,
                items: snapshot.items.filter { !unavailable.contains($0.authorityKind) } + retained,
                unavailableKinds: snapshot.unavailableKinds,
                stale: false,
                error: nil)
        }
    }

    /// Every machine's rows in one list.
    ///
    /// A phase transition must never move a row. Across a fleet an older job
    /// can still be queued on one machine while a newer one is already
    /// running on another, so grouping by phase makes work jump to the top.
    /// SUBMISSION TIME is the only ordering shared by every kind and every
    /// machine, and equal timestamps keep the order the machines sent
    /// (`activity.ts:218-228`).
    public static func merged(
        _ snapshots: [(host: MoldHost.ID, snapshot: ActivityHostSnapshot)]
    ) -> [FleetActiveWork] {
        let rows = snapshots.flatMap { host, snapshot in
            snapshot.items.map { item in
                FleetActiveWork(
                    host: host, item: item,
                    stale: snapshot.stale
                        || snapshot.unavailableKinds.contains(item.authorityKind),
                    unavailableKind: snapshot.unavailableKinds.contains(item.authorityKind))
            }
        }
        return rows.enumerated()
            .sorted { left, right in
                let (a, b) = (left.element.item.createdAtUnixMs, right.element.item.createdAtUnixMs)
                return a == b ? left.offset < right.offset : a > b
            }
            .map(\.element)
    }
}

/// One machine's row, in the fleet's list.
public struct FleetActiveWork: Hashable, Sendable, Identifiable {
    public let host: MoldHost.ID
    public let item: ActiveWorkItem
    /// This row is the last thing the machine said, not the current truth.
    public let stale: Bool
    /// The authority behind THIS kind could not be read on the last answer.
    public let unavailableKind: Bool

    /// Unique across the fleet: two machines can hold the same job id, and
    /// one machine can report the same id under two kinds.
    public var id: String { "\(host):\(item.kind):\(item.id)" }

    /// Public so a surface can build one for a test without going through a
    /// whole reconciliation -- the value is plain data.
    public init(host: MoldHost.ID, item: ActiveWorkItem, stale: Bool,
                unavailableKind: Bool) {
        self.host = host
        self.item = item
        self.stale = stale
        self.unavailableKind = unavailableKind
    }
}

private extension Error {
    /// What the snapshot records about a failure. `MoldClientError` already
    /// knows how to say itself; anything else falls back to its description.
    var activitySentence: String {
        (self as? LocalizedError)?.errorDescription ?? "\(self)"
    }
}
