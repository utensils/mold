import Foundation

/// Which live gallery frames are this app's own edit coming back.
///
/// A machine reports every gallery change it makes, including the ones this
/// app asked for, and applying our own change a second time is at best a
/// wasted re-list. The gate used to be "is ANYTHING queued for this machine",
/// which dropped every frame from it -- a `gallery_added` for a render that
/// just landed, another client's `gallery_trashed` -- for as long as one star
/// took to round-trip, and on a flaky host for the whole backoff ladder.
/// Nothing re-listed afterwards, so a lost `gallery_trashed` left a ghost row
/// until the next ⌘R.
///
/// An echo is about a ROW, so the gate is about a row: only a frame naming a
/// filename this app is still sending an edit for is skipped. That leaves one
/// real gap -- another client touching the SAME row inside our window -- and
/// the machine that had a frame skipped is remembered so the drain can re-list
/// it once its chain empties.
public struct GalleryEcho: Sendable {
    private var stale: Set<MoldHost.ID> = []

    public init() {}

    /// The row this change is about, when it names one.
    public static func names(_ change: MoldEvent.Gallery) -> String? {
        switch change {
        case let .added(filename, _), let .updated(filename, _), let .restored(filename, _),
             let .removed(filename), let .trashed(filename):
            filename
        case .collectionsChanged:
            nil
        }
    }

    /// Whether this frame is one of ours, remembering the machine if it is.
    public mutating func isEcho(_ change: MoldEvent.Gallery, on host: MoldHost.ID,
                                pending: [MutationOutbox.Entry]) -> Bool {
        guard let filename = Self.names(change),
              pending.contains(where: { $0.filenames.contains(filename) })
        else { return false }
        stale.insert(host)
        return true
    }

    /// True once for a machine that had a frame skipped, and forgets it.
    public mutating func takeStale(_ host: MoldHost.ID) -> Bool {
        stale.remove(host) != nil
    }
}
