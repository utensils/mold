import Foundation

/// Hands `/api/events` frames to a bounded stream, and says when one was
/// lost -- once per EPISODE, not once per frame.
///
/// A dropped `gallery_*` or `job_*` frame is a piece of state this client
/// will never be told again. The stream stays open, so nothing reconnects;
/// the only other thing that triggers a repair is a CHANGED instance id,
/// which a burst does not produce. The library is then silently wrong until
/// the app is restarted. `.resyncRequired` is the server's own word for
/// exactly this -- "the buffer overran and this client missed deltas; repair
/// from the listings" -- and `HostStore.deliver` already passes it to every
/// listener.
///
/// Announcing each drop on its own is what made it worse than silence.
/// Under `.bufferingNewest` EVERY yield reports `.dropped` once the buffer is
/// full, so a sustained burst -- an `emptyTrash`, a 64-child batch settling --
/// produced a marker per frame: each marker took a slot and evicted another
/// real frame, and each one made every consumer fire its own re-read. The
/// repair path amplified the overload it exists to repair.
///
/// So an episode gets at most TWO markers, and that is enough to converge:
/// one when it starts, so a consumer reading along can begin repairing; and
/// one when it ends, if anything was dropped after the first -- because those
/// frames arrived after the consumer had already read, and without a second
/// marker they would be lost for good.
///
/// The OPENING marker is best-effort and the closing one is the guarantee.
/// `.bufferingNewest` evicts the oldest, so a marker is only safe while it is
/// the newest thing there: a burst that keeps going pushes the opening marker
/// out along with the frames around it. That is fine while the stream lives,
/// because the episode's end always carries one where there is headroom -- and
/// it is why `hasUnannouncedLoss` exists for the case where the stream ENDS
/// mid-burst instead, which is a server closing the connection under load. A
/// reconnect does not repair it: the instance id is unchanged, so nothing else
/// would ever tell the consumer.
///
/// **Used from the producing task only.** `events()` declares it inside its
/// own `Task`, so there is one owner and nothing to synchronise; it is a
/// `struct` so that cannot quietly stop being true.
struct EventOverflow {
    /// The capacity the stream is built with, so the two can never disagree
    /// about when the buffer is full (`theGuardIsSizedFromTheStreamsOwnCapacity`).
    static let capacity = StreamBuffering.frames

    private let endsAt: Int
    private var overflowing = false
    private var droppedSinceMarker = false

    /// - Parameter capacity: the stream's own `bufferingNewest` capacity.
    init(capacity: Int = EventOverflow.capacity) {
        // HYSTERESIS. An episode cannot end on the first successful enqueue:
        // a buffer that is exactly full enqueues with `remaining == 0` and
        // drops on the very next yield, so ending there would flap once per
        // frame and reproduce the marker storm. Half the buffer is the point
        // at which the consumer has demonstrably drained rather than briefly
        // kept up -- and it is never zero, so the closing marker always has
        // somewhere to land instead of evicting the state it asks you to
        // reconcile against.
        endsAt = Swift.max(1, capacity / 2)
    }

    /// Whether the stream would end leaving a loss nobody has been told
    /// about.
    ///
    /// True while an episode is still open -- the opening marker may have
    /// been evicted by the burst that followed it, and there will be no
    /// episode end to carry a replacement -- and true when frames were
    /// dropped after the last marker. The caller yields one final
    /// `.resyncRequired` before finishing, where it is the newest element and
    /// therefore certain to survive.
    var hasUnannouncedLoss: Bool { overflowing || droppedSinceMarker }

    /// Yields `event`, and the marker if this is where one belongs.
    mutating func send(
        _ event: MoldEvent, to continuation: AsyncThrowingStream<MoldEvent, Error>.Continuation
    ) {
        switch continuation.yield(event) {
        case .dropped:
            if overflowing {
                // Already announced. Note it for the closing marker: this
                // frame is after whatever the consumer has read so far.
                droppedSinceMarker = true
            } else {
                overflowing = true
                droppedSinceMarker = false
                continuation.yield(.resyncRequired)
            }
        case let .enqueued(remaining):
            guard overflowing, remaining >= endsAt else { return }
            overflowing = false
            if droppedSinceMarker {
                droppedSinceMarker = false
                continuation.yield(.resyncRequired)
            }
        default:
            // `.terminated`, and whatever the stdlib adds: nothing is being
            // delivered, so there is nobody to tell.
            break
        }
    }
}
