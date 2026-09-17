import Foundation
import MoldClient

/// One re-list per machine at a time, and at most one waiting behind it.
///
/// A `resync_required` marker says "the deltas you missed are gone, read the
/// listing again". The producer can emit several in a burst and every
/// reconnect emits one, so a bare `Task { await relist(host) }` per marker
/// meant K concurrent reads of one machine's whole index -- each assigning
/// `perHost` wholesale, in whatever order they happened to come back, so the
/// rows on screen could end up being the OLDEST answer.
///
/// One in flight and at most one queued is enough by construction. A marker
/// arriving mid-read means "read once more when this one is done", never "read
/// N more times": the repair it asks for is satisfied by any read that STARTS
/// after it, and the queued read does. Markers arriving after that read begins
/// queue the next one in turn, so nothing is ever dropped either.
@MainActor
final class RelistGate {
    private var running: Set<MoldHost.ID> = []
    private var queued: Set<MoldHost.ID> = []

    /// How many reads this gate has actually run, per machine. What a test
    /// asserts on, and the only reason this is not a pure value.
    private(set) var reads: [MoldHost.ID: Int] = [:]

    func run(_ host: MoldHost.ID, read: @MainActor () async -> Void) async {
        guard running.insert(host).inserted else {
            // Somebody is already reading. Ask for ONE more after it rather
            // than joining a queue that grows with the burst.
            queued.insert(host)
            return
        }
        defer { running.remove(host) }
        // The check happens after the read returns, with no suspension between
        // it and the `defer`, so a marker can only ever land either during a
        // read (it is taken by the next turn of this loop) or when the gate is
        // idle (it starts a read of its own).
        repeat {
            queued.remove(host)
            reads[host, default: 0] += 1
            await read()
        } while queued.contains(host)
    }
}
