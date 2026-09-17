import Foundation
import MoldClient

// The whole-queue gate's fixtures. A separate file rather than two more
// parameters on `FakeFixtures.serverStatus` / `capabilities`, which several
// lanes are editing at once.
extension FakeFixtures {
    /// A machine's status with the gate's own field. `nil` omits the key,
    /// which is what a machine that predates it sends.
    static func serverStatus(queuePaused: Bool?) -> ServerStatus {
        let json = """
        {"version": "0.29.0", "hostname": "fake", "busy": false, "uptime_secs": 0,
         "queue_paused": \(queuePaused.map { "\($0)" } ?? "null")}
        """
        return try! MoldJSON.decoder.decode(ServerStatus.self, from: Data(json.utf8))
    }

    /// `capabilities.queue.can_pause`. `false` here and an absent block both
    /// mean no gate; this plants the explicit answer.
    ///
    /// `events.available` rides along because a machine that cannot stream is
    /// never watched, and a test that emits a frame at one would settle on
    /// nothing -- silently, since `settle` is bounded rather than assertive.
    static func capabilities(canPauseQueue: Bool, events: Bool = true) -> Capabilities {
        let json = #"""
        {"queue": {"can_pause": \#(canPauseQueue)}, "events": {"available": \#(events)}}
        """#
        return try! MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }
}
