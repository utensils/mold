import Foundation

// The host-wide event stream. One connection carries everything a machine
// does, which is why this exists at all: the alternative is a poll per pane.
extension HTTPBackend {

    /// Everything this machine reports, as it happens.
    ///
    /// Yields only what this app understands. Job lifecycle, chain jobs and
    /// whatever mold adds next come down the same stream, and `MoldEvent`
    /// answers nil for all of them -- which is the contract, not a gap: a
    /// client that treats an unrecognised tag as a failure breaks the first
    /// time a machine is upgraded ahead of it.
    ///
    /// Each frame is a piece of state this client will not be told again, so
    /// a loss is never silent: the buffer keeps the NEWEST
    /// (`StreamBuffering.frames`) and `EventOverflow` announces an overflow
    /// as `.resyncRequired`, which is what the server itself sends when ITS
    /// buffer overruns -- once when it starts and once when it ends, never
    /// once per lost frame.
    public func events() -> AsyncThrowingStream<MoldEvent, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingNewest(EventOverflow.capacity)) { continuation in
            let task = Task {
                // Declared HERE so the producing task is its only owner:
                // nothing else can reach it, so there is nothing to lock.
                var overflow = EventOverflow()
                do {
                    // An idle machine says nothing for hours, and that is the
                    // stream working rather than the stream stuck.
                    for try await frame in stream("/api/events", timeout: 86_400) {
                        if let event = MoldEvent(name: frame.name, data: frame.data) {
                            overflow.send(event, to: continuation)
                        }
                    }
                    // A stream that ENDS mid-burst gets no episode end to
                    // carry a marker, and a reconnect repairs nothing: the
                    // instance id is unchanged, so nothing else would ever
                    // tell the consumer. Yielded last, it is the newest
                    // element and certain to survive -- and a throwing finish
                    // still delivers what is buffered before the error.
                    if overflow.hasUnannouncedLoss { continuation.yield(.resyncRequired) }
                    continuation.finish()
                } catch {
                    if overflow.hasUnannouncedLoss { continuation.yield(.resyncRequired) }
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
