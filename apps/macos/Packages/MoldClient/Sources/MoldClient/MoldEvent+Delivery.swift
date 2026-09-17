import Foundation

extension AsyncThrowingStream<MoldEvent, Error>.Continuation {
    /// Yields `event` -- and when the buffer had to drop one to take it,
    /// follows with `.resyncRequired` rather than letting the loss pass.
    ///
    /// A dropped `gallery_*` or `job_*` frame is a piece of state this client
    /// will never be told again. The stream stays open, so nothing reconnects;
    /// the only other thing that triggers a repair is a CHANGED instance id,
    /// which a burst does not produce. The library is then silently wrong
    /// until the app is restarted. `.resyncRequired` is the server's own word
    /// for exactly this situation -- "the buffer overran and this client
    /// missed deltas; repair from the listings" -- and `HostStore.deliver`
    /// already passes it through to every listener.
    ///
    /// `.bufferingNewest` is what makes the announcement reliable, and it is
    /// also the right policy on its own terms: the element it drops is the
    /// OLDEST, so what survives a burst is the most CURRENT state and the
    /// marker yielded straight after is always kept. `.bufferingOldest` does
    /// the opposite -- it keeps the oldest and refuses everything new once
    /// full -- so both the recent state and the marker would be what is
    /// thrown away.
    func yieldOrResync(_ event: MoldEvent) {
        if case .dropped = yield(event) { yield(.resyncRequired) }
    }
}
