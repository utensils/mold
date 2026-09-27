import Foundation
import Testing

@testable import MoldClient

// What happens to a state frame the client cannot keep up with.

private func added(_ n: Int) -> MoldEvent { .gallery(.added(filename: "\(n).png", row: nil)) }

private func drain(_ stream: AsyncThrowingStream<MoldEvent, Error>) async throws -> [MoldEvent] {
    var received: [MoldEvent] = []
    for try await event in stream { received.append(event) }
    return received
}

/// A bounded stream and the guard that feeds it, sized together the way
/// `events()` sizes them.
private func bounded(capacity: Int)
    -> (AsyncThrowingStream<MoldEvent, Error>, AsyncThrowingStream<MoldEvent, Error>.Continuation,
        EventOverflow) {
    let (stream, continuation) = AsyncThrowingStream.makeStream(
        of: MoldEvent.self, bufferingPolicy: .bufferingNewest(capacity))
    return (stream, continuation, EventOverflow(capacity: capacity))
}

/// A burst bigger than the buffer is told about, once.
///
/// **Fails today**: the rule is stateless -- one `.resyncRequired` per
/// DROPPED FRAME -- and under `.bufferingNewest` every yield reports
/// `.dropped` once the buffer is full. A sustained burst (an `emptyTrash`, a
/// 64-child batch settling) therefore produces a marker per frame: each one
/// takes a slot and evicts another real frame, and every consumer fires a
/// re-read per marker, so the repair path amplifies the overload it exists to
/// repair.
@Test func aSustainedBurstIsAnnouncedOnceRatherThanThousandsOfTimes() async throws {
    let (stream, continuation, guard_) = bounded(capacity: 8)
    var overflow = guard_
    for n in 0 ..< 80 { overflow.send(added(n), to: continuation) }
    // What `events()` does when its stream ends: the opening marker was
    // itself evicted by the 70 frames after it, and a burst that is still
    // running gets no episode end, so the loss is announced here or never.
    if overflow.hasUnannouncedLoss { continuation.yield(.resyncRequired) }
    continuation.finish()

    let received = try await drain(stream)
    let markers = received.filter { $0 == .resyncRequired }.count
    #expect(markers <= 2, "one opening marker, at most one closing one")
    #expect(markers >= 1, "a loss is never silent")
}

/// A burst that ended cleanly with nothing lost says nothing at the end
/// either -- `hasUnannouncedLoss` is about a LOSS, not about having been
/// busy.
@Test func aStreamThatLostNothingEndsWithNothingToSay() async throws {
    let (stream, continuation, guard_) = bounded(capacity: 8)
    var overflow = guard_
    for n in 0 ..< 8 { overflow.send(added(n), to: continuation) }
    #expect(!overflow.hasUnannouncedLoss)
    continuation.finish()
    #expect(try await drain(stream) == (0 ..< 8).map(added))
}

/// The whole point of bounding it: the frames a marker would have evicted
/// stay, so the state a consumer reconciles against is still current.
@Test func theMarkersDoNotEvictTheStateTheyAskYouToReconcile() async throws {
    let (stream, continuation, guard_) = bounded(capacity: 8)
    var overflow = guard_
    for n in 0 ..< 80 { overflow.send(added(n), to: continuation) }
    if overflow.hasUnannouncedLoss { continuation.yield(.resyncRequired) }
    continuation.finish()

    let received = try await drain(stream)
    #expect(received.contains(added(79)), "the newest state survived")
    #expect(received.filter { $0 != .resyncRequired }.count >= 7,
            "the buffer is state, not markers")
}

/// An episode ENDS when the consumer has genuinely caught up, and a frame
/// dropped after the first marker was already acted on has to be announced
/// again -- otherwise it is lost for good.
@Test func aDropAfterTheFirstMarkerWasActedOnIsAnnouncedAgain() async throws {
    let (stream, continuation, guard_) = bounded(capacity: 4)
    var overflow = guard_
    var iterator = stream.makeAsyncIterator()

    // Overflow by two: the opening marker is emitted on the first drop and,
    // being the newest, survives the one drop after it.
    for n in 0 ..< 6 { overflow.send(added(n), to: continuation) }

    var seen: [MoldEvent] = []
    for _ in 0 ..< 4 { if let event = try await iterator.next() { seen.append(event) } }
    #expect(seen.contains(.resyncRequired), "the consumer was told, and has now acted")
    #expect(seen.filter { $0 == .resyncRequired }.count == 1)

    // The frame dropped AFTER that marker arrived after the consumer read,
    // so the first marker does not cover it. The next yield finds the buffer
    // drained -- the episode is over -- and closes it with the second.
    overflow.send(added(6), to: continuation)
    continuation.finish()
    while let event = try await iterator.next() { seen.append(event) }

    #expect(seen.filter { $0 == .resyncRequired }.count == 2)
    #expect(seen.contains(added(6)))
    #expect(!overflow.hasUnannouncedLoss, "the episode is closed, not still open")
}

/// A stream that keeps up says nothing extra: this is a loss signal, not a
/// heartbeat, and a spurious one costs a full listing fetch per host.
@Test func aStreamWithinItsBufferNeverAsksForAResync() async throws {
    let (stream, continuation, guard_) = bounded(capacity: 8)
    var overflow = guard_
    for n in 0 ..< 8 { overflow.send(added(n), to: continuation) }
    continuation.finish()

    #expect(try await drain(stream) == (0 ..< 8).map(added))
}

/// The closing marker is emitted where there IS headroom, so it is never
/// itself the thing evicted -- which is why the episode ends on real
/// headroom rather than on the first successful enqueue.
@Test func theClosingMarkerIsNeverTheFrameThatGetsEvicted() async throws {
    let (stream, continuation, guard_) = bounded(capacity: 4)
    var overflow = guard_
    var iterator = stream.makeAsyncIterator()
    for n in 0 ..< 20 { overflow.send(added(n), to: continuation) }

    // Drain the whole buffer, so the episode ends on real headroom.
    var seen: [MoldEvent] = []
    for _ in 0 ..< 4 { if let event = try await iterator.next() { seen.append(event) } }

    // The closing marker goes in where there is room, never by evicting the
    // very state it is asking the consumer to reconcile against.
    overflow.send(added(200), to: continuation)
    continuation.finish()
    while let event = try await iterator.next() { seen.append(event) }

    #expect(seen.contains(.resyncRequired))
    #expect(seen.contains(added(200)), "the frame that closed the episode is still there")
    #expect(seen.filter { $0 == .resyncRequired }.count <= 2)
}

/// An unbounded stream cannot drop, so it never announces one.
@Test func anUnboundedStreamNeverAnnouncesALoss() async throws {
    let (stream, continuation) = AsyncThrowingStream.makeStream(
        of: MoldEvent.self, bufferingPolicy: .unbounded)
    var overflow = EventOverflow(capacity: StreamBuffering.frames)
    for n in 0 ..< 1_000 { overflow.send(added(n), to: continuation) }
    continuation.finish()

    let received = try await drain(stream)
    #expect(received.count == 1_000)
    #expect(!received.contains(.resyncRequired))
}

/// The guard and the stream are sized from the SAME number: a guard that
/// thinks the buffer is bigger than it is never sees the episode end.
@Test func theGuardIsSizedFromTheStreamsOwnCapacity() {
    #expect(EventOverflow.capacity == StreamBuffering.frames)
}
