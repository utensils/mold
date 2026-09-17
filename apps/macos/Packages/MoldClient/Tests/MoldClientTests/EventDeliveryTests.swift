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

/// **Fails today**: `yieldOrResync` does not exist, and `events()` uses
/// `.bufferingOldest`, which KEEPS THE OLDEST and refuses every new yield
/// once full. A burst -- `emptyTrash`, a bulk import, one `gallery_*` frame
/// per print against a `@MainActor` consumer -- therefore loses the most
/// RECENT state permanently, the stream stays open, and the only other thing
/// that triggers a repair is a CHANGED instance id, which a burst does not
/// produce. The library is then silently wrong until the app is restarted.
@Test func aBurstBiggerThanTheBufferEndsInATellingToResync() async throws {
    let (stream, continuation) = AsyncThrowingStream.makeStream(
        of: MoldEvent.self, bufferingPolicy: .bufferingNewest(4))
    for n in 0..<20 { continuation.yieldOrResync(added(n)) }
    continuation.finish()

    let received = try await drain(stream)
    #expect(received.contains(.resyncRequired), "a loss is announced, never silent")
    // And the buffer kept the NEWEST state rather than the oldest: what a
    // resync then reconciles against is current.
    #expect(received.contains(added(19)))
    #expect(!received.contains(added(0)))
}

/// A stream that keeps up says nothing extra: this is a loss signal, not a
/// heartbeat, and a spurious one costs two full listing fetches per host.
@Test func aStreamWithinItsBufferNeverAsksForAResync() async throws {
    let (stream, continuation) = AsyncThrowingStream.makeStream(
        of: MoldEvent.self, bufferingPolicy: .bufferingNewest(8))
    for n in 0..<8 { continuation.yieldOrResync(added(n)) }
    continuation.finish()

    let received = try await drain(stream)
    #expect(received == (0..<8).map(added))
}

/// The marker itself is never the thing that gets dropped. Under
/// `.bufferingNewest` the element evicted is the OLDEST, so a marker yielded
/// straight after a loss is always kept -- which is the reason the policy is
/// `newest` and not `oldest`.
@Test func theResyncMarkerSurvivesTheBurstThatCausedIt() async throws {
    let (stream, continuation) = AsyncThrowingStream.makeStream(
        of: MoldEvent.self, bufferingPolicy: .bufferingNewest(2))
    for n in 0..<50 { continuation.yieldOrResync(added(n)) }
    continuation.finish()

    let received = try await drain(stream)
    #expect(received.last == .resyncRequired || received.contains(.resyncRequired))
    #expect(received.count <= 2)
}

/// An unbounded stream cannot drop, so it never announces one.
@Test func anUnboundedStreamNeverAnnouncesALoss() async throws {
    let (stream, continuation) = AsyncThrowingStream.makeStream(
        of: MoldEvent.self, bufferingPolicy: .unbounded)
    for n in 0..<1_000 { continuation.yieldOrResync(added(n)) }
    continuation.finish()

    let received = try await drain(stream)
    #expect(received.count == 1_000)
    #expect(!received.contains(.resyncRequired))
}
