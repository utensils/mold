import Foundation
import Testing

@testable import MoldClient

// Reading the body that comes with a refusal. Bounded two ways, because a
// non-2xx promises NEITHER that the body is small nor that the connection
// closes.

/// Yields `prefix`, then never finishes -- a refused response whose body
/// arrived but whose connection is held open.
private struct HeldOpen: AsyncSequence, Sendable {
    typealias Element = UInt8
    let prefix: [UInt8]

    struct Iterator: AsyncIteratorProtocol {
        var remaining: ArraySlice<UInt8>
        mutating func next() async throws -> UInt8? {
            if let byte = remaining.first {
                remaining = remaining.dropFirst()
                return byte
            }
            // Never returns, never throws: the socket is open and idle.
            while true { try await Task.sleep(for: .seconds(3600)) }
        }
    }

    func makeAsyncIterator() -> Iterator { Iterator(remaining: prefix[...]) }
}

/// **Fails today**: `RefusalBody.read` is bounded by SIZE and nothing else, so it
/// waits on a connection that is never going to say anything more -- up to
/// the request's own timeout, which on `resourceStream` and `events` is
/// 86,400 seconds. The error path, which already knows the status, hangs
/// behind a body it does not need.
@Test func aBodyOnAHeldOpenConnectionGivesUpAndReportsWhatArrived() async {
    let body = Data(#"{"error":"restarting","code":"SERVER_RESTARTING"}"#.utf8)
    let started = ContinuousClock.now
    let read = await RefusalBody.read(
        HeldOpen(prefix: Array(body)), within: .milliseconds(200))
    let elapsed = ContinuousClock.now - started

    #expect(elapsed < .seconds(2))
    // What arrived before the deadline is what there is to report, and it is
    // enough: the refusal decodes.
    #expect(read == body)
}

/// A body that arrives and ENDS is not waited on at all -- the deadline is a
/// last resort, not a delay.
@Test func aBodyThatEndsIsReadImmediately() async {
    let body = Data(#"{"error":"nope"}"#.utf8)
    let started = ContinuousClock.now
    let read = await RefusalBody.read(Array(body).async, within: .seconds(30))
    #expect(ContinuousClock.now - started < .seconds(1))
    #expect(read == body)
}

/// The size ceiling still holds, and still wins when it is reached first.
@Test func theSizeCeilingStillHolds() async {
    let flood = Array(repeating: UInt8(ascii: "x"), count: RefusalBody.limit * 4)
    let read = await RefusalBody.read(flood.async, within: .seconds(30))
    #expect(read.count == RefusalBody.limit)
}

/// A source that throws hands back whatever it had: the status is already
/// known, and a truncated body simply decodes to nothing.
@Test func aFailedReadReportsWhatItGot() async {
    struct Torn: AsyncSequence, Sendable {
        typealias Element = UInt8
        struct Iterator: AsyncIteratorProtocol {
            var sent = 0
            mutating func next() async throws -> UInt8? {
                guard sent < 3 else { throw MoldClientError.malformedResponse }
                sent += 1
                return UInt8(ascii: "a")
            }
        }
        func makeAsyncIterator() -> Iterator { Iterator() }
    }
    #expect(await RefusalBody.read(Torn(), within: .seconds(30)) == Data("aaa".utf8))
}

/// A plain array as an async byte source, for the cases that just need bytes.
private extension Array where Element == UInt8 {
    var async: AsyncStream<UInt8> {
        AsyncStream(bufferingPolicy: .unbounded) { continuation in
            for byte in self { continuation.yield(byte) }
            continuation.finish()
        }
    }
}
