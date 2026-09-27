import Foundation
import Testing

@testable import MoldClient

// `send(_:)` and `stream(_:timeout:)` both turn a thrown `URLError` into an
// app-facing error through this one mapping. A cancelled request -- a
// `.task(id:)` re-keying, a view going away mid-request -- is the app
// changing its mind, not the machine failing, and must throw Swift's own
// `CancellationError` rather than `.unreachable`, or a cooperative
// cancellation reads as "this machine can't be reached".

@Test func aCancelledRequestThrowsCancellationNotUnreachable() {
    let error = TransportFailure.from(URLError(.cancelled))
    #expect(error is CancellationError)
}

@Test func everyOtherURLErrorStillMeansUnreachable() {
    let error = TransportFailure.from(URLError(.timedOut))
    guard case let MoldClientError.unreachable(reason) = error else {
        Issue.record("expected .unreachable, got \(error)")
        return
    }
    #expect(!reason.isEmpty)
}
