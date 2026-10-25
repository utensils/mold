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

// A Secure machine's requests ride a WebSocket. A dropped socket throws a
// POSIX `NSError`, not a `URLError`: it must still read as the machine being
// unreachable (greyed out), never as "That action failed" in a banner.
@Test func aDroppedSecureSocketMeansUnreachable() {
    for error: any Error in [POSIXError(.ENOTCONN), NSError(domain: NSPOSIXErrorDomain, code: 61),
                             SecureConnectionError.closed, URLError(.notConnectedToInternet)] {
        guard case MoldClientError.unreachable = TransportFailure.from(any: error) else {
            Issue.record("expected .unreachable for \(error)")
            continue
        }
    }
}

@Test func repliesAndOwnErrorsPassThroughUntouched() {
    #expect(TransportFailure.from(any: CancellationError()) is CancellationError)
    #expect(TransportFailure.from(any: URLError(.cancelled)) is CancellationError)
    guard case MoldClientError.unauthorized = TransportFailure.from(any: MoldClientError.unauthorized) else {
        Issue.record("a refusal must stay a refusal"); return
    }
    #expect(TransportFailure.from(any: SecureConnectionError.authenticationFailed) as? SecureConnectionError == .authenticationFailed)
    #expect(TransportFailure.from(any: CocoaError(.fileReadNoSuchFile)) is CocoaError)
}
