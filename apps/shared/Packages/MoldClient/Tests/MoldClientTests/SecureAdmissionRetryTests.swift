import Foundation
import Testing
@testable import MoldClient

struct SecureAdmissionRetryTests {
    private actor Attempts {
        var count = 0
        func next() -> Int { count += 1; return count }
    }

    @Test func retriesOnlyPreAdmissionBusyAndCapsAtThreeFreshAttempts() async throws {
        let attempts = Attempts()
        let result = try await SecureAdmissionRetry.open(timeout: 2) { _ in
            let count = await attempts.next()
            if count < 3 { throw SecureAdmissionRetry.ProxyBusy() }
            return count
        }
        #expect(result == 3)
        let exhausted = Attempts()
        await #expect(throws: SecureAdmissionRetry.ProxyBusy.self) {
            try await SecureAdmissionRetry.open(timeout: 2) { _ -> Int in
                _ = await exhausted.next()
                throw SecureAdmissionRetry.ProxyBusy()
            }
        }
        #expect(await exhausted.count == 3)
    }

    @Test func authenticationProtocolAndDispatchedFailuresNeverRetry() async {
        for failure in [SecureConnectionError.authenticationFailed, .protocolViolation, .closed] {
            let attempts = Attempts()
            await #expect(throws: failure) {
                try await SecureAdmissionRetry.open(timeout: 2) { _ -> Int in
                    _ = await attempts.next()
                    throw failure
                }
            }
            #expect(await attempts.count == 1)
        }
    }

    @Test func cancellationAndExistingDeadlineBoundBackoff() async throws {
        let attempts = Attempts()
        let task = Task {
            try await SecureAdmissionRetry.open(timeout: 2) { _ -> Int in
                _ = await attempts.next()
                throw SecureAdmissionRetry.ProxyBusy()
            }
        }
        while await attempts.count == 0 { await Task.yield() }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(await attempts.count == 1)
        let short = Attempts()
        await #expect(throws: URLError.self) {
            try await SecureAdmissionRetry.open(timeout: 0.01) { _ -> Int in
                _ = await short.next()
                throw SecureAdmissionRetry.ProxyBusy()
            }
        }
        #expect(await short.count == 1)
    }
}
