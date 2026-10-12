import Foundation

/// Retries a fresh admission only. The caller has not dispatched any HTTP record.
enum SecureAdmissionRetry {
    static func isBusyUpgrade(response: URLResponse?, opened: Bool) -> Bool {
        guard !opened, let response = response as? HTTPURLResponse else { return false }
        return response.statusCode == 429 || response.statusCode == 503
    }
    struct ProxyBusy: LocalizedError, Sendable {
        var errorDescription: String? { "The secure proxy is temporarily busy. Try connecting again." }
    }

    static func open<T: Sendable>(timeout: Double,
                                 attempt: @Sendable (Double) async throws -> T) async throws -> T {
        let clock = ContinuousClock()
        let deadline = clock.now + .seconds(timeout)
        for index in 0..<3 {
            try Task.checkCancellation()
            let remaining = clock.now.duration(to: deadline)
            guard remaining > .zero else { throw URLError(.timedOut) }
            let components = remaining.components
            let seconds = Double(components.seconds) + Double(components.attoseconds) / 1e18
            do { return try await attempt(seconds) }
            catch is ProxyBusy {
                try Task.checkCancellation()
                guard index < 2 else { throw ProxyBusy() }
                let delay = Duration.milliseconds(200 * (index + 1) + Int.random(in: 0..<100))
                guard clock.now + delay < deadline else { throw URLError(.timedOut) }
                try await Task.sleep(for: delay)
            }
        }
        throw ProxyBusy()
    }
}
