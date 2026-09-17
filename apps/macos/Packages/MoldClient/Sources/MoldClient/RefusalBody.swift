import Foundation

/// How much of a refused answer's body is worth reading, and what a body that
/// is not mold's envelope is worth saying.
enum RefusalBody {
    /// How much of a refused stream's body is worth reading.
    ///
    /// A refusal body is mold's small `APIError` envelope. Nothing about a
    /// non-2xx promises the other side closes the connection, though, so this
    /// is a ceiling and not an expectation -- 8 KiB is an order of magnitude
    /// more than the largest licence refusal and still nothing to hold.
    static let limit = 8 * 1024

    /// How long a refused response gets to finish saying why.
    ///
    /// The size ceiling is not enough on its own: a non-2xx promises neither
    /// that the body is small nor that the connection CLOSES, and the error
    /// path already knows the status, so waiting on a held-open socket for a
    /// sentence it does not need was a hang -- up to the request's own
    /// timeout, which on `events` and `resourceStream` is 86,400 seconds.
    static let deadline: Duration = .seconds(3)

    /// At most `limit` bytes of a refused response, and at most `within`
    /// waiting for them.
    ///
    /// Whatever arrived before the deadline or a read failure is what there
    /// is to report: the status is already known, and a truncated body simply
    /// decodes to nothing, which is the same answer as no body at all.
    static func read(
        _ bytes: some AsyncSequence<UInt8, some Error> & Sendable,
        within deadline: Duration = Self.deadline
    ) async -> Data {
        // A box rather than a return value: the racing read may be CANCELLED
        // part way, and what it had by then is still the best answer there is.
        let read = PartialBody()
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                do {
                    for try await byte in bytes {
                        if read.append(byte) >= Self.limit { return }
                    }
                } catch {}
            }
            group.addTask { try? await Task.sleep(for: deadline) }
            await group.next()
            group.cancelAll()
        }
        return read.bytes
    }

    /// A body that is not mold's JSON envelope, when it is short enough to be
    /// a sentence rather than a proxy's HTML page.
    static func plainMessage(_ data: Data) -> String? {
        guard !data.isEmpty, data.count <= 400,
              let text = String(data: data, encoding: .utf8)?
                  .trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty, !text.hasPrefix("<")
        else { return nil }
        return text
    }
}
