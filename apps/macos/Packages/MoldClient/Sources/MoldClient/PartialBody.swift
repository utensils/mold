import Foundation

/// Bytes a read had collected when it was cut short.
///
/// `refusalBody` races a read against a deadline, and the read's own return
/// value is lost when the deadline wins -- so what it managed to collect
/// lands here instead. A refusal body is usually complete long before the
/// deadline; this exists for the case where the connection is held open and
/// the answer is "what arrived is what we have".
///
/// A lock rather than an actor: `withTaskGroup`'s two children touch it from
/// two tasks, and every operation is a handful of instructions with nothing
/// to await.
final class PartialBody: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = Data()

    /// Appends one byte and answers the new length, so a caller can stop at a
    /// ceiling without a second lock acquisition.
    @discardableResult
    func append(_ byte: UInt8) -> Int {
        lock.lock()
        defer { lock.unlock() }
        storage.append(byte)
        return storage.count
    }

    var bytes: Data {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}
