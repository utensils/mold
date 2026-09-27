import Foundation

/// Why a secret could not be read or written. Surfaced, never swallowed: the
/// Keychain's discarded `OSStatus` is exactly how a whole fleet's keys went
/// missing without anybody being told.
///
/// A case carries a NAME and a PATH, never a value.
public enum SecretStoreError: Error, Equatable, LocalizedError {
    case unknownName(String)
    case couldNotReplace(path: String, code: Int32)
    /// The file is THERE and will not read. Never "there are no keys": every
    /// read and every write refuses until it reads again, so nothing is
    /// written over what could not be looked at (review E2).
    case unreadable(path: String, reason: String)

    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case let .unknownName(name):
            "\(name) is not a credential this app keeps."
        case let .couldNotReplace(path, code):
            "Couldn't write \(path): \(String(cString: strerror(code)))."
        case let .unreadable(path, reason):
            "Couldn't read \(path): \(reason) Nothing was changed."
        }
    }
}
