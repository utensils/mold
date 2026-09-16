import Foundation

public enum MoldClientError: Error, Sendable, LocalizedError {
    case unreachable(String)
    case unauthorized
    case http(status: Int, code: String?, message: String?)
    case malformedResponse

    /// Whether sending the same request again could plausibly work.
    ///
    /// Retrying something that cannot succeed is not resilience, it is a
    /// spinner that never stops -- so this is deliberately narrow: the link
    /// being down, the machine being too busy, and the machine having a bad
    /// minute. A missing key does not appear by waiting, a refused request
    /// stays refused, and a reply this build cannot parse will not parse on
    /// the next attempt either.
    public var isTransient: Bool {
        switch self {
        case .unreachable: true
        case .unauthorized: false
        case let .http(status, _, _): status >= 500 || status == 429
        case .malformedResponse: false
        }
    }

    public var errorDescription: String? {
        switch self {
        case let .unreachable(reason):
            "Couldn't reach this machine. \(reason)"
        case .unauthorized:
            "This machine needs an API key. Add one in Settings."
        case let .http(status, _, message):
            message ?? "The machine answered with an error (\(status))."
        case .malformedResponse:
            "The machine sent something this version of Mold can't read."
        }
    }
}
