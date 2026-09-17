import Foundation
import MoldClient

/// What to tell a person about a failure. Deliberately short -- it is read at
/// every site `HostStore.report` is called from, not written out as a stack
/// trace.
///
/// Internal, and app-side: MoldClient never needs to turn an error into
/// prose for a person, and a public extension on `Error` in a shared package
/// is a global nobody asked for.
extension Error {
    var sentence: String {
        (self as? LocalizedError)?.errorDescription ?? localizedDescription
    }

    /// The failure's own clause, with the "couldn't reach this machine" /
    /// "the machine answered" preamble stripped -- `HostStore.report` already
    /// made the machine the sentence's subject, so repeating that here would
    /// say it twice.
    var reason: String {
        guard let clientError = self as? MoldClientError else {
            return lowercasingFirstLetter(of: sentence)
        }
        switch clientError {
        case let .unreachable(reason):
            return lowercasingFirstLetter(of: reason)
        case .unauthorized:
            return "it needs an API key. Add one in Settings."
        case let .http(status, _, message):
            return message.map(lowercasingFirstLetter(of:)) ?? "it answered with an error (\(status))."
        case .malformedResponse:
            return "it answered something this version of Mold can't read."
        }
    }
}

private func lowercasingFirstLetter(of string: String) -> String {
    guard let first = string.first else { return string }
    return first.lowercased() + string.dropFirst()
}
