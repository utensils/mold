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
            // The key is the REASON; adding one is the way forward, and that
            // half lives in `advice` so nothing says it twice.
            return "it needs an API key."
        case let .http(status, _, message):
            return message.map(lowercasingFirstLetter(of:)) ?? "it answered with an error (\(status))."
        case .malformedResponse:
            return "it answered something this version of Mold can't read."
        case let .licenseRequired(refusal, mismatch):
            return mismatch
                ? lowercasingFirstLetter(of: "\(refusal.name) pins different terms on this machine.")
                : lowercasingFirstLetter(of: "\(refusal.name) has to be accepted on this machine first.")
        }
    }
}

extension Error {
    /// `reason`, standing on its own -- for a place with no machine-as-subject
    /// clause ahead of it, such as the popover a failed rewrite answers in.
    var reasonSentence: String {
        let reason = reason
        guard let first = reason.first else { return reason }
        return first.uppercased() + reason.dropFirst()
    }

    /// The way forward, where this failure has one.
    ///
    /// The third part of what a person reads: what did not happen, the
    /// machine's own reason for it, and what to do about it. A failure with
    /// no route out of it is a dead end, which is what "our error handling is
    /// kind of lacking" meant -- so every route is worded HERE, once, and no
    /// surface carries a second copy. `nil` where there is honestly nothing
    /// to do; an offer that cannot help is worse than none.
    ///
    /// Whether waiting could work is asked of `MoldClientError.isTransient`
    /// rather than re-decided here. An unreachable machine is transient too,
    /// but it has something better to say than "try again".
    var advice: String? {
        guard let clientError = self as? MoldClientError else { return nil }
        switch clientError {
        case .unreachable:
            return "Check the machine under Machines."
        case .unauthorized:
            return "Add one in Settings."
        case .http:
            return clientError.isTransient ? "Try again in a moment." : nil
        case .malformedResponse:
            return "Update Mold here, or on that machine."
        case .licenseRequired:
            return "Accept the terms under Models."
        }
    }

    /// The whole thing, for a surface that would otherwise be a dead end: the
    /// machine's own reason, then the way forward.
    ///
    /// Not what a compact STATUS line reads -- a machine row in the sidebar
    /// is already sitting under Machines, and telling it to go there would be
    /// noise. Those keep `reasonSentence`.
    var failureSentence: String {
        [reasonSentence, advice].compactMap { $0 }.joined(separator: " ")
    }
}

private func lowercasingFirstLetter(of string: String) -> String {
    guard let first = string.first else { return string }
    return first.lowercased() + string.dropFirst()
}
