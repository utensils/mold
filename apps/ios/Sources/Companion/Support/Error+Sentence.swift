import Foundation
import MoldClient

/// What to tell a person about a failure: what did not happen is the caller's
/// clause, then the machine's own reason, then the way forward. The Mac's
/// `Error+Sentence.swift` with the phone's places in its advice -- "under
/// Machines" is where a key is added here, not "in Settings" -- which is why
/// it lives in the app and not in MoldClient.
extension Error {
    var sentence: String {
        (self as? LocalizedError)?.errorDescription ?? localizedDescription
    }

    /// The failure's own clause, lowercased to follow "workstation couldn't…".
    var reason: String {
        guard let clientError = self as? MoldClientError else {
            return lowercasingFirstLetter(of: sentence)
        }
        switch clientError {
        case let .unreachable(reason):
            return lowercasingFirstLetter(of: reason)
        case .unauthorized:
            return "it needs an API key."
        case let .http(status, _, message):
            return message.map(lowercasingFirstLetter(of:)) ?? "it answered with an error (\(status))."
        case .malformedResponse:
            return "it answered something this version of Mold Studio can't read."
        case let .licenseRequired(refusal, mismatch):
            return mismatch
                ? lowercasingFirstLetter(of: "\(refusal.name) pins different terms on this machine.")
                : lowercasingFirstLetter(of: "\(refusal.name) has to be accepted on this machine first.")
        }
    }

    /// `reason`, standing on its own.
    var reasonSentence: String {
        let reason = reason
        guard let first = reason.first else { return reason }
        return first.uppercased() + reason.dropFirst()
    }

    /// The way forward, where there is one. `nil` rather than an offer that
    /// cannot help.
    var advice: String? {
        guard let clientError = self as? MoldClientError else { return nil }
        switch clientError {
        case .unreachable: return "Check the machine under Machines."
        case .unauthorized: return "Add its key under Machines."
        case .http: return clientError.isTransient ? "Try again in a moment." : nil
        case .malformedResponse: return "Update Mold Studio, or mold on that machine."
        case .licenseRequired: return "Accept the terms under Models."
        }
    }

    /// The reason, then the way forward: for a surface that would otherwise
    /// be a dead end.
    var failureSentence: String {
        [reasonSentence, advice].compactMap { $0 }.joined(separator: " ")
    }
}

private func lowercasingFirstLetter(of string: String) -> String {
    guard let first = string.first else { return string }
    return first.lowercased() + string.dropFirst()
}
