import Foundation

/// Why a machine parked a job, and whether this app can do anything about it.
public enum QueueHold: Hashable, Sendable {
    /// A server-verified missing downloadable component, or legacy typed
    /// `MODEL_NOT_FOUND` / `UNKNOWN_MODEL`. The model always matches the row;
    /// prose is never parsed to choose a download.
    case missingModel(String, sentence: String)
    /// Everything else the machine said. `retryable` is the machine's answer
    /// to "would trying again help" -- `false` means it needs repair, and a
    /// Retry button there would just hold the job again
    /// (`routes.rs:7651-7655`).
    case prose(String, retryable: Bool)

    public func summary(modelName: String, hostName: String) -> String {
        switch self {
        case let .missingModel(_, sentence):
            sentence.isEmpty ? "\(modelName) isn’t installed on \(hostName)." : UserFacingError.message(sentence)
        case .prose(let sentence, _): sentence.isEmpty ? "The machine put this job aside." : UserFacingError.message(sentence)
        }
    }

    /// `nil` unless `entry` is held. `child` is `nil` on a host that answered
    /// no batch status, or on a row with no batch at all -- then the row's
    /// own sentence is all there is, which is exactly the state the pane has
    /// shipped in since M1. New hosts send a manifest-backed `model_download`
    /// hint on the row. Only an absent hint permits the older batch-code fallback.
    public static func resolve(entry: QueueEntry, child: BatchChild?) -> QueueHold? {
        guard entry.state == .held else { return nil }
        // The two fields are the same string under two names
        // (`types.rs:4432-4438`); `heldReason` is preferred because that is
        // the field the state is named for.
        let sentence = entry.heldReason ?? entry.error ?? ""
        let retryable = child?.retryable != false && entry.retryable != false
        if let hint = entry.modelDownload {
            guard entry.retryable == true, retryable, let model = hint.model, model == entry.model,
                  !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return .prose(sentence, retryable: retryable)
            }
            return .missingModel(model, sentence: sentence)
        }
        if retryable, let model = entry.model, !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           let code = child?.errorCode,
           code == "MODEL_NOT_FOUND" || code == "UNKNOWN_MODEL"
        {
            return .missingModel(model, sentence: sentence)
        }
        return .prose(sentence, retryable: retryable)
    }
}
