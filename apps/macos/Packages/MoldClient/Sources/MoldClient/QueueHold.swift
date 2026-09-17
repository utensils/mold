import Foundation

/// Why a machine parked a job, and whether this app can do anything about it.
public enum QueueHold: Hashable, Sendable {
    /// `MODEL_NOT_FOUND` or `UNKNOWN_MODEL` -- the ONLY two codes the server
    /// maps to a resolvable hold (`durable_admission_authority.rs:9-22`). The
    /// model is the ROW's own `model` field; the sentence is never parsed for
    /// it.
    case missingModel(String, sentence: String)
    /// Everything else the machine said. `retryable` is the machine's answer
    /// to "would trying again help" -- `false` means it needs repair, and a
    /// Retry button there would just hold the job again
    /// (`routes.rs:7651-7655`).
    case prose(String, retryable: Bool)

    /// `nil` unless `entry` is held. `child` is `nil` on a host that answered
    /// no batch status, or on a row with no batch at all -- then the row's
    /// own sentence is all there is, which is exactly the state the pane has
    /// shipped in since M1. `error_code` lives only on the batch child, and
    /// only while it is held (`routes.rs:2951-2956`); `QueueEntry` carries no
    /// such field at all (`types.rs:4393-4462`).
    public static func resolve(entry: QueueEntry, child: BatchChild?) -> QueueHold? {
        guard entry.state == .held else { return nil }
        // The two fields are the same string under two names
        // (`types.rs:4432-4438`); `heldReason` is preferred because that is
        // the field the state is named for.
        let sentence = entry.heldReason ?? entry.error ?? ""
        if let model = entry.model, let code = child?.errorCode,
           code == "MODEL_NOT_FOUND" || code == "UNKNOWN_MODEL"
        {
            return .missingModel(model, sentence: sentence)
        }
        let retryable = child?.retryable ?? entry.retryable ?? true
        return .prose(sentence, retryable: retryable)
    }
}
