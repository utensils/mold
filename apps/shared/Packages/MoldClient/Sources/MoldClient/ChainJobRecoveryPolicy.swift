import Foundation

/// Existing chain lifecycle authority; the server still rechecks every gesture.
/// Restart/discard offers additionally require their explicit server options.
public enum ChainJobRecoveryPolicy {
    public static func canCancel(state: ChainJobState) -> Bool {
        state == .queued || state == .running || state == .paused
    }
    public static func canResume(state: ChainJobState, ephemeral: Bool) -> Bool {
        state == .paused || state == .interrupted
            || (!ephemeral && (state == .failed || state == .cancelled))
    }
}
