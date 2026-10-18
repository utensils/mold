import Foundation

/// An observation about missing completed work, never a terminal job state.
public struct StallDiagnostic: Codable, Hashable, Sendable {
    public let stage: String
    public let lastCompletedWorkUnixMs: UInt64
    public let elapsedStallMs: UInt64
    public let cancelRequested: Bool
    public let deviceStillHeld: Bool

    public init(stage: String, lastCompletedWorkUnixMs: UInt64, elapsedStallMs: UInt64,
                cancelRequested: Bool, deviceStillHeld: Bool) {
        self.stage = stage
        self.lastCompletedWorkUnixMs = lastCompletedWorkUnixMs
        self.elapsedStallMs = elapsedStallMs
        self.cancelRequested = cancelRequested
        self.deviceStillHeld = deviceStillHeld
    }

    public var statusMessage: String {
        let ownership = deviceStillHeld ? "device still held" : "inspect host diagnostics"
        let intent = cancelRequested ? "Cancel requested; " : ""
        return "Stalled: \(stage) (\(elapsedStallMs / 1_000)s without completed work). \(intent)\(ownership)."
    }

    public func copyText(machine: String, job: String) -> String {
        """
        Machine: \(machine)
        Job: \(job)
        Stage: \(stage)
        Last completed work (Unix ms): \(lastCompletedWorkUnixMs)
        No completed work for (ms): \(elapsedStallMs)
        Cancel requested: \(cancelRequested)
        Device still held: \(deviceStillHeld)

        \(recoveryMessage)
        """
    }

    public var recoveryMessage: String {
        "Cancellation stops at the next safe boundary. Inspect host diagnostics and account for other jobs before an operator-controlled restart. A stalled kernel may not return automatically."
    }
}

/// A successful request may still be waiting on a live device owner.
public enum QueueCancellationDisposition: Hashable, Sendable {
    case cancelled
    case pending(stall: StallDiagnostic?)
    /// A legacy/custom backend accepted intent without terminal proof.
    case unconfirmed
}
