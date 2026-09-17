import Foundation
import MoldClient

// Stopping a chain, and the two small writes the follow loop makes. Split from
// `ChainRun`'s own shape purely for size.
@MainActor
extension ChainRun {
    /// Stops the job on its own machine. `false` means there was nothing to
    /// stop, which is the caller's cue to stop whatever else is on screen.
    @discardableResult
    func stop(backend: (MoldHost.ID) -> (any MoldBackend)?) -> Bool {
        if creating {
            // Nothing to cancel YET. The landing does it.
            withdrawn = true
            active = nil
            return true
        }
        guard let active, let host else { return false }
        task?.cancel()
        task = nil
        self.active = nil
        PendingChain.forget(active.jobId)
        // Cancelled by the user, not lost: nothing to recover on relaunch.
        guard let backend = backend(host) else { return true }
        Task { try? await backend.cancelChainJob(id: active.jobId) }
        return true
    }

    /// Not `private`: `ChainRun+Follow` is the event loop, in its own file for
    /// size.
    func update(_ transform: (inout ChainProgress) -> Void, report: ChainRun.Reporter) {
        guard var progress = active else { return }
        transform(&progress)
        active = progress
        report.progress(progress)
    }

    func settle() {
        task = nil
        if let active { PendingChain.forget(active.jobId) }
        active = nil
    }
}
